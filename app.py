import os
import uuid
import subprocess
import threading
import json
import shutil
import glob
import time
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from redis_client import get_redis, close_redis
from job_store import RedisJobStore
from models import (
    JobData, JobStatus, CaptionSettings, CaptionStyleEnum,
    JobResult, ClipResult, ProcessResponseV2, JobStatusResponse, JobResultResponse
)

# Import new modules for editor and subtitle features
import editor
import subtitles

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 500  # 500 MB limit
JOB_RETENTION_SECONDS = 3600  # 1 hour retention

# Application State
# Semaphore to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# V2 API State (Redis-backed)
job_queue_v2 = asyncio.Queue()
# Store API keys in memory (not in Redis for security)
job_api_keys: Dict[str, str] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start v2 worker if Redis available
    redis = await get_redis()
    v2_worker_task = None
    if redis:
        v2_worker_task = asyncio.create_task(process_queue_v2())
        print("✅ Redis connected, V2 API enabled")
    else:
        print("⚠️ No REDIS_URL configured, V2 API disabled")

    yield

    # Cleanup
    await close_redis()

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

class ProcessRequest(BaseModel):
    url: str

class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None

class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom"  # top, middle, bottom
    font_size: int = 16
    input_filename: Optional[str] = None

# ============= V2 API (Redis-backed) =============

async def process_queue_v2():
    """Background worker for v2 jobs with Redis persistence."""
    print(f"v2 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            job_id = await job_queue_v2.get()
            await concurrency_semaphore.acquire()
            print(f"v2 Acquired slot for job: {job_id}")
            asyncio.create_task(run_job_v2_wrapper(job_id))
        except Exception as e:
            print(f"v2 Queue dispatch error: {e}")
            await asyncio.sleep(1)


async def run_job_v2_wrapper(job_id: str):
    """Wrapper for v2 job execution."""
    try:
        redis = await get_redis()
        if not redis:
            return

        store = RedisJobStore(redis)
        job = await store.get_job(job_id)

        if job:
            await run_job_v2(job_id, job, store)
    except Exception as e:
        print(f"v2 Job wrapper error {job_id}: {e}")
    finally:
        concurrency_semaphore.release()
        job_queue_v2.task_done()
        print(f"v2 Released slot for job: {job_id}")


def parse_progress(log_line: str) -> Optional[Tuple[int, str]]:
    """Parse progress from log line. Returns (percentage, stage) or None."""
    line_lower = log_line.lower()
    if "downloading" in line_lower:
        return (10, "Downloading video")
    if "transcribing" in line_lower:
        return (30, "Transcribing audio")
    if "analyzing" in line_lower or "gemini" in line_lower:
        return (50, "AI analysis")
    if "processing clip" in line_lower or "extracting" in line_lower:
        return (70, "Creating clips")
    if "clip saved" in line_lower or "saved to" in line_lower:
        return (90, "Finalizing")
    return None


def enqueue_output_v2(out, job_id: str, store: RedisJobStore, loop):
    """Reads output from subprocess and updates Redis."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"v2 [Job Output] {decoded_line}")
                # Schedule async Redis update
                asyncio.run_coroutine_threadsafe(
                    store.append_log(job_id, decoded_line),
                    loop
                )
                # Parse and update progress
                progress = parse_progress(decoded_line)
                if progress:
                    asyncio.run_coroutine_threadsafe(
                        store.update_progress(job_id, progress[0], progress[1]),
                        loop
                    )
    except Exception as e:
        print(f"v2 Error reading output for job {job_id}: {e}")
    finally:
        out.close()


async def check_partial_results_v2(job_id: str, output_dir: str, store: RedisJobStore):
    """Check for partial results during processing."""
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        return

    try:
        target_json = json_files[0]
        if os.path.getsize(target_json) == 0:
            return

        with open(target_json, 'r') as f:
            data = json.load(f)

        base_name = os.path.basename(target_json).replace('_metadata.json', '')
        clips = data.get('shorts', [])
        ready_clips = []

        for i, clip in enumerate(clips):
            clip_filename = f"{base_name}_clip_{i+1}.mp4"
            clip_path = os.path.join(output_dir, clip_filename)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                ready_clips.append(ClipResult(
                    video_url=f"/videos/{job_id}/{clip_filename}",
                    title=clip.get('video_title_for_youtube_short'),
                    description_tiktok=clip.get('video_description_for_tiktok'),
                    description_instagram=clip.get('video_description_for_instagram'),
                    description_youtube=clip.get('video_title_for_youtube_short')
                ))

        if ready_clips:
            await store.set_result(job_id, JobResult(clips=ready_clips))
    except Exception:
        pass


async def finalize_job_v2(job_id: str, output_dir: str, store: RedisJobStore):
    """Finalize completed job."""
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
        clips = data.get('shorts', [])
        result_clips = []

        for i, clip in enumerate(clips):
            clip_filename = f"{base_name}_clip_{i+1}.mp4"
            result_clips.append(ClipResult(
                video_url=f"/videos/{job_id}/{clip_filename}",
                title=clip.get('video_title_for_youtube_short'),
                description_tiktok=clip.get('video_description_for_tiktok'),
                description_instagram=clip.get('video_description_for_instagram'),
                description_youtube=clip.get('video_title_for_youtube_short')
            ))

        # Store transcript for editor/subtitle features
        transcript = data.get('transcript')
        await store.set_result(job_id, JobResult(clips=result_clips, transcript=transcript))
        await store.set_status(job_id, JobStatus.COMPLETED)
    else:
        await store.set_status(job_id, JobStatus.FAILED, "No metadata file generated")


async def run_job_v2(job_id: str, job_data: JobData, store: RedisJobStore):
    """Execute v2 job with Redis progress tracking."""
    await store.set_status(job_id, JobStatus.PROCESSING)
    await store.append_log(job_id, "Job started by worker.")

    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    # Build command
    cmd = ["python", "-u", "main.py", "-u", job_data.input_url, "-o", job_output_dir]

    if job_data.caption_settings.include_captions:
        style = job_data.caption_settings.style
        if style and style != CaptionStyleEnum.NONE:
            cmd.extend(["--caption-style", style.value])
        if job_data.caption_settings.color:
            cmd.extend(["--caption-color", job_data.caption_settings.color])
        if job_data.caption_settings.outline_color:
            cmd.extend(["--caption-outline-color", job_data.caption_settings.outline_color])

    env = os.environ.copy()
    # Get API key from in-memory store
    if job_id in job_api_keys:
        env["GEMINI_API_KEY"] = job_api_keys[job_id]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )

        # Log reader thread
        loop = asyncio.get_event_loop()
        t_log = threading.Thread(
            target=enqueue_output_v2,
            args=(process.stdout, job_id, store, loop),
            daemon=True
        )
        t_log.start()

        # Wait for completion with incremental updates
        while process.poll() is None:
            await asyncio.sleep(2)
            await check_partial_results_v2(job_id, job_output_dir, store)

        if process.returncode == 0:
            await finalize_job_v2(job_id, job_output_dir, store)
        else:
            await store.set_status(
                job_id, JobStatus.FAILED,
                f"Process failed with exit code {process.returncode}"
            )

    except Exception as e:
        await store.set_status(job_id, JobStatus.FAILED, str(e))
    finally:
        # Clean up API key from memory
        if job_id in job_api_keys:
            del job_api_keys[job_id]


@app.post("/api/v2/process", response_model=ProcessResponseV2)
async def process_v2(
    request: Request,
    url: str = Query(..., description="Video URL or YouTube link"),
    include_captions: bool = Query(True, description="Include captions in output"),
    caption_style: str = Query("none", description="Caption style"),
    caption_color: Optional[str] = Query(None, description="Hex color for caption text"),
    caption_outline_color: Optional[str] = Query(None, description="Hex color for caption outline")
):
    """Submit a video for processing (v2 with Redis persistence)."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(
            status_code=503,
            detail="v2 API requires Redis. Set REDIS_URL environment variable."
        )

    # Get API key from header, fall back to environment variable
    api_key = request.headers.get("X-Gemini-Key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing Gemini API key. Provide X-Gemini-Key header or set GEMINI_API_KEY env var."
        )

    # Validate caption style
    valid_styles = [e.value for e in CaptionStyleEnum]
    if caption_style not in valid_styles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid caption_style. Must be one of: {valid_styles}"
        )

    job_id = str(uuid.uuid4())
    store = RedisJobStore(redis)

    job = JobData(
        job_id=job_id,
        status=JobStatus.QUEUED,
        input_url=url,
        caption_settings=CaptionSettings(
            include_captions=include_captions,
            style=CaptionStyleEnum(caption_style),
            color=caption_color,
            outline_color=caption_outline_color
        ),
        created_at=datetime.utcnow(),
        logs=[f"Job {job_id} queued."]
    )

    await store.create_job(job)

    # Store API key in memory (not in Redis for security)
    job_api_keys[job_id] = api_key

    await job_queue_v2.put(job_id)

    return ProcessResponseV2(job_id=job_id, status="queued")


@app.get("/api/v2/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status_v2(job_id: str):
    """Get job status and progress."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="v2 API requires Redis")

    store = RedisJobStore(redis)
    job = await store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress_percentage=job.progress_percentage,
        progress_stage=job.progress_stage,
        logs=job.logs,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        error=job.error
    )


@app.get("/api/v2/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result_v2(job_id: str):
    """Get job result (completed videos and metadata)."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="v2 API requires Redis")

    store = RedisJobStore(redis)
    job = await store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status.value,
        result=job.result,
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )


@app.post("/api/v2/edit")
async def edit_clip_v2(
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    """Edit a clip with AI-generated enhancements using Gemini."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="v2 API requires Redis")

    store = RedisJobStore(redis)
    job = await store.get_job(req.job_id)

    if not job or not job.result or not job.result.clips:
        raise HTTPException(status_code=404, detail="Job not found or no clips available")

    # Get clip
    try:
        clip = job.result.clips[req.clip_index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Clip index {req.clip_index} not found")

    # Get video file path
    filename = clip.video_url.split('/')[-1]
    input_path = os.path.join(OUTPUT_DIR, req.job_id, req.input_filename or filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # Get API key
    api_key = req.api_key or x_gemini_key or job_api_keys.get(req.job_id)
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key required")

    # Get transcript
    if not job.result.transcript:
        raise HTTPException(status_code=400, detail="No transcript available for this job")

    try:
        # Run editor in thread pool (blocking I/O)
        loop = asyncio.get_event_loop()
        edited_path = await loop.run_in_executor(
            None,
            editor.VideoEditor(api_key).edit_video,
            input_path,
            job.result.transcript
        )

        # Add edited video to job result
        edited_filename = os.path.basename(edited_path)
        new_url = f"/videos/{req.job_id}/{edited_filename}"

        return {"success": True, "edited_video_url": new_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")


@app.post("/api/v2/subtitle")
async def add_subtitles_v2(req: SubtitleRequest):
    """Add word-level subtitles to a clip."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="v2 API requires Redis")

    store = RedisJobStore(redis)
    job = await store.get_job(req.job_id)

    if not job or not job.result or not job.result.clips:
        raise HTTPException(status_code=404, detail="Job not found or no clips available")

    # Get clip
    try:
        clip = job.result.clips[req.clip_index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Clip index {req.clip_index} not found")

    # Get video file path
    filename = clip.video_url.split('/')[-1]
    input_path = os.path.join(OUTPUT_DIR, req.job_id, req.input_filename or filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # Get transcript
    if not job.result.transcript:
        raise HTTPException(status_code=400, detail="No transcript available for this job")

    try:
        # Generate SRT and burn subtitles (blocking I/O)
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            None,
            subtitles.add_subtitles_to_video,
            input_path,
            job.result.transcript,
            req.position,
            req.font_size
        )

        # Return subtitled video URL
        subtitled_filename = os.path.basename(output_path)
        new_url = f"/videos/{req.job_id}/{subtitled_filename}"

        return {"success": True, "subtitled_video_url": new_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subtitle failed: {str(e)}")


# ============= Social Media Integration =============

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: str
    user_id: str
    platforms: List[str] # ["tiktok", "instagram", "youtube"]
    # Optional overrides if frontend wants to edit them
    title: Optional[str] = None
    tiktok_description: Optional[str] = None
    instagram_description: Optional[str] = None
    youtube_description: Optional[str] = None

import httpx

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest):
    # Get job from V2 Redis store
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="Redis not available")

    store = RedisJobStore(redis)
    job_data = await store.get_job(req.job_id)

    if not job_data or not job_data.result:
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    if not job_data.result.clips:
        raise HTTPException(status_code=400, detail="Job result not available")

    try:
        clip = job_data.result.clips[req.clip_index]
        # Video URL is relative /videos/..., we need absolute file path
        # clip.video_url is like "/videos/{job_id}/{filename}"
        # We constructed it as: f"/videos/{job_id}/{clip_filename}"
        # And file is at f"{OUTPUT_DIR}/{job_id}/{clip_filename}"

        filename = clip.video_url.split('/')[-1]
        file_path = os.path.join(OUTPUT_DIR, req.job_id, filename)

        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

        # Construct parameters for Upload-Post API
        # Fallbacks
        final_title = req.title or clip.title or 'Viral Short'
        
        # Prepare form data
        url = "https://api.upload-post.com/api/upload"
        headers = {
            "Authorization": f"Apikey {req.api_key}"
        }
        
        # Prepare data as dict (httpx handles lists for multiple values)
        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms # Pass list directly
        }
        
        # Add Platform specifics
        if "tiktok" in req.platforms:
             desc = req.tiktok_description or clip.description_tiktok or final_title
             data_payload["tiktok_title"] = desc

        if "instagram" in req.platforms:
             desc = req.instagram_description or clip.description_instagram or final_title
             data_payload["instagram_title"] = desc
             data_payload["media_type"] = "REELS"

        if "youtube" in req.platforms:
             yt_title = req.title or clip.title or final_title
             desc = req.youtube_description or clip.description_youtube or clip.description_instagram or final_title
             data_payload["youtube_title"] = yt_title
             data_payload["youtube_description"] = desc
             data_payload["privacyStatus"] = "public"

        # Send File
        # httpx AsyncClient requires async file reading or bytes. 
        # Since we have MAX_FILE_SIZE_MB, reading into memory is safe-ish.
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        files = {
            "video": (filename, file_content, "video/mp4")
        }

        # Switch to synchronous Client to avoid "sync request with AsyncClient" error with multipart/files
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending to Upload-Post for platforms: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)
            
        if response.status_code not in [200, 201, 202]: # Added 201
             print(f"❌ Upload-Post Error: {response.text}")
             raise HTTPException(status_code=response.status_code, detail=f"Vendor API Error: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/user")
async def get_social_user(api_key: str = Header(..., alias="X-Upload-Post-Key")):
    """Proxy to fetch user ID from Upload-Post"""
    if not api_key:
         raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")
         
    url = "https://api.upload-post.com/api/uploadposts/users"
    print(f"🔍 Fetching User ID from: {url}")
    headers = {"Authorization": f"Apikey {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Upload-Post User Fetch Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch user: {resp.text}")
            
            data = resp.json()
            print(f"🔍 Upload-Post User Response: {data}")
            
            user_id = None
            # The structure is {'success': True, 'profiles': [{'username': '...'}, ...]}
            profiles_list = []
            if isinstance(data, dict):
                 raw_profiles = data.get('profiles', [])
                 if isinstance(raw_profiles, list):
                     for p in raw_profiles:
                         username = p.get('username')
                         if username:
                             # Determine connected platforms
                             socials = p.get('social_accounts', {})
                             connected = []
                             # Check typical platforms
                             for platform in ['tiktok', 'instagram', 'youtube']:
                                 account_info = socials.get(platform)
                                 # If it's a dict and typically has data, or just not empty string
                                 if isinstance(account_info, dict):
                                     connected.append(platform)
                             
                             profiles_list.append({
                                 "username": username,
                                 "connected": connected
                             })
            
            if not profiles_list:
                # Fallback if no profiles found
                return {"profiles": [], "error": "No profiles found"}
                
            return {"profiles": profiles_list}
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Transcription endpoint — accepts a YouTube URL (or direct video URL) and
# returns the transcript text. Used by the podcast pipeline.
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    url: str

class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float

def _download_video(url: str, output_dir: str) -> str:
    """Download video via yt-dlp (works for YouTube and direct URLs)."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or '').lower()
    is_youtube = any(h in hostname for h in ('youtube.com', 'youtu.be', 'youtube-nocookie.com'))

    if not is_youtube:
        # Direct URL download
        import urllib.request
        ext = os.path.splitext(parsed.path)[1] or '.mp4'
        out = os.path.join(output_dir, f"video{ext}")
        urllib.request.urlretrieve(url, out)
        return out

    # YouTube download via yt-dlp
    output_template = os.path.join(output_dir, "video.%(ext)s")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav",
        "-o", output_template, "--no-playlist", url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("video"):
            return os.path.join(output_dir, f)
    raise RuntimeError("yt-dlp produced no output file")


def _transcribe(video_path: str) -> dict:
    """Transcribe a local audio/video file with Faster-Whisper."""
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, word_timestamps=False)
    parts = []
    total_duration = 0.0
    for seg in segments:
        parts.append(seg.text.strip())
        total_duration = seg.end
    text = " ".join(parts)
    return {"text": text, "language": info.language, "duration": total_duration}


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_url(req: TranscribeRequest):
    """Download a video (YouTube or direct URL) and return its transcript."""
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="transcribe_")
    try:
        video_path = await asyncio.to_thread(_download_video, req.url, tmp_dir)
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail="Failed to download video")

        result = await asyncio.to_thread(_transcribe, video_path)

        return TranscribeResponse(
            text=result["text"],
            language=result.get("language", "en"),
            duration_seconds=result.get("duration", 0.0),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
