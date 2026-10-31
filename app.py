import os
import re
import sys
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
import cleaner
import youtube_download
import user_errors
from url_utils import build_video_url, video_filename_from_url

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
def _env_int(name, default):
    try:
        return int(float(os.environ.get(name, "") or default))
    except ValueError:
        return default


MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 500)  # also enforced by main.py downloads
# output/<job_id> (clips, metadata, any leftover source) is deleted this long
# after its last write. The Redis job record expires after 24h on its own.
JOB_RETENTION_SECONDS = _env_int("JOB_RETENTION_SECONDS", 48 * 3600)
JOB_CLEANUP_INTERVAL_SECONDS = _env_int("JOB_CLEANUP_INTERVAL_SECONDS", 1800)
# A job whose main.py is still running after this is killed and failed.
JOB_TIMEOUT_SECONDS = _env_int("JOB_TIMEOUT_SECONDS", 3 * 3600)
# Reject sources shorter than this before starting (0 disables). A 24s Short
# cannot yield 15-60s clips; main.py enforces the same floor after download
# for YouTube links, which cannot be probed here.
MIN_SOURCE_SECONDS = int(os.environ.get("MIN_SOURCE_SECONDS", "45"))

# Application State
# Semaphore to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# V2 API State (Redis-backed)
job_queue_v2 = asyncio.Queue()
# Store API keys in memory (not in Redis for security)
job_api_keys: Dict[str, str] = {}
RENDER_FAILED_MESSAGE = ("We picked clips from this video but couldn't render any of them. "
                         "Please try again; if it keeps failing, re-export the video as MP4 "
                         "(H.264) and upload it.")
# User-facing failure reason per job, from main.py's "❌ CODE: message" lines.
job_fail_reasons: Dict[str, str] = {}
_FAIL_LINE = re.compile(r"❌ ([A-Z_]{3,}): (.+)")

TIMEOUT_MESSAGE = ("This video took too long to process and the job was stopped. "
                   "Try a shorter video, or try again later.")
# Jobs whose main.py is running right now (never swept by the cleanup loop).
active_job_ids: set = set()
_CLIP_FILE_RE = re.compile(r"_clip_\d+(?:_[A-Za-z0-9]+)*\.mp4$")


def _newest_mtime(path: str) -> float:
    newest = os.path.getmtime(path)
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass
    return newest


def sweep_expired_outputs(now: Optional[float] = None, retention: Optional[int] = None,
                          base_dirs=(OUTPUT_DIR, UPLOAD_DIR), active=None) -> int:
    """Delete job dirs / uploads untouched for ``retention`` seconds.
    Returns how many entries were removed. Active jobs are never touched."""
    now = time.time() if now is None else now
    retention = JOB_RETENTION_SECONDS if retention is None else retention
    active = active_job_ids if active is None else active
    removed = 0
    for base in base_dirs:
        try:
            entries = os.listdir(base)
        except OSError:
            continue
        for name in entries:
            if name in active:
                continue
            path = os.path.join(base, name)
            try:
                age = now - (_newest_mtime(path) if os.path.isdir(path) else os.path.getmtime(path))
                if age < retention:
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                removed += 1
            except OSError as e:
                print(f"⚠️ Cleanup could not remove {path}: {e}")
    return removed


async def cleanup_loop():
    """Periodic retention sweep (JOB_RETENTION_SECONDS, default 48h)."""
    while True:
        try:
            removed = await asyncio.to_thread(sweep_expired_outputs)
            if removed:
                print(f"🧹 Removed {removed} expired job output(s)")
        except Exception as e:
            print(f"⚠️ Output cleanup failed: {e}")
        await asyncio.sleep(max(60, JOB_CLEANUP_INTERVAL_SECONDS))


def remove_job_leftovers(output_dir: str) -> None:
    """After a FAILED job: delete everything except published clips (the
    downloaded source, half-rendered temp files). Published clips stay until
    the retention sweep, in case partial results were already handed out."""
    try:
        names = os.listdir(output_dir)
    except OSError:
        return
    for name in names:
        if _CLIP_FILE_RE.search(name) and not name.startswith((".", "temp_", "captioned_")):
            continue
        path = os.path.join(output_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def _kill_process_tree(process) -> None:
    """main.py runs in its own session; take its ffmpeg children down with it."""
    import signal
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


INTERRUPTED_MESSAGE = ("This job was interrupted by a server restart before it finished. "
                       "Please submit the video again.")


async def fail_orphaned_jobs(store: RedisJobStore) -> None:
    """Jobs still queued/processing at startup belonged to the previous
    process (the queue and the main.py subprocesses live in memory), so they
    will never finish. Fail them with a resubmit message instead of leaving
    them "processing" until the 24h TTL."""
    from job_store import JOB_KEY_PREFIX
    failed = 0
    try:
        async for key in store.redis.scan_iter(match=f"{JOB_KEY_PREFIX}*", count=200):
            key = key.decode() if isinstance(key, bytes) else key
            job_id = key[len(JOB_KEY_PREFIX):]
            job = await store.get_job(job_id)
            if job and job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                await store.set_status(job_id, JobStatus.FAILED, INTERRUPTED_MESSAGE)
                failed += 1
    except Exception as e:
        print(f"⚠️ Orphaned-job sweep failed: {e}")
    if failed:
        print(f"⚠️ Marked {failed} job(s) interrupted by the restart as failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start v2 worker if Redis available
    redis = await get_redis()
    v2_worker_task = None
    cleanup_task = asyncio.create_task(cleanup_loop())
    if redis:
        await fail_orphaned_jobs(RedisJobStore(redis))
        v2_worker_task = asyncio.create_task(process_queue_v2())
        print("✅ Redis connected, V2 API enabled")
    else:
        print("⚠️ No REDIS_URL configured, V2 API disabled")

    yield

    # Cleanup
    cleanup_task.cancel()
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

class CleanRequest(BaseModel):
    job_id: str
    clip_index: int = 0
    max_silence_gap: float = 0.5  # seconds; gaps longer than this get trimmed
    silence_pad: float = 0.15     # seconds of silence to leave at each cut edge

class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom"  # top, middle, bottom
    font_size: int = 16  # legacy, ignored: size comes from the preset
    preset: str = "default"  # subtitles.CAPTION_PRESETS id
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
    if "downloading" in line_lower or "savenow" in line_lower or "savefrom" in line_lower:
        return (10, "Downloading video")
    if "transcribing" in line_lower:
        return (30, "Transcribing audio")
    if "analyzing" in line_lower or "gemini" in line_lower:
        return (50, "AI analysis")
    if "processing clip" in line_lower or "extracting" in line_lower:
        return (70, "Creating clips")
    # Only a finished clip means "finalizing" — a bare "saved to" also matched
    # main.py's "Saved metadata to" line, which is printed BEFORE any render.
    if "clip saved to" in line_lower or re.search(r"\bclip \d+ ready\b", line_lower):
        return (90, "Finalizing")
    return None


def enqueue_output_v2(out, job_id: str, store: RedisJobStore, loop):
    """Reads output from subprocess and updates Redis."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"v2 [Job Output] {decoded_line}")
                fail = _FAIL_LINE.search(decoded_line)
                if fail:
                    job_fail_reasons[job_id] = fail.group(2).strip()
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


def _int_or_none(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def clip_result_from_metadata(job_id: str, clip_filename: str, clip: dict) -> ClipResult:
    """One API ClipResult from a metadata.json ``shorts`` entry."""
    title = clip.get('video_title_for_youtube_short')
    # The selector writes no separate YouTube description; the title alone
    # made a poor one. Prefer a real description when one exists.
    youtube_desc = (clip.get('video_description_for_youtube')
                    or clip.get('video_description_for_instagram')
                    or clip.get('video_description_for_tiktok')
                    or title)
    qa = clip.get('qa')
    return ClipResult(
        video_url=build_video_url(job_id, clip_filename),
        title=title,
        description_tiktok=clip.get('video_description_for_tiktok'),
        description_instagram=clip.get('video_description_for_instagram'),
        description_youtube=youtube_desc,
        start=clip.get('start'),
        end=clip.get('end'),
        viral_hook_text=clip.get('viral_hook_text') or None,
        predicted_score=_int_or_none(clip.get('predicted_score')),
        why=clip.get('why') or None,
        qa=qa if isinstance(qa, dict) else None,
        tightening=clip.get('tightening') if isinstance(clip.get('tightening'), dict) else None,
        clip_transcript=(clip.get('clip_transcript')
                         if isinstance(clip.get('clip_transcript'), dict) else None),
    )


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
                ready_clips.append(clip_result_from_metadata(job_id, clip_filename, clip))

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
            clip_path = os.path.join(output_dir, clip_filename)
            # A clip whose render failed has no file: leave it out rather than
            # hand the caller a URL that 404s.
            if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                continue
            result_clips.append(clip_result_from_metadata(job_id, clip_filename, clip))

        if not result_clips:
            await store.set_status(job_id, JobStatus.FAILED, RENDER_FAILED_MESSAGE)
            return

        # Store transcript for editor/subtitle features
        transcript = data.get('transcript')
        await store.set_result(job_id, JobResult(clips=result_clips, transcript=transcript))
        await store.update_progress(job_id, 100, "Completed")
        await store.set_status(job_id, JobStatus.COMPLETED)
    else:
        await store.set_status(job_id, JobStatus.FAILED, user_errors.GENERIC[1])


async def run_job_v2(job_id: str, job_data: JobData, store: RedisJobStore):
    """Execute v2 job with Redis progress tracking."""
    await store.set_status(job_id, JobStatus.PROCESSING)
    await store.update_progress(job_id, 5, "Starting download")
    await store.append_log(job_id, "Job started by worker.")

    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    # Build command
    cmd = [sys.executable, "-u", "main.py", "-u", job_data.input_url, "-o", job_output_dir]

    if job_data.caption_settings.include_captions:
        style = job_data.caption_settings.style
        if style and style != CaptionStyleEnum.NONE:
            cmd.extend(["--caption-style", style.value])
        if job_data.caption_settings.color:
            cmd.extend(["--caption-color", job_data.caption_settings.color])
        if job_data.caption_settings.outline_color:
            cmd.extend(["--caption-outline-color", job_data.caption_settings.outline_color])
    if not job_data.tighten:
        cmd.append("--no-tighten")

    env = os.environ.copy()
    # Get API key from in-memory store
    if job_id in job_api_keys:
        env["GEMINI_API_KEY"] = job_api_keys[job_id]

    try:
        active_job_ids.add(job_id)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd(),
            start_new_session=True,  # own process group, so a timeout kills ffmpeg too
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
        started = time.monotonic()
        timed_out = False
        while process.poll() is None:
            await asyncio.sleep(2)
            if JOB_TIMEOUT_SECONDS > 0 and time.monotonic() - started > JOB_TIMEOUT_SECONDS:
                print(f"v2 Job {job_id} exceeded JOB_TIMEOUT_SECONDS={JOB_TIMEOUT_SECONDS}; killing it")
                _kill_process_tree(process)
                await asyncio.to_thread(process.wait)
                timed_out = True
                break
            await check_partial_results_v2(job_id, job_output_dir, store)

        t_log.join(timeout=5)  # let the reader catch main.py's last lines
        if timed_out:
            await store.append_log(job_id, f"Job stopped after {JOB_TIMEOUT_SECONDS}s (timeout).")
            await store.set_status(job_id, JobStatus.FAILED, TIMEOUT_MESSAGE)
            remove_job_leftovers(job_output_dir)
        elif process.returncode == 0:
            await finalize_job_v2(job_id, job_output_dir, store)
        else:
            print(f"v2 Job {job_id} exited with code {process.returncode}")
            if job_fail_reasons.get(job_id):
                reason = job_fail_reasons[job_id]
            elif process.returncode < 0:
                # Killed by a signal — on Railway that is the OOM killer.
                reason = user_errors.classify("out of memory")[1]
            else:
                reason = user_errors.GENERIC[1]
            await store.set_status(job_id, JobStatus.FAILED, reason)
            remove_job_leftovers(job_output_dir)

    except Exception as e:
        print(f"v2 Job {job_id} runner error: {type(e).__name__}: {e}")
        await store.set_status(job_id, JobStatus.FAILED, user_errors.classify(str(e))[1])
    finally:
        active_job_ids.discard(job_id)
        # Clean up API key from memory
        if job_id in job_api_keys:
            del job_api_keys[job_id]
        job_fail_reasons.pop(job_id, None)
        store.forget(job_id)


def _media_duration_seconds(path_or_url: str) -> float:
    """Container duration via ffprobe (local path or http URL); 0.0 on any
    failure, so the check fails open."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path_or_url],
            capture_output=True, timeout=30)
        return float(proc.stdout.decode().strip() or 0)
    except Exception:
        return 0.0


def _reject_short_source(duration: float):
    raise HTTPException(status_code=400, detail=(
        f"This video is only {int(duration)}s long — clip generation needs at "
        f"least {MIN_SOURCE_SECONDS}s of material to cut from. It already is "
        f"short-form content."))


@app.post("/api/v2/process", response_model=ProcessResponseV2)
async def process_v2(
    request: Request,
    url: str = Query(..., description="Video URL or YouTube link"),
    include_captions: bool = Query(True, description="Include captions in output"),
    caption_style: str = Query("none", description="Caption style"),
    caption_color: Optional[str] = Query(None, description="Hex color for caption text"),
    caption_outline_color: Optional[str] = Query(None, description="Hex color for caption outline"),
    tighten: bool = Query(True, description="Remove fillers, dead air and double takes inside clips"),
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
    if not api_key and not os.environ.get("BASETEN_API_KEY") and not os.environ.get("LLM_BASE_URL"):
        raise HTTPException(
            status_code=400,
            detail="No clip-selection model configured. Set BASETEN_API_KEY or GEMINI_API_KEY, or send X-Gemini-Key."
        )

    # Validate caption style
    valid_styles = [e.value for e in CaptionStyleEnum]
    if caption_style not in valid_styles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid caption_style. Must be one of: {valid_styles}"
        )

    link_problem = user_errors.youtube_link_problem(url)
    if link_problem:
        raise HTTPException(status_code=400, detail=link_problem)

    # Too-short direct sources fail here, before a job is queued. YouTube links
    # are checked by main.py after the download.
    if (MIN_SOURCE_SECONDS > 0 and url.startswith(("http://", "https://"))
            and not youtube_download.is_youtube_url(url)):
        source_duration = await asyncio.to_thread(_media_duration_seconds, url)
        if 0 < source_duration < MIN_SOURCE_SECONDS:
            _reject_short_source(source_duration)

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
        tighten=tighten,
        created_at=datetime.utcnow(),
        logs=[f"Job {job_id} queued."]
    )

    await store.create_job(job)

    # Store API key in memory (not in Redis for security)
    if api_key:
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
async def edit_clip_v2(req: EditRequest):
    """Edit a clip by removing filler words and dead silence."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="v2 API requires Redis")

    store = RedisJobStore(redis)
    job = await store.get_job(req.job_id)

    if not job or not job.result or not job.result.clips:
        raise HTTPException(status_code=404, detail="Job not found or no clips available")

    try:
        clip = job.result.clips[req.clip_index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Clip index {req.clip_index} not found")

    filename = video_filename_from_url(clip.video_url)
    requested_filename = video_filename_from_url(req.input_filename) if req.input_filename else filename
    input_path = os.path.join(OUTPUT_DIR, req.job_id, requested_filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_edited{ext}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: cleaner.clean_clip(input_path, output_path))

        edited_filename = os.path.basename(output_path)
        new_url = build_video_url(req.job_id, edited_filename)

        return {"success": True, "edited_video_url": new_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")


@app.post("/api/v2/clean")
async def clean_clip_v2(req: CleanRequest):
    """Remove filler words (umm, um, uh, etc.) and dead silence from a clip."""
    redis = await get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="v2 API requires Redis")

    store = RedisJobStore(redis)
    job = await store.get_job(req.job_id)

    if not job or not job.result or not job.result.clips:
        raise HTTPException(status_code=404, detail="Job not found or no clips available")

    try:
        clip = job.result.clips[req.clip_index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Clip index {req.clip_index} not found")

    filename = video_filename_from_url(clip.video_url)
    input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    try:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: cleaner.clean_clip(input_path, output_path, req.max_silence_gap, req.silence_pad)
        )

        cleaned_filename = os.path.basename(output_path)
        new_url = build_video_url(req.job_id, cleaned_filename)
        return {"success": True, "cleaned_video_url": new_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clean failed: {str(e)}")


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
    filename = video_filename_from_url(clip.video_url)
    requested_filename = video_filename_from_url(req.input_filename) if req.input_filename else filename
    input_path = os.path.join(OUTPUT_DIR, req.job_id, requested_filename)

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
            lambda: subtitles.add_subtitles_to_video(
                input_path,
                # A tightened clip carries its own words on its own timeline.
                clip.clip_transcript or job.result.transcript,
                clip_start=0.0 if clip.clip_transcript else (clip.start or 0.0),
                clip_end=None if clip.clip_transcript else clip.end,
                position=req.position,
                preset=req.preset,
            )
        )

        # Return subtitled video URL
        subtitled_filename = os.path.basename(output_path)
        new_url = build_video_url(req.job_id, subtitled_filename)

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

        filename = video_filename_from_url(clip.video_url)
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
    """Download a video URL. YouTube -> savenow/savefrom service chain; other URLs direct."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or '').lower()
    is_youtube = any(h in hostname for h in ('youtube.com', 'youtu.be', 'youtube-nocookie.com'))

    if not is_youtube:
        # Direct URL (S3, Supabase, audio CDN, etc.). Fetch with a browser User-Agent —
        # many hosts (e.g. the audio CDNs) 403 the default Python-urllib UA. Extension is
        # irrelevant: faster-whisper decodes by content, so an m4a transcribes either way.
        ext = os.path.splitext(parsed.path)[1] or '.m4a'
        out = os.path.join(output_dir, f"video{ext}")
        youtube_download._download_file(url, out, min_bytes=2048)
        return out

    # YouTube: use the shared service chain — savenow (1080p REST API) -> savefrom driven
    # by CloakBrowser (stealth browser that clears Cloudflare from Railway's IP). No yt-dlp:
    # YouTube hard-blocks Railway's datacenter IP (HTTP 429 / "confirm you're not a bot").
    path, _title = youtube_download.download_youtube(url, output_dir, want='video')
    return path


def _has_audio(path: str) -> bool:
    """True when ffprobe finds an audio stream (or cannot tell — fail open)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return True
        return bool(proc.stdout.strip())
    except Exception:
        return True


def _transcribe(video_path: str) -> dict:
    """Transcribe a local audio/video file with the shared transcription
    module (transcription.py: Grok STT tiers, then faster-whisper WHISPER_MODEL with VAD;
    the model is loaded once per process)."""
    import transcription
    result = transcription.transcribe(video_path, word_timestamps=False)
    return {"text": result["text"], "language": result["language"],
            "duration": result.get("duration") or 0.0}


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_url(req: TranscribeRequest):
    """Download a video (YouTube or direct URL) and return its transcript."""
    import tempfile

    link_problem = user_errors.youtube_link_problem(req.url)
    if link_problem:
        raise HTTPException(status_code=400, detail=link_problem)

    tmp_dir = tempfile.mkdtemp(prefix="transcribe_")
    try:
        video_path = await asyncio.to_thread(_download_video, req.url, tmp_dir)
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail="Failed to download video")

        if not await asyncio.to_thread(_has_audio, video_path):
            raise HTTPException(status_code=400, detail="This video has no audio track to transcribe.")

        result = await asyncio.to_thread(_transcribe, video_path)

        return TranscribeResponse(
            text=result["text"],
            language=result.get("language", "en"),
            duration_seconds=result.get("duration", 0.0),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"transcribe failed for {req.url}: {type(e).__name__}: {e}")
        code, message = user_errors.classify(f"{type(e).__name__}: {e}")
        status = 500 if code in (user_errors.GENERIC[0], "DOWNLOAD_FAILED", "NO_SPACE") else 400
        raise HTTPException(status_code=status, detail=message)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
