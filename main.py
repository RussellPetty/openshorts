import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
import uuid
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import urllib.request
import urllib.parse
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import json
from caption_renderer import render_caption_on_frame, extract_words_from_transcript
import gemini_worker
import llm_backend
from clip_selection import (build_transcript_windows, clip_count_targets,
                            clip_duration_bounds, dedupe_overlapping,
                            score_batches, shortlist_target, snap_clip_to_words,
                            trim_to_best)
from ffmpeg_utils import (video_encode_args, audio_encode_args, cut_clip,
                          QUALITY_FAST, METADATA_SCRUB)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

# Sources shorter than this cannot yield a 15-60s clip; the job fails fast with
# a clear message instead of burning a download + transcription on it.
# MIN_SOURCE_SECONDS=0 disables the check.
MIN_SOURCE_SECONDS = int(os.environ.get("MIN_SOURCE_SECONDS", "45"))

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = YOLO('yolov8n.pt')

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        
        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        
        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / ASPECT_RATIO)
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box):
        """
        Updates the target center based on detected face/person.
        """
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2
    
    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            
            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1
                
                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0 # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan
                
                self.current_center_x += direction * speed
                
                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x
            
            # If inside safe zone, DO NOTHING (Stationary Camera)
                
        # Clamp center
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)
        
        # A source taller than the target aspect trims both ends instead of
        # pinning y=0 and throwing away the bottom of the frame.
        y1 = max(0, (self.video_height - self.crop_height) // 2)
        y2 = y1 + self.crop_height
        
        return x1, y1, x2, y2

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []
        
        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15 # Reduced matching radius to avoid jumping in groups
            
            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30: # Forgot faces older than 1s (was 2s)
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']
            
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    """
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h # Area as score
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    # Use the globally loaded model
    results = model(frame, verbose=False, classes=[0]) # class 0 is person
    
    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
        
    # Blur background
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h), interpolation=cv2.INTER_LANCZOS4)

    # A source taller than the output fills the width at a height that does not
    # fit: centre-crop it instead of indexing the frame with a negative offset,
    # which raises rather than renders.
    if fg_h > output_height:
        top = (fg_h - output_height) // 2
        foreground = foreground[top:top + output_height, :]
        fg_h = output_height

    # 3. Overlay
    y_offset = (output_height - fg_h) // 2
    
    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, :] = foreground
    
    return final_frame

def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []
    
    if not cap.isOpened():
        return ['TRACK'] * len(scenes)
        
    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        # Sample 3 frames (start, middle, end)
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5
        ]
        
        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            
        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)
            
        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)
        
        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')
            
    cap.release()
    return strategies

def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps


def source_already_fits(orig_w, orig_h, aspect_ratio=ASPECT_RATIO, tol=0.01):
    """True when the source is already at (or past) the target aspect.

    Such a source has no width to throw away: GENERAL would shrink it into a
    blurred bed of itself and TRACK's crop is the whole frame anyway, so a
    vertical upload is passed straight through (upstream b4be92c).
    """
    return orig_w / float(orig_h) <= aspect_ratio * (1 + tol)

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    return filename[:100]


def _is_youtube_url(url):
    """Check if a URL is a YouTube URL."""
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or '').lower()
    return any(h in hostname for h in ('youtube.com', 'youtu.be', 'youtube-nocookie.com'))


def _download_direct_url(url, output_dir="."):
    """Download a non-YouTube URL directly using urllib (no yt-dlp overhead)."""
    print(f"📥 Direct download (non-YouTube): {url}")
    step_start_time = time.time()

    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path) or 'video.mp4'
    if not filename.endswith('.mp4'):
        filename += '.mp4'
    sanitized = sanitize_filename(os.path.splitext(filename)[0])
    output_path = os.path.join(output_dir, f'{sanitized}.mp4')

    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    with urllib.request.urlopen(req, timeout=300) as resp, open(output_path, 'wb') as f:
        while True:
            chunk = resp.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)

    elapsed = time.time() - step_start_time
    print(f"✅ Downloaded in {elapsed:.2f}s: {output_path}")
    return output_path, sanitized


def _download_via_savenow(url, output_dir="."):
    """
    Download a YouTube video through the video-download-api.com REST API
    (savenow / loader-compatible). Their backend handles YouTube bot detection,
    PO tokens, IP rotation, and format extraction; we submit a job, poll progress,
    then fetch a clean merged mp4 — so Railway's datacenter IP is never the one
    talking to YouTube. Requires the SAVENOW_API_KEY env var.

      1) GET  p.savenow.to/ajax/download.php?url=<yt>&format=1080&apikey=<KEY>
              &allow_extended_duration=1                  -> {success, id, progress_url, title}
      2) poll progress_url (or /ajax/progress.php?id=<id>) -> until download_url present
      3) GET  download_url                                 -> final mp4

    Raises on any failure so the caller can fall back to the local yt-dlp path.
    """
    import json

    api_key = os.environ.get("SAVENOW_API_KEY")
    if not api_key:
        raise RuntimeError("SAVENOW_API_KEY not set")

    api_host = os.environ.get("SAVENOW_API_HOST", "https://p.savenow.to").rstrip('/')
    fmt = os.environ.get("SAVENOW_FORMAT", "1080")
    UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
          '(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36')

    def _get_json(u, timeout=30):
        req = urllib.request.Request(u, headers={'User-Agent': UA, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    step_start_time = time.time()
    print(f"\U0001F4E5 Trying video-download-api (savenow) for {url}")

    # 1) submit the download job
    submit_url = f"{api_host}/ajax/download.php?" + urllib.parse.urlencode({
        'url': url,
        'format': fmt,
        'apikey': api_key,
        'add_info': '1',
        'allow_extended_duration': '1',
        'no_merge': '0',
    })
    meta = _get_json(submit_url)
    if not meta.get('success') or not meta.get('id'):
        raise RuntimeError(f"savenow submit failed: {str(meta)[:200]}")

    job_id = meta['id']
    progress_url = meta.get('progress_url') or (
        f"{api_host}/ajax/progress.php?id={urllib.parse.quote(str(job_id))}")
    title = meta.get('title') or (meta.get('info') or {}).get('title') or 'youtube_video'
    sanitized_title = sanitize_filename(title)
    output_template = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if os.path.exists(output_template):
        os.remove(output_template)
    print(f"   job id={job_id} title={title!r}")

    # 2) poll until the file is ready (their side does the heavy lifting; be patient
    #    for long videos). download_url host varies per job (travisNN.savenow.to etc).
    deadline = time.time() + 600
    download_url = None
    poll = 0
    while time.time() < deadline:
        poll += 1
        try:
            job = _get_json(progress_url, timeout=30)
        except Exception as e:
            print(f"   savenow poll {poll} error: {e}")
            time.sleep(3)
            continue
        candidate = job.get('download_url') or job.get('url')
        if candidate and str(candidate).startswith(('http://', 'https://')):
            download_url = candidate
            print(f"   savenow render completed after {poll} polls in {time.time() - step_start_time:.1f}s")
            break
        if poll % 5 == 1:
            print(f"   savenow progress={job.get('progress')} {job.get('text')}")
        time.sleep(3)
    else:
        raise RuntimeError(f"savenow did not complete within 600s ({poll} polls)")

    if not download_url:
        raise RuntimeError("savenow completion missing download_url")

    # 3) download the merged mp4
    file_req = urllib.request.Request(download_url, headers={'User-Agent': UA})
    with urllib.request.urlopen(file_req, timeout=600) as resp, open(output_template, 'wb') as f:
        while True:
            chunk = resp.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            f.write(chunk)

    if not os.path.exists(output_template) or os.path.getsize(output_template) < 100 * 1024:
        raise RuntimeError(f"savenow downloaded file missing/too small: {output_template}")

    elapsed = time.time() - step_start_time
    print(f"\u2705 savenow done in {elapsed:.2f}s: {output_template}")
    return output_template, sanitized_title


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video.

    Tries the video-download-api (savenow) fast path first (their backend handles
    YouTube bot detection, PO tokens, and format extraction; we just consume a clean
    mp4). If that fails for any reason — service down, no API key, video they can't
    process — falls back to the local yt-dlp + bgutil POT pipeline.
    """
    # Bypass everything for non-YouTube direct URLs (Supabase, S3, etc.)
    if not _is_youtube_url(url):
        return _download_direct_url(url, output_dir)

    # Shared service chain: savenow (1080p REST API) -> savefrom via CloakBrowser. No yt-dlp
    # (YouTube blocks Railway's datacenter IP for it). The legacy yt-dlp block below this
    # return is now unreachable; kept temporarily to avoid a risky mass-delete.
    import youtube_download
    return youtube_download.download_youtube(url, output_dir, want='video')

    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")

    # One-shot bgutil/POT diagnostic — surfaces whether the local PO Token
    # server is reachable and whether the Python plugin loaded into yt-dlp.
    import subprocess
    try:
        bgutil_check = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '--max-time', '2',
             'http://127.0.0.1:4416/ping'],
            capture_output=True, text=True, timeout=5,
        )
        print(f"🔍 Debug: bgutil POT server /ping → HTTP {bgutil_check.stdout.strip() or 'ERR'}")
    except Exception as e:
        print(f"🔍 Debug: bgutil POT server probe error: {e}")
    try:
        from yt_dlp_plugins.extractor import getpot_bgutil_http  # noqa: F401
        print("🔍 Debug: bgutil-http Python plugin importable ✅")
    except Exception as e:
        print(f"🔍 Debug: bgutil-http Python plugin import failed: {e}")

    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = None
    cookies_env = os.environ.get("YOUTUBE_COOKIES")

    if cookies_env:
        import base64
        cookies_path = '/app/cookies.txt'
        print("🍪 YOUTUBE_COOKIES env var found, decoding...")
        try:
            raw = cookies_env.strip().strip('"').strip("'")
            # Try base64 decode (strip whitespace Railway may inject)
            cleaned = ''.join(raw.split())
            try:
                decoded = base64.b64decode(cleaned).decode('utf-8')
                if '.youtube.com' in decoded or '# Netscape' in decoded:
                    raw = decoded
                    print("   Debug: Base64 decoded successfully")
            except Exception as e:
                print(f"   Debug: Not base64 or decode failed: {e}")

            # Normalize to Netscape tab-separated format
            lines = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    lines.append(line)
                else:
                    parts = line.split()
                    if len(parts) >= 7:
                        lines.append('\t'.join(parts))

            if lines:
                with open(cookies_path, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
                print(f"   ✅ Cookie file written: {len(lines)} lines")
            else:
                print("   ⚠️ No valid cookie lines found")
                cookies_path = None
        except Exception as e:
            print(f"   ⚠️ Cookie processing failed: {e}")
            cookies_path = None
    else:
        print("ℹ️ No YOUTUBE_COOKIES env var set, proceeding without cookies")
    
    ydl_opts_info = {
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'cookiefile': cookies_path if cookies_path else None,
        'sleep_interval_requests': 5,
        'sleep_interval': 10,
        'max_sleep_interval': 30,
        'socket_timeout': 30,
        'retries': 10,
        'nocheckcertificate': True,
        'force_ipv4': True,
        'proxy': os.environ.get('YTDLP_PROXY') or None,
        'cachedir': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'js_runtimes': {'node': {}},
    }
    
    # Retry with escalating player client strategies to avoid bot detection.
    # `None` (yt-dlp defaults) is first because it's the only strategy that empirically
    # works with the burner-account cookies in YOUTUBE_COOKIES: the desktop-Chrome
    # cookies attached to non-web clients (`android_vr`, `web_safari`) trigger
    # LOGIN_REQUIRED at YouTube's player-response stage, and forcing `web` triggers
    # SABR streaming (yt-dlp #12482) which strips HD URLs and drops us to itag 18.
    # The defaults path produces ~720p HLS combined — accepting that as the working
    # quality ceiling until cookie scoping is split per-client.
    BOT_DETECTION_PATTERNS = ('Sign in to confirm', 'LOGIN_REQUIRED', 'HTTP Error 429')
    CLIENT_STRATEGIES = [
        None,                               # Attempt 1: yt-dlp defaults + cookies → 720p HLS (works)
        ['android_vr'],                     # Attempt 2: DASH 1080p path — only viable if cookies dropped
        ['web_safari'],                     # Attempt 3: HLS 1080p path — same caveat
    ]

    info = None
    successful_strategy = None
    last_error = None

    for attempt, strategy in enumerate(CLIENT_STRATEGIES, 1):
        opts = dict(ydl_opts_info)
        if strategy:
            opts['extractor_args'] = {'youtube': {'player_client': strategy}}

        strategy_label = ', '.join(strategy) if strategy else 'yt-dlp defaults'
        print(f"🔄 Attempt {attempt}/{len(CLIENT_STRATEGIES)}: Trying player clients: {strategy_label}")

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                successful_strategy = strategy
                print(f"   ✅ extract_info succeeded with: {strategy_label}")
                break
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_bot_error = any(pat in error_str for pat in BOT_DETECTION_PATTERNS)

            if is_bot_error and attempt < len(CLIENT_STRATEGIES):
                print(f"   ⚠️ Bot detection hit: {error_str[:120]}... retrying with next strategy")
                continue
            else:
                # Non-bot error or last attempt — fail immediately
                import traceback
                print("🚨 YOUTUBE DOWNLOAD ERROR 🚨", file=sys.stderr)

                error_msg = f"""

❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED
❌ ================================================================= ❌

REASON: YouTube has blocked the download request (Error 429/Unavailable).
        This is likely a temporary IP ban on this server.
        Tried {attempt} player client strategies without success.

👇 SOLUTION FOR USER 👇
---------------------------------------------------------------------
1. Download the video manually to your computer.
2. Use the 'Upload Video' tab in this app to process it.
---------------------------------------------------------------------

Technical Details: {str(e)}
            """
                print(error_msg, file=sys.stdout)
                print(error_msg, file=sys.stderr)
                sys.stdout.flush()
                sys.stderr.flush()
                time.sleep(0.5)
                raise e

    video_title = info.get('title', 'youtube_video')
    sanitized_title = sanitize_filename(video_title)
    
    output_template = os.path.join(output_dir, f'{sanitized_title}.%(ext)s')
    expected_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if os.path.exists(expected_file):
        os.remove(expected_file)
        print(f"🗑️  Removed existing file to re-download at HD")
    
    ydl_opts = {
        # Highest available resolution; the pipeline re-encodes to H.264 downstream so we
        # don't constrain source codec (above 1080p YouTube ships VP9/AV1, not H.264).
        # `bv*`/`b` (instead of `bestvideo`/`best`) lets us pick HLS progressive streams when
        # DASH isn't available — without this, mobile/web fallback clients fall through to
        # the 360p single-file mp4. `format_sort` keeps mp4/H.264 preferred when tied on res.
        'format': 'bv*+ba/b',
        'format_sort': ['res', 'ext:mp4:m4a', 'codec:h264:m4a', 'proto:https'],
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'overwrites': True,
        'cookiefile': cookies_path if cookies_path else None,
        'proxy': os.environ.get('YTDLP_PROXY') or None,
        'js_runtimes': {'node': {}},
        'socket_timeout': 300,
        'retries': 5,
    }
    # Propagate the player client strategy that worked during extract_info
    if successful_strategy:
        ydl_opts['extractor_args'] = {'youtube': {'player_client': successful_strategy}}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break
    
    step_end_time = time.time()
    print(f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
    
    return downloaded_file, sanitized_title

def process_video_to_vertical(input_video, final_output_video, transcript_words=None,
                               caption_style=None, caption_color=None, caption_outline_color=None,
                               clip_offset=0.0):
    """
    Core logic to convert horizontal video to vertical using scene detection and Active Speaker Tracking (MediaPipe).
    Optionally renders legacy OpenCV captions if caption_style is provided.

    ``transcript_words`` carry absolute source timestamps; ``clip_offset`` is
    where this clip starts in the source, so caption timing lines up with the
    clip's own timeline.
    """
    script_start_time = time.time()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    
    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    render_captions = bool(caption_style and caption_style != 'none' and transcript_words)
    if render_captions and clip_offset:
        transcript_words = [
            {**w, 'start': w['start'] - clip_offset, 'end': w['end'] - clip_offset}
            for w in transcript_words
        ]

    print(f"🎬 Processing clip: {input_video}")
    original_width, original_height = get_video_resolution(input_video)

    # A source shot vertical is already the output: nothing to reframe.
    passthrough = source_already_fits(original_width, original_height)
    if passthrough:
        print(f"   ↕️  Source is already {original_width}x{original_height} vertical — "
              f"passing it through, no reframe")

    if passthrough and not render_captions:
        return finalize_clip_passthrough(input_video, final_output_video)

    print("   Step 1: Detecting scenes...")
    if passthrough:
        scenes = []
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    else:
        scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        if not passthrough:
            print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    
    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * ASPECT_RATIO)
    if passthrough:
        OUTPUT_WIDTH = original_width
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1
    if OUTPUT_HEIGHT % 2 != 0:
        OUTPUT_HEIGHT += 1

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height)
    
    # --- New Strategy: Per-Scene Analysis ---
    if passthrough:
        scene_strategies = ['PASSTHROUGH'] * len(scenes)
    else:
        print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
        scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', *video_encode_args(QUALITY_FAST),
        '-pix_fmt', 'yuv420p', '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            # Determine Strategy for current frame based on scene
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            
            # Apply Strategy
            if current_strategy == 'PASSTHROUGH':
                if frame.shape[1] != OUTPUT_WIDTH or frame.shape[0] != OUTPUT_HEIGHT:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                else:
                    output_frame = frame
            elif current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                
            else:
                # "Single Speaker" -> Track & Crop
                
                # Detect every 2nd frame for performance
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
                
                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

            # Render captions if enabled
            if render_captions:
                current_time = frame_number / fps
                output_frame = render_caption_on_frame(
                    output_frame, transcript_words, current_time,
                    style_name=caption_style,
                    custom_color=caption_color,
                    custom_outline_color=caption_outline_color
                )

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
    
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   ✨ Step 5: Merging audio...")
    # Audio comes straight from the cut (already AAC + loudness-normalised by
    # cut_clip); anything else is encoded to AAC so the MP4 plays everywhere.
    merge_command = [
        'ffmpeg', '-y', '-i', temp_video_output, '-i', input_video,
        '-map', '0:v:0', '-map', '1:a:0?',
        '-c:v', 'copy', *_aac_args_for(input_video),
        *METADATA_SCRUB, '-movflags', '+faststart', '-shortest',
        final_output_video
    ]
        
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    # Clean up temp files
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    
    return True


def _audio_codec(path):
    """Codec name of the first audio stream, or None when there is none."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return out or None
    except Exception:
        return None


def _video_format(path):
    """(codec, pix_fmt) of the first video stream, or None."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name,pix_fmt', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        codec, pix_fmt = out.split(',')[:2]
        return codec, pix_fmt
    except Exception:
        return None


def has_audio_stream(path):
    return _audio_codec(path) is not None


def _aac_args_for(path):
    """Copy AAC audio untouched; re-encode anything else to AAC."""
    return ['-c:a', 'copy'] if _audio_codec(path) == 'aac' else ['-c:a', 'aac']


def finalize_clip_passthrough(input_video, final_output_video):
    """Keep the clip's native framing (already-vertical source).

    The input is the freshly encoded cut, so a stream-copy remux is enough to
    add +faststart — re-encoding here would only cost time and quality.
    """
    if os.path.exists(final_output_video):
        os.remove(final_output_video)
    # Stream-copy only what every player handles (H.264 4:2:0); a 10-bit or
    # 4:4:4 cut is re-encoded instead of shipped unplayable on iOS/Safari.
    video_args = ['-c:v', 'copy']
    if _video_format(input_video) != ('h264', 'yuv420p'):
        video_args = [*video_encode_args(QUALITY_FAST), '-pix_fmt', 'yuv420p']
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        *video_args, *_aac_args_for(input_video),
        *METADATA_SCRUB, '-movflags', '+faststart',
        final_output_video,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
    except subprocess.CalledProcessError as e:
        print("   ❌ Passthrough remux failed.")
        print("   Stderr:", e.stderr.decode(errors='replace'))
        return False
    print(f"   ✅ Clip saved to {final_output_video}")
    return True


def burn_preset_captions(clip_path, transcript, clip_start, clip_end, preset, color=None):
    """Burn an ASS caption preset (karaoke / trending looks) onto a finished
    clip, in place. Fail-open: a caption problem never costs the clip."""
    import subtitles as _subs
    output_dir = os.path.dirname(clip_path)
    # Neutral names: the .ass path is interpolated into an ffmpeg filter, where
    # an apostrophe from the video title would break it (upstream be4dd06).
    tmp_out = os.path.join(output_dir, f"captioned_{uuid.uuid4().hex[:8]}.mp4")
    try:
        if not _subs.burn_preset_captions(clip_path, transcript, clip_start, clip_end,
                                          tmp_out, preset=preset, font_color=color):
            print("   ℹ️ No words in range — clip ships without captions.")
            return False
        os.replace(tmp_out, clip_path)
        print(f"   💬 Captions burned ({preset}).")
        return True
    except Exception as e:
        print(f"   ⚠️ Captions failed ({type(e).__name__}: {e}) — delivering the clip without them.")
        return False
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

class NoAudioError(RuntimeError):
    """The source has no audio track, so there is nothing to transcribe."""


def transcribe_video(video_path):
    if not has_audio_stream(video_path):
        raise NoAudioError("This video has no audio track")
    print("🎙️  Transcribing video with Faster-Whisper (CPU Optimized)...")
    from faster_whisper import WhisperModel
    
    # Run on CPU with INT8 quantization for speed
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    segments, info = model.transcribe(video_path, word_timestamps=True)
    
    print(f"   Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    # Convert to openai-whisper compatible format
    transcript_segments = []
    full_text = ""
    
    for segment in segments:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        seg_dict = {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'words': []
        }
        
        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })
        
        transcript_segments.append(seg_dict)
        full_text += segment.text + " "
        
    return {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }


# --- Clip selection (upstream 2-pass: score windows -> detail shortlist) ----
# Primary model: DeepSeek V4.1 Flash on Fireworks via llm_backend (switched on
# by FIREWORKS_API_KEY). Gemini is the fallback when that lane fails, and the
# only option for the silent-video vision path.

_TRANSIENT_TOKENS = (
    '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED',
    '500', 'INTERNAL', 'overloaded', 'Deadline',
    'empty response body', 'did not contain a JSON object',
    'Failed to parse Gemini JSON response',
    'ConnectError', 'ReadTimeout', 'RemoteProtocolError', '502', '504',
    'validation error', 'timed out')


def _gemini_model_name():
    return os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'


def _run_llm_stage(lane, prompt, schema):
    """One schema-enforced model call with transient-error backoff.
    ``lane`` is ("llm", model) or ("gemini", client, model).
    Returns (parsed_dict, cost_analysis)."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if lane[0] == "llm":
                return llm_backend.generate_json(prompt, schema, model=lane[1])
            client, model_name = lane[1], lane[2]
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            # Policy blocks are deterministic — retrying only burns quota.
            gemini_worker.raise_if_blocked(response)
            # Parsing lives inside the retry loop on purpose: Gemini sometimes
            # returns 200 with an empty body; the same payload succeeds on retry.
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
            else:
                parsed = gemini_worker._parse_json_response_text(
                    gemini_worker._get_response_text(response))
            return parsed, gemini_worker._calculate_cost_analysis(response, model_name)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry
        except Exception as e:
            msg = str(e)
            transient = any(tok in msg for tok in _TRANSIENT_TOKENS)
            if attempt == max_attempts or not transient:
                raise
            wait = 5 * (2 ** (attempt - 1))
            who = "LLM" if lane[0] == "llm" else "Gemini"
            print(f"⚠️ {who} transient error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def _run_stage_split(lane, items, build_prompt, schema, key, costs, label):
    """Run a selection stage over ``items``; on a content-policy block, bisect.

    Google's prompt filter fires on some COMBINATIONS of transcript windows
    that pass individually, so instead of failing the job the batch is split
    in halves until the offending window is isolated and dropped."""
    if not items:
        return []
    prompt = build_prompt(items)
    try:
        parsed, cost = _run_llm_stage(lane, prompt, schema)
        if cost:
            costs.append(cost)
        return list(parsed.get(key) or [])
    except gemini_worker.GeminiBlockedError as e:
        if len(items) == 1:
            print(f"   🚫 {label}: blocked window {items[0].get('id')} on its own; skipping it ({e})")
            return []
        mid = len(items) // 2
        print(f"   🚫 {label}: blocked a batch of {len(items)}; retrying as {mid} + {len(items) - mid}")
        return (_run_stage_split(lane, items[:mid], build_prompt, schema, key, costs, label)
                + _run_stage_split(lane, items[mid:], build_prompt, schema, key, costs, label))


def score_batch_size():
    """Transcript windows per scoring call (``LLM_SCORE_BATCH`` overrides).
    Both DeepSeek V4.1 Flash and Gemini have 1M-token context, so 8."""
    raw = os.environ.get("LLM_SCORE_BATCH", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 8


def _selection_lanes():
    lanes = []
    if llm_backend.active():
        lanes.append(("llm", llm_backend.model_name()))
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        lanes.append(("gemini", genai.Client(api_key=api_key), _gemini_model_name()))
    return lanes


def _lane_label(lane):
    return f"{lane[1]} @ {llm_backend.base_url()}" if lane[0] == "llm" else lane[2]


def _select_clips_with(lane, transcript_result, video_duration):
    """Two-pass clip selection on one model lane. Returns (shorts, costs)."""
    language = str(transcript_result.get('language') or 'unknown')

    # Full word list — ground truth for snapping cut points.
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    # Scoring windows must be able to CONTAIN a max-length clip.
    min_secs, max_secs = clip_duration_bounds()
    windows = build_transcript_windows(
        transcript_result, video_duration,
        window_seconds=max(90, int(max_secs * 1.5)))
    print(f"   Built {len(windows)} scoring window(s).")
    costs = []

    # --- Pass 1: score every window, keep the global top `target` ---
    scored = []
    target = shortlist_target(video_duration)

    def _payload(ws):
        return [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in ws]

    def _score_prompt(ws):
        return gemini_worker.SCORE_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            windows_json=json.dumps(_payload(ws), ensure_ascii=False))

    for batch in score_batches(windows, score_batch_size()):
        scored.extend(_run_stage_split(
            lane, batch, _score_prompt,
            gemini_worker.ScoreResponse, "windows", costs, "score"))

    scored.sort(key=lambda w: w.get("score", 0), reverse=True)
    by_id = {w["id"]: w for w in windows}
    shortlist = [by_id[w["id"]] for w in scored[:target] if w.get("id") in by_id]
    if not shortlist:
        shortlist = windows[:target]  # scoring returned nothing usable
    print(f"   Shortlisted {len(shortlist)} window(s) for detail.")

    # --- Pass 2: detailed clip extraction on the shortlist ---
    min_clips, max_clips = clip_count_targets(len(shortlist))

    def _detail_prompt_for(lo, hi):
        def build(ws):
            return gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                min_clips=lo, max_clips=hi,
                min_secs=min_secs, max_secs=max_secs,
                windows_json=json.dumps(_payload(ws), ensure_ascii=False))
        return build

    shorts = _run_stage_split(lane, shortlist,
                              _detail_prompt_for(min_clips, max_clips),
                              gemini_worker.DetailResponse, "shorts", costs, "detail")

    # The floor lives only in the prompt; give the unused windows one more
    # call for the missing clips.
    if len(shorts) < min_clips:
        used = {str(s.get("source_window_id") or "") for s in shorts}
        spare = [w for w in shortlist if w["id"] not in used]
        if spare:
            missing = min_clips - len(shorts)
            print(f"   Detail returned {len(shorts)} clip(s) of {min_clips}; "
                  f"asking the {len(spare)} unused window(s) for {missing} more.")
            extra = _run_stage_split(
                lane, spare, _detail_prompt_for(missing, max(missing, len(spare))),
                gemini_worker.DetailResponse, "shorts", costs, "detail-floor")
            if extra:
                shorts = sorted(shorts + extra, key=lambda s: float(s.get("start") or 0))
                print(f"   Recovered {len(extra)} clip(s) from them.")

    if len(shorts) > max_clips:
        # By score, never by position (upstream 4a13700).
        dropped = len(shorts) - max_clips
        shorts = trim_to_best(shorts, max_clips)
        print(f"   Kept the {max_clips} best-scoring clip(s) of {max_clips + dropped}.")

    # Snap each proposed clip onto real word boundaries (+ a bit of silence).
    for s in shorts:
        s["proposed"] = [s.get("start", 0), s.get("end", 0)]
        ns, ne = snap_clip_to_words(s.get("start", 0), s.get("end", 0), words, video_duration,
                                    min_duration=min_secs, max_duration=max_secs)
        s["start"], s["end"] = ns, ne
    deduped = dedupe_overlapping(shorts)
    if len(deduped) < len(shorts):
        print(f"   Dropped {len(shorts) - len(deduped)} clip(s) overlapping a better-scored one.")
        shorts = deduped
    return shorts, costs


def get_viral_clips(transcript_result, video_duration):
    """Pick clips from the transcript: DeepSeek first, Gemini as fallback.
    Returns {"shorts": [...], "cost_analysis": {...}} or None."""
    lanes = _selection_lanes()
    if not lanes:
        print("❌ Error: no clip-selection model configured "
              "(set FIREWORKS_API_KEY or GEMINI_API_KEY).")
        return None

    print(f"🤖  Analyzing transcript (2-pass: score → detail), language: "
          f"{transcript_result.get('language') or 'unknown'}")
    blocked = None
    for lane in lanes:
        label = _lane_label(lane)
        print(f"🤖  Model: {label}")
        try:
            shorts, costs = _select_clips_with(lane, transcript_result, video_duration)
        except gemini_worker.GeminiBlockedError as e:
            print(f"🚫 {e}")
            blocked = e
            continue
        except Exception as e:
            print(f"❌ Clip selection failed on {label}: {type(e).__name__}: {e}")
            continue
        if not shorts:
            print(f"⚠️ {label} returned no clips.")
            continue

        result = {"shorts": shorts}
        if costs:
            result["cost_analysis"] = {
                "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
                "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
                "total_cost": sum(c.get("total_cost", 0) for c in costs),
                "model": costs[-1].get("model"),
                "calls": len(costs),
            }
            print(f"💰 {len(costs)} selection call(s) on {label}: "
                  f"{result['cost_analysis']['input_tokens']} in / "
                  f"{result['cost_analysis']['output_tokens']} out tokens")
        return result

    if blocked is not None:
        # Content-policy rejection with nothing else to try: fail with the
        # real reason instead of a generic "no clips found".
        raise blocked
    return None


# --- Speech too sparse to clip by transcript -------------------------------
MIN_SPEECH_WORDS_PER_MIN = float(os.environ.get("MIN_SPEECH_WORDS_PER_MIN", "5"))
MIN_SPEECH_WORDS = int(os.environ.get("MIN_SPEECH_WORDS", "8"))


def speech_is_sparse(transcript, duration):
    """True when the transcript is too thin to drive clip selection."""
    words = sum(len((seg.get("text") or "").split())
                for seg in (transcript or {}).get("segments", []))
    minutes = max(float(duration or 0) / 60.0, 1e-6)
    return words < MIN_SPEECH_WORDS or words / minutes < MIN_SPEECH_WORDS_PER_MIN


def get_visual_clips(video_path, video_duration, language="en"):
    """Clip a SILENT (or speechless) video by vision: Gemini watches the
    footage and picks the most engaging visual moments. Returns the same
    {"shorts", "cost_analysis"} shape as get_viral_clips, or None."""
    print("🎥  No usable speech — analyzing with Gemini vision (no transcript)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found (needed for silent videos).")
        return None
    client = genai.Client(api_key=api_key)
    model_name = _gemini_model_name()
    print(f"🎥  Model: {model_name} | uploading {os.path.basename(video_path)}…")

    file_upload = None
    try:
        # By handle, not path: a non-ASCII title in the filename breaks the
        # SDK's upload header (upstream ff2fd10).
        file_upload = gemini_worker.upload_media(client, video_path)
        deadline = time.time() + 180
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                print("❌ Gemini could not process the video.")
                return None
            if time.time() > deadline:
                print("❌ Gemini video processing timed out.")
                return None
            time.sleep(2)

        prompt = gemini_worker.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.VisualResponse,
        )
        response = client.models.generate_content(
            model=model_name, contents=[file_upload, prompt], config=config)
        gemini_worker.raise_if_blocked(response)
        parsed = gemini_worker._parse_json_response_text(gemini_worker._get_response_text(response))
        shorts = parsed.get("shorts") or []
        # Clamp to the real duration; drop anything degenerate.
        clean = []
        for s in shorts:
            s["start"] = max(0.0, float(s.get("start", 0)))
            s["end"] = min(float(video_duration), float(s.get("end", 0)))
            if s["end"] - s["start"] >= 1.0:
                clean.append(s)
        if not clean:
            print("⚠️ Vision pass returned no usable clips.")
            return None

        cost = gemini_worker._calculate_cost_analysis(response, model_name)
        result = {"shorts": clean}
        if cost:
            result["cost_analysis"] = cost
        return result
    except gemini_worker.GeminiBlockedError:
        raise
    except Exception as e:
        print(f"❌ Gemini vision error: {e}")
        return None
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass


def _fail(code, message):
    """Exit the job with a user-facing reason app.py can surface."""
    msg = f"❌ {code}: {message}"
    print(msg, file=sys.stdout)
    print(msg, file=sys.stderr)
    sys.stdout.flush(); sys.stderr.flush()
    raise SystemExit(1)


# Caption styles rendered from ASS presets (proper fonts, word-level karaoke
# highlight). The remaining styles are the legacy OpenCV renderer.
ASS_CAPTION_STYLES = ('karaoke', 'default', 'hormozi', 'pill', 'lime', 'oneword', 'clean')
LEGACY_CAPTION_STYLES = ('classic', 'boxed', 'yellow', 'minimal', 'bold', 'neon', 'gradient')


def _probe_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                              '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip() or 0)
    except Exception:
        return 0.0


def render_clip_file(input_video, clip_path, start, end, clip_number, transcript, all_words,
                     caption_style, caption_color, caption_outline_color):
    """Cut [start, end] from the source, reframe to 9:16 and caption it."""
    clip_temp_path = os.path.join(os.path.dirname(clip_path),
                                  f"temp_{uuid.uuid4().hex[:8]}_{os.path.basename(clip_path)}")
    try:
        cut_clip(input_video, clip_temp_path, start, end, clip_number)
        legacy = caption_style in LEGACY_CAPTION_STYLES
        success = process_video_to_vertical(
            clip_temp_path, clip_path,
            all_words if legacy else None,
            caption_style if legacy else None, caption_color, caption_outline_color,
            clip_offset=start)
        if success and caption_style in ASS_CAPTION_STYLES and transcript:
            burn_preset_captions(clip_path, transcript, start, end, caption_style, caption_color)
        return success
    finally:
        if os.path.exists(clip_temp_path):
            os.remove(clip_temp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--caption-style', type=str,
                        choices=list(LEGACY_CAPTION_STYLES) + list(ASS_CAPTION_STYLES) + ['none'],
                        default='none', help="Caption style to apply")
    parser.add_argument('--caption-color', type=str, help="Custom text color in hex (e.g. #FFFFFF)")
    parser.add_argument('--caption-outline-color', type=str, help="Custom outline color in hex (e.g. #000000)")

    args = parser.parse_args()

    script_start_time = time.time()
    
    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path
    
    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."
        
        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        
        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # Get caption parameters
    caption_style = getattr(args, 'caption_style', 'none')
    caption_color = getattr(args, 'caption_color', None)
    caption_outline_color = getattr(args, 'caption_outline_color', None)

    duration = _probe_duration(input_video)

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")

        transcript = None
        if caption_style and caption_style != 'none' and has_audio_stream(input_video):
            print("📝 Transcribing for captions...")
            transcript = transcribe_video(input_video)
        render_clip_file(input_video, output_file, 0, duration, 1, transcript,
                         extract_words_from_transcript(transcript),
                         caption_style, caption_color, caption_outline_color)
    else:
        # Too-short sources cannot yield a 15-60s clip (upstream 730f7de).
        if MIN_SOURCE_SECONDS > 0 and 0 < duration < MIN_SOURCE_SECONDS:
            _fail("SOURCE_TOO_SHORT",
                  f"This video is only {int(duration)}s long — clip generation needs at least "
                  f"{MIN_SOURCE_SECONDS}s of material to cut from. It already is short-form content.")

        # 3. Transcribe — unless the video has no audio (or no real speech), in
        # which case Gemini vision picks clips from the imagery instead.
        transcript = None
        try:
            transcript = transcribe_video(input_video)
        except NoAudioError as e:
            print(f"🔇 {e} — switching to visual analysis.")

        try:
            if transcript is not None and not speech_is_sparse(transcript, duration):
                clips_data = get_viral_clips(transcript, duration)
            else:
                if transcript is not None:
                    print("🔇 Too little speech to clip by transcript — switching to visual analysis.")
                clips_data = get_visual_clips(input_video, duration,
                                              language=(transcript or {}).get('language') or 'en')
        except gemini_worker.GeminiBlockedError as e:
            _fail("CONTENT_BLOCKED", str(e))

        # Extract words for captions (absolute source timestamps)
        all_words = extract_words_from_transcript(transcript)
        if transcript is None:
            transcript = {"language": "none", "segments": [], "text": ""}

        if not clips_data or 'shorts' not in clips_data:
            # Keep the job useful: one vertical clip from the start of the
            # video, capped at the max clip length (rendering a whole
            # hour-long source as a single "clip" helped nobody).
            _, max_secs = clip_duration_bounds()
            fallback_end = min(duration, max_secs) if duration else max_secs
            print(f"⚠️ No clips identified. Creating a single {fallback_end:.0f}s clip as fallback.")
            output_file = os.path.join(output_dir, f"{video_title}_clip_1.mp4")
            render_clip_file(input_video, output_file, 0, fallback_end, 1, transcript, all_words,
                             caption_style, caption_color, caption_outline_color)
            # Write fallback metadata so app.py can finalize the job
            fallback_metadata = {
                "shorts": [{
                    "start": 0,
                    "end": fallback_end,
                    "video_title_for_youtube_short": video_title,
                    "video_description_for_tiktok": "",
                    "video_description_for_instagram": ""
                }],
                "transcript": transcript
            }
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(fallback_metadata, f, indent=2)
            print(f"   Saved fallback metadata to {metadata_file}")
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")
            
            # Save metadata
            clips_data['transcript'] = transcript # Save full transcript for subtitles
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
            print(f"   Saved metadata to {metadata_file}")

            # 5. Process each clip
            for i, clip in enumerate(clips_data['shorts']):
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")
                
                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_final_path = os.path.join(output_dir, clip_filename)
                try:
                    success = render_clip_file(input_video, clip_final_path, start, end, i + 1,
                                               transcript, all_words,
                                               caption_style, caption_color, caption_outline_color)
                    if success:
                        print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                except Exception as e:
                    print(f"   ❌ Clip {i+1} failed: {type(e).__name__}: {e}")

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
