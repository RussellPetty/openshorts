"""One faster-whisper setup for every transcription path.

main.py (clip selection + captions), cleaner.py (/api/v2/edit, /api/v2/clean),
subtitles.py and /api/transcribe used to each build their own WhisperModel —
main.py with a hard-coded "base", no VAD and the default
condition_on_previous_text=True (repetition/hallucination loops on long
sources), while subtitles.py had the tuned settings nobody else used. They all
go through ``transcribe()`` now.

Env overrides:
  WHISPER_MODEL       model size/name (default "small": noticeably better
                      words and punctuation than "base" — punctuation is what
                      the clip boundary snapper keys on — at ~2x base's CPU time)
  WHISPER_DEVICE      default "cpu"
  WHISPER_COMPUTE     default "int8"
  WHISPER_BEAM_SIZE   default 5 (faster-whisper's own default)
  WHISPER_CPU_THREADS default 0 = library default

Engine: the LLM gateway's speech-to-text (``POST {gateway}/grok/stt``) is the
PRIMARY transcriber, then xAI's API directly (GROK_API_KEY, then
GROK_API_KEY_FALLBACK), then local faster-whisper (see stt_targets). Unlike Whisper it keeps
disfluencies ("um", "a, a") in the words, which is what clip tightening
(tighten.py) keys on. The result carries ``engine`` = "grok-proxy" |
"grok-api" | "whisper-small".
  TRANSCRIBE_ENGINE   grok (default) | whisper (skip the gateway entirely)
  GROK_STT_MODEL      default grok-voice-transcribe-2.0
  LLM_GATEWAY_API_KEY / LLM_GATEWAY_URL   same gateway settings llm_backend uses
  GROK_API_KEY, GROK_API_KEY_FALLBACK     direct xAI API tiers
"""
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_WHISPER_MODEL = "small"


def get_whisper_config():
    """The faster-whisper model config, overridable via env vars."""
    try:
        threads = int(os.environ.get("WHISPER_CPU_THREADS", "0") or 0)
    except ValueError:
        threads = 0
    return {
        "model_size": os.environ.get("WHISPER_MODEL") or DEFAULT_WHISPER_MODEL,
        "device": os.environ.get("WHISPER_DEVICE") or "cpu",
        "compute_type": os.environ.get("WHISPER_COMPUTE") or "int8",
        "cpu_threads": max(0, threads),
    }


def _beam_size():
    try:
        return max(1, int(os.environ.get("WHISPER_BEAM_SIZE", "5") or 5))
    except ValueError:
        return 5


# Decode params shared by every path. condition_on_previous_text is off to
# avoid repetition/hallucination loops; vad_filter drops silence (and the
# "Thank you." hallucinations Whisper invents over it).
WHISPER_TRANSCRIBE_PARAMS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": False,
    "word_timestamps": True,
}


def transcribe_params(word_timestamps=True):
    params = dict(WHISPER_TRANSCRIBE_PARAMS)
    params["beam_size"] = _beam_size()
    params["word_timestamps"] = bool(word_timestamps)
    return params


def merge_continuation_words(words):
    """Merge faster-whisper continuation fragments into their base word.

    faster-whisper marks a word boundary with a LEADING SPACE on each token.
    Compound-word fragments (e.g. "-Kanal.", ".200") arrive WITHOUT a leading
    space and belong to the preceding word. Without merging, "YouTube" and
    "-Kanal." get space-joined into "YouTube -Kanal." or split across subtitle
    blocks. We concatenate such fragments onto the previous word and extend its
    end time. Normal words keep their leading space, so real word boundaries
    (e.g. "ich habe") are never glued together. Idempotent.

    Returns a new list; the input dicts are not mutated.
    """
    merged = []
    for word in words:
        text = word.get("word", "")
        if merged and isinstance(text, str) and text and not text.startswith(" "):
            prev = merged[-1]
            prev["word"] = f"{prev.get('word', '')}{text}"
            if word.get("end") is not None:
                prev["end"] = word["end"]
        else:
            merged.append(dict(word))
    return merged


# Scripts written without spaces: faster-whisper's tokens there never carry a
# leading space, so merge_continuation_words would glue a whole segment into
# one "word" and destroy the word timings clip snapping and captions need.
NO_SPACE_LANGUAGES = frozenset({"zh", "yue", "ja", "th", "lo", "km", "my", "bo"})


def merge_for_language(words, language):
    """merge_continuation_words, except for languages written without spaces."""
    if str(language or "").lower() in NO_SPACE_LANGUAGES:
        return [dict(w) for w in words]
    return merge_continuation_words(words)


_model_lock = threading.Lock()
_models = {}


def get_model():
    """The WhisperModel for the current config, loaded once per process."""
    cfg = get_whisper_config()
    key = (cfg["model_size"], cfg["device"], cfg["compute_type"], cfg["cpu_threads"])
    model = _models.get(key)
    if model is None:
        with _model_lock:
            model = _models.get(key)
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(cfg["model_size"], device=cfg["device"],
                                     compute_type=cfg["compute_type"],
                                     cpu_threads=cfg["cpu_threads"])
                _models[key] = model
    return model


def _transcribe_whisper(media_path, word_timestamps=True, log=None, echo_segments=False):
    """Transcribe a local audio/video file.

    Returns {"text", "segments": [{"text","start","end","words":[{"word",
    "start","end","probability"}]}], "language", "language_probability",
    "duration"} — the openai-whisper shape every caller already consumes.
    """
    log = log or (lambda *_: None)
    cfg = get_whisper_config()
    log(f"🎙️  Transcribing with faster-whisper ({cfg['model_size']}, {cfg['device']}/"
        f"{cfg['compute_type']}, VAD on)...")
    segments, info = get_model().transcribe(media_path, **transcribe_params(word_timestamps))
    log(f"   Detected language '{info.language}' with probability "
        f"{(info.language_probability or 0):.2f}")
    out = []
    for seg in segments:
        if echo_segments:
            log(f"   [{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
        words = [{"word": w.word, "start": w.start, "end": w.end,
                  "probability": w.probability} for w in (seg.words or [])]
        out.append({"text": seg.text, "start": seg.start, "end": seg.end,
                    "words": merge_for_language(words, info.language)})
    duration = float(getattr(info, "duration", 0) or 0) or (out[-1]["end"] if out else 0.0)
    return {
        "text": " ".join(s["text"].strip() for s in out).strip(),
        "segments": out,
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "duration": duration,
        "engine": f"whisper-{cfg['model_size']}",
    }


# --- Primary engine: the LLM gateway's speech-to-text ------------------------

GROK_STT_DEFAULT_MODEL = "grok-voice-transcribe-2.0"
# The gateway takes <= 32 MiB per request; stay clear of it.
GROK_MAX_UPLOAD_BYTES = 30 * 1024 * 1024
GROK_AUDIO_BITRATE_KBPS = 40          # mono 16 kHz mp3: ~105 min per 30 MiB
GROK_CHUNK_SECONDS = 20 * 60.0        # chunk length once a file is too big
GROK_CHUNK_OVERLAP = 3.0              # audio overlap on each side of a seam
GROK_CONCURRENCY = 2                  # gateway allows 4 standalone speech requests
GROK_TIMEOUT_SECONDS = 300.0
# Segment building from the flat word list.
SEGMENT_GAP_SECONDS = 1.0
SEGMENT_MAX_SECONDS = 15.0
_SENTENCE_END_RE = re.compile(r"[.!?…。！？]['\")\]]*$")


XAI_STT_URL = "https://api.x.ai/v1/stt"
RETRY_WAITS = (1.0, 2.0, 4.0)         # paid API: exponential backoff before retries 1..3
# Gateway 429 = its 4-concurrent speech cap, shared by every job (5 jobs x 2
# chunks can exceed it): wait longer, with jitter so retries don't re-collide.
GATEWAY_429_WAITS = (3.0, 8.0, 20.0)
GATEWAY_429_JITTER = 0.5              # +0..50% of each wait
# Our own gateway STT calls are capped per MACHINE (all jobs/workers share
# the box) with flock'ed slot files; a call that cannot get a slot within
# GATEWAY_SLOT_WAIT_SECONDS moves on to the paid API.
GATEWAY_SLOT_WAIT_SECONDS = 180.0
# Whole Grok stage budget: min(15 min, 0.5 x audio duration + 120s); past it
# the job falls back to Whisper instead of retrying on.
GROK_STAGE_MAX_SECONDS = 900.0


class SttError(RuntimeError):
    """A tier failed. The message is safe to log: a status code or a failure
    class only — never a response body, URL, key or transcript text."""


def transcribe_engine():
    """'grok' (gateway, then xAI API keys, then Whisper) or 'whisper' (local only)."""
    v = (os.environ.get("TRANSCRIBE_ENGINE") or "grok").strip().lower()
    return "whisper" if v == "whisper" else "grok"


def _gateway_stt_url():
    try:
        from llm_backend import GATEWAY_DEFAULT_URL
    except Exception:  # pragma: no cover - llm_backend always ships alongside
        GATEWAY_DEFAULT_URL = "https://llm.broker-marketplace.com/v1"
    base = (os.environ.get("LLM_GATEWAY_URL") or GATEWAY_DEFAULT_URL).strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/grok/stt"


def stt_targets():
    """The STT tiers in order, mirroring the app's requestGrokAudio
    (src/lib/grok-audio.ts):

      grok-proxy  LLM gateway /v1/grok/stt (LLM_GATEWAY_API_KEY): a 429 is the
                  gateway's brief concurrency cap, so it is retried; any other
                  4xx/5xx or a network error moves on to the next tier.
      grok-api    xAI directly with GROK_API_KEY, then GROK_API_KEY_FALLBACK:
                  500/502/503/504 and network errors are retried.

    Each retryable failure is retried up to 3 times: paid API 1s/2s/4s;
    gateway 429 3s/8s/20s plus jitter. A paid-API read/write timeout is NOT
    retried (the request may already have been processed and billed)."""
    targets = []
    gw = (os.environ.get("LLM_GATEWAY_API_KEY") or "").strip()
    if gw:
        targets.append({"tier": "grok-proxy", "label": "gateway", "url": _gateway_stt_url(),
                        "key": gw, "retry_statuses": (429,), "retry_network": False,
                        "waits": GATEWAY_429_WAITS, "jitter": GATEWAY_429_JITTER,
                        "gateway": True})
    seen = set()
    for n, name in enumerate(("GROK_API_KEY", "GROK_API_KEY_FALLBACK"), start=1):
        key = (os.environ.get(name) or "").strip()
        if key and key not in seen:
            seen.add(key)
            targets.append({"tier": "grok-api", "label": f"API key {n}", "url": XAI_STT_URL,
                            "key": key, "retry_statuses": (500, 502, 503, 504),
                            # Connect-phase errors only: after a read timeout
                            # the request may have been processed (and billed).
                            "retry_network": True, "waits": RETRY_WAITS, "jitter": 0.0,
                            "gateway": False})
    return targets


def stt_model():
    return (os.environ.get("GROK_STT_MODEL") or GROK_STT_DEFAULT_MODEL).strip()


def _client(timeout=GROK_TIMEOUT_SECONDS):
    """Factory so tests can swap in httpx.MockTransport."""
    import httpx
    return httpx.Client(timeout=timeout, follow_redirects=False)


def valid_stt_body(body):
    """A usable STT response: a dict with a ``words`` list, and words present
    whenever there is text (text without words = no timings = useless here;
    a missing ``words`` would silently become an empty transcript)."""
    if not isinstance(body, dict) or not isinstance(body.get("words"), list):
        return False
    return bool(body["words"]) or not str(body.get("text") or "").strip()


def _gateway_max_concurrent():
    try:
        return max(1, int(os.environ.get("GROK_GATEWAY_MAX_CONCURRENT", "3")))
    except ValueError:
        return 3


class _GatewaySlot:
    """Cross-process cap on our concurrent gateway STT calls: flock on one of
    N slot files. ``acquire`` returns False when no slot frees up in time."""

    def __init__(self):
        self.fh = None

    def acquire(self, wait_seconds, sleep):
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX: no cap
            return True
        d = os.environ.get("GROK_STT_SLOT_DIR") or os.path.join(tempfile.gettempdir(),
                                                                "openshorts-stt-slots")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return True
        end = time.monotonic() + max(0.0, wait_seconds)
        n = _gateway_max_concurrent()
        while True:
            for k in range(n):
                try:
                    fh = open(os.path.join(d, f"slot{k}"), "a")
                except OSError:
                    return True
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.fh = fh
                    return True
                except OSError:
                    fh.close()
            if time.monotonic() >= end:
                return False
            sleep(0.5 + random.random())

    def release(self):
        if self.fh is not None:
            try:
                self.fh.close()  # closing drops the flock
            finally:
                self.fh = None


def _remaining(deadline):
    return float("inf") if deadline is None else deadline - time.monotonic()


def post_stt(audio_bytes, targets, language=None, filename="audio.mp3", sleep=None,
             log=None, deadline=None):
    """POST one audio file through the STT tiers. Returns (body, tier).

    ``deadline`` (time.monotonic) bounds the whole thing, waits included:
    past it SttError is raised and the caller falls back to Whisper."""
    import httpx
    log = log or (lambda *_: None)
    sleep = sleep or time.sleep
    if not targets:
        raise SttError("speech transcription not configured")
    data = {"model": stt_model()}
    if language:
        data["language"] = language
    failure = "request failed"
    connect_errors = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    for t in targets:
        waits = t.get("waits", RETRY_WAITS)
        for attempt in range(len(waits) + 1):
            left = _remaining(deadline)
            if left < 5:
                raise SttError("speech transcription time budget exhausted")
            retryable = False
            slot = _GatewaySlot() if t.get("gateway") else None
            try:
                if slot and not slot.acquire(min(GATEWAY_SLOT_WAIT_SECONDS, left - 5), sleep):
                    failure = "gateway busy"
                    break
                with _client(timeout=min(GROK_TIMEOUT_SECONDS, _remaining(deadline))) as client:
                    # httpx writes the plain fields before the file part, which
                    # the endpoint requires (model/language BEFORE file).
                    r = client.post(t["url"], headers={"Authorization": f"Bearer {t['key']}"},
                                    data=data, files={"file": (filename, audio_bytes, "audio/mpeg")})
                if r.status_code == 200:
                    try:
                        body = r.json()
                    except ValueError:
                        body = None
                    if valid_stt_body(body):
                        return body, t["tier"]
                    failure = "invalid response"
                else:
                    failure = f"HTTP {r.status_code}"
                    retryable = r.status_code in t["retry_statuses"]
            except httpx.HTTPError as e:
                # Never log the exception text: it can carry the URL.
                failure = "network error or timeout"
                retryable = t["retry_network"] and isinstance(e, connect_errors)
            finally:
                if slot:
                    slot.release()
            if not retryable or attempt == len(waits):
                break
            wait = waits[attempt] * (1 + random.random() * t.get("jitter", 0.0))
            if wait > _remaining(deadline) - 5:
                raise SttError("speech transcription time budget exhausted")
            sleep(wait)
        log(f"   ⚠️ Speech transcription ({t['tier']}, {t['label']}) failed: {failure}")
    raise SttError(f"speech transcription failed: {failure}")


def stage_deadline(duration):
    """time.monotonic() deadline for the whole Grok stage of one file."""
    budget = min(GROK_STAGE_MAX_SECONDS, 0.5 * float(duration or 0) + 120.0)
    return time.monotonic() + budget


def media_duration(path):
    """Container duration in seconds (0.0 when unknown)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out or 0)
    except Exception:
        return 0.0


def plan_chunks(duration, bitrate_kbps=GROK_AUDIO_BITRATE_KBPS, max_bytes=GROK_MAX_UPLOAD_BYTES,
                chunk_seconds=GROK_CHUNK_SECONDS, overlap=GROK_CHUNK_OVERLAP):
    """[(audio_start, audio_end, own_start, own_end)] covering [0, duration].

    One chunk when the encoded audio fits under ``max_bytes`` AND the audio
    is no longer than ``chunk_seconds`` (one huge request is one huge retry
    and one long wait on a single connection). Otherwise fixed
    ``chunk_seconds`` pieces whose AUDIO reaches ``overlap`` past each seam,
    while each chunk only OWNS the words that start inside [own_start,
    own_end) — so a word straddling a seam is taken whole from the chunk where
    it starts, and never twice."""
    duration = float(duration or 0)
    est = duration * bitrate_kbps * 1000 / 8
    if duration <= 0 or (est <= max_bytes and duration <= chunk_seconds):
        return [(0.0, duration, 0.0, float("inf"))]
    out, k = [], 0
    while k * chunk_seconds < duration:
        own_s = k * chunk_seconds
        own_e = min(duration, (k + 1) * chunk_seconds)
        a_s = max(0.0, own_s - overlap)
        a_e = min(duration, own_e + overlap)
        last = own_e >= duration - 1e-6
        out.append((a_s, a_e, 0.0 if k == 0 else own_s, float("inf") if last else own_e))
        k += 1
    return out


def stitch_chunk_words(chunk_results):
    """Merge per-chunk word lists into one absolute-time list.

    ``chunk_results`` = [((audio_start, audio_end, own_start, own_end), body)];
    word times in ``body`` are relative to ``audio_start``. Two chunks can
    time the same seam word tens of ms apart, putting it on both sides of the
    seam: a word repeated across chunks within SEAM_DEDUPE_SECONDS is kept once."""
    words = []
    for ci, ((a_s, _a_e, own_s, own_e), body) in enumerate(chunk_results):
        for w in (body or {}).get("words") or []:
            try:
                s = float(w["start"]) + a_s
                e = float(w["end"]) + a_s
            except (KeyError, TypeError, ValueError):
                continue
            text = str(w.get("text", w.get("word", "")) or "").strip()
            if not text or not (own_s - 1e-6 <= s < own_e):
                continue
            words.append({"word": f" {text}", "start": round(s, 3), "end": round(max(e, s), 3),
                          "probability": None, "_chunk": ci})
    words.sort(key=lambda w: w["start"])
    out = []
    for w in words:
        dup = next((p for p in reversed(out[-3:]) if p["_chunk"] != w["_chunk"]
                    and abs(p["start"] - w["start"]) <= SEAM_DEDUPE_SECONDS
                    and _seam_tok(p["word"]) == _seam_tok(w["word"])), None)
        if dup is None:
            out.append(w)
    for w in out:
        del w["_chunk"]
    return out


SEAM_DEDUPE_SECONDS = 0.25


def _seam_tok(text):
    return re.sub(r"[^\w']", "", str(text).lower())


def build_segments(words, gap=SEGMENT_GAP_SECONDS, max_len=SEGMENT_MAX_SECONDS):
    """Whisper-shaped segments from a flat word list: split after sentence-final
    punctuation, before a pause > ``gap`` seconds, or past ``max_len``."""
    segments, cur = [], []

    def flush():
        if cur:
            segments.append({"text": " ".join(w["word"].strip() for w in cur),
                             "start": cur[0]["start"], "end": cur[-1]["end"], "words": list(cur)})
            cur.clear()

    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_len):
            flush()
        cur.append(w)
        if _SENTENCE_END_RE.search(w["word"].strip()):
            flush()
    flush()
    return segments


def extract_audio(src, dst, start=0.0, duration=None, bitrate_kbps=GROK_AUDIO_BITRATE_KBPS):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-b:a", f"{bitrate_kbps}k", "-f", "mp3", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"audio extraction failed: {(r.stderr or '').strip()[-300:]}")


def transcribe_grok(media_path, targets, log=None, language=None, post=None):
    """Transcribe through the STT tiers. Raises on failure (caller falls back).
    ``engine`` is the tier that served: "grok-api" if any chunk needed the
    paid API, else "grok-proxy"."""
    log = log or (lambda *_: None)
    post = post or post_stt
    duration = media_duration(media_path)
    chunks = plan_chunks(duration)
    deadline = stage_deadline(duration)
    tmpdir = tempfile.mkdtemp(prefix="stt_")
    try:
        def run_chunk(i_chunk):
            i, ch = i_chunk
            a_s, a_e = ch[0], ch[1]
            path = os.path.join(tmpdir, f"chunk_{i}.mp3")
            single = len(chunks) == 1
            extract_audio(media_path, path, start=0.0 if single else a_s,
                          duration=None if single else max(0.1, a_e - a_s))
            with open(path, "rb") as f:
                audio = f.read()
            if len(audio) > 32 * 1024 * 1024:
                raise SttError("audio chunk too large for the transcriber")
            body, tier = post(audio, targets, language=language, filename=f"chunk_{i}.mp3",
                              log=log, deadline=deadline)
            return ch, body, tier

        if len(chunks) == 1:
            done = [run_chunk((0, chunks[0]))]
        else:
            log(f"   Long audio: transcribing in {len(chunks)} chunks")
            with ThreadPoolExecutor(max_workers=GROK_CONCURRENCY) as pool:
                done = list(pool.map(run_chunk, enumerate(chunks)))
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    results = [(ch, body) for ch, body, _ in done]
    tiers = {tier for _, _, tier in done}
    words = stitch_chunk_words(results)
    lang = next((b.get("language") for _, b in results if b.get("language")), None) or "unknown"
    resp_dur = float(results[0][1].get("duration") or 0) if len(results) == 1 else 0.0
    if str(lang).lower() in NO_SPACE_LANGUAGES:
        # Same token shape Whisper produces for these scripts (no leading space).
        for w in words:
            w["word"] = w["word"].lstrip()
    segments = build_segments(words)
    return {
        "text": " ".join(s["text"] for s in segments).strip(),
        "segments": segments,
        "language": lang,
        "language_probability": None,
        "duration": duration or resp_dur or (words[-1]["end"] if words else 0.0),
        "engine": "grok-api" if "grok-api" in tiers else "grok-proxy",
    }


def transcribe(media_path, word_timestamps=True, log=None, echo_segments=False):
    """Transcribe a local audio/video file: Grok STT (gateway, then the xAI
    API keys), then local faster-whisper.

    Returns {"text", "segments": [{"text","start","end","words":[{"word",
    "start","end","probability"}]}], "language", "language_probability",
    "duration", "engine"} — the openai-whisper shape every caller consumes;
    ``engine`` (the job's ``transcription_engine``) is grok-proxy | grok-api |
    whisper-small (whisper-<WHISPER_MODEL>).
    """
    log = log or (lambda *_: None)
    targets = stt_targets() if transcribe_engine() == "grok" else []
    if targets:
        try:
            t0 = time.time()
            result = transcribe_grok(media_path, targets, log=log)
            n = sum(len(s["words"]) for s in result["segments"])
            log(f"🎙️  Transcribed via {result['engine']} ({stt_model()}): {n} words, "
                f"language '{result['language']}', {time.time() - t0:.1f}s")
            if echo_segments:
                for seg in result["segments"]:
                    log(f"   [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
            return result
        except Exception as e:
            # SttError messages are status/failure-class only; anything else is
            # reduced to its type so no URL, key or text reaches the logs.
            why = str(e) if isinstance(e, SttError) else type(e).__name__
            log(f"⚠️  Grok transcription unavailable ({why}) — falling back to local Whisper.")
    return _transcribe_whisper(media_path, word_timestamps=word_timestamps, log=log,
                               echo_segments=echo_segments)
