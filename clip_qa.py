"""Post-render QA gate for every delivered clip.

Measured on the RENDERED file (ffprobe/ffmpeg), plus the source transcript
slice the clip covers:

  duration   within the requested clip band (+/- DURATION_SLACK)
  audio      an audio stream exists (when the source had one) and the
             integrated loudness is sane — not silent, not absurdly hot
  last frame not black (a dead frame as the final image of a short)
  ending     no word straddles the out-point (the last word must end before
             the clip does — the TEXT of a chopped word re-transcribes as whole,
             so this is checked on the source word timestamps instead), and
             the last word is not a false ending ("and", "the", "to", ...)
  double take no consecutive repeated 3-5-word phrase (a speaker restarting
             a sentence; Whisper exposes it as the same words twice)

``evaluate()`` is pure (thresholds only) so it is unit-tested without ffmpeg;
``qa_clip()`` does the measuring. The orchestrator drops failing clips but
never all of them (see ``pick_survivors``). Ported in spirit from
vibe-editing's reqc.py (MIT).
"""
import json
import os
import re
import subprocess

from clip_selection import ends_complete, norm_token, normalize_words

DURATION_SLACK = float(os.environ.get("CLIP_QA_DURATION_SLACK", "1.0"))
MIN_LUFS = float(os.environ.get("CLIP_QA_MIN_LUFS", "-45"))
MAX_LUFS = float(os.environ.get("CLIP_QA_MAX_LUFS", "-5"))
# The last frame is "black" when it is uniformly dark: mean AND max luma low.
BLACK_MEAN_LUMA = 12.0
BLACK_MAX_LUMA = 32.0
# A word may run this far past the out-point before the ending counts as
# chopped (Whisper end labels are +/- a few tens of ms).
CHOP_TOLERANCE = 0.08


def find_double_takes(words, sizes=(5, 4, 3)):
    """Consecutive repeated n-word phrases, longest first: [(time, phrase)].

    "we need to we need to fix it" -> [(t, "we need to")]. A phrase made of a
    single repeated token ("no no no no no no") is emphasis, not a restart,
    and is ignored."""
    ws = normalize_words(words)
    toks = [norm_token(w["w"]) for w in ws]
    hits, covered = [], set()
    for n in sizes:
        for i in range(len(toks) - 2 * n + 1):
            a, b = toks[i:i + n], toks[i + n:i + 2 * n]
            if a == b and all(a) and len(set(a)) > 1 and i not in covered:
                hits.append((round(ws[i]["s"], 2), " ".join(a)))
                covered.update(range(i, i + 2 * n))
    return sorted(hits)


def words_in_range(words, start, end):
    """Source words that START inside [start, end)."""
    return [w for w in normalize_words(words) if start - 1e-6 <= w["s"] < end]


def ending_problem(words, clip_start, clip_end, tolerance=CHOP_TOLERANCE):
    """Why the clip's ending is bad, or None. ``words`` = whole-source words."""
    all_words = normalize_words(words)
    inside = [i for i, w in enumerate(all_words) if clip_start - 1e-6 <= w["s"] < clip_end]
    if not inside:
        return None
    last = inside[-1]
    w = all_words[last]
    if w["s"] < clip_end - 0.05 and w["e"] > clip_end + tolerance:
        return (f"last word {w['w'].strip()!r} runs to {w['e']:.2f}s, past the clip end "
                f"{clip_end:.2f}s (chopped mid-word)")
    ok, reason = ends_complete(all_words, last)
    return None if ok else reason


def evaluate(metrics, min_secs, max_secs, expect_audio=True):
    """Pure verdict from measured ``metrics``. Returns {"passed", "failures",
    "metrics"}. Missing measurements (None) never fail a clip."""
    failures = []
    d = metrics.get("duration")
    if d is None or d <= 0:
        failures.append("duration: unreadable or zero")
    elif d < min_secs - DURATION_SLACK or d > max_secs + DURATION_SLACK:
        failures.append(f"duration {d:.1f}s outside {min_secs:g}-{max_secs:g}s")
    if expect_audio:
        if metrics.get("has_audio") is False:
            failures.append("audio: no audio stream")
        else:
            lufs = metrics.get("lufs")
            if lufs is not None and lufs < MIN_LUFS:
                failures.append(f"audio: silent ({lufs:.1f} LUFS)")
            elif lufs is not None and lufs > MAX_LUFS:
                failures.append(f"audio: too loud ({lufs:.1f} LUFS)")
    luma = metrics.get("last_frame_luma")
    if luma and luma.get("mean") is not None and luma.get("max") is not None:
        if luma["mean"] < BLACK_MEAN_LUMA and luma["max"] < BLACK_MAX_LUMA:
            failures.append("last frame is black")
    if metrics.get("ending_problem"):
        failures.append(f"ending: {metrics['ending_problem']}")
    for t, phrase in metrics.get("double_takes") or []:
        failures.append(f"double take at {t:.1f}s: '{phrase}'")
    return {"passed": not failures, "failures": failures, "metrics": metrics}


# --- measurements -----------------------------------------------------------

def probe(path):
    """{"duration", "has_audio", "has_video"} from ffprobe (None values on failure)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
             "-of", "json", path], capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out or "{}")
        types = [s.get("codec_type") for s in data.get("streams") or []]
        dur = data.get("format", {}).get("duration")
        return {"duration": float(dur) if dur not in (None, "N/A") else None,
                "has_audio": "audio" in types, "has_video": "video" in types}
    except Exception:
        return {"duration": None, "has_audio": None, "has_video": None}


_LUFS_RE = re.compile(r"I:\s+(-?inf|-?[\d.]+)\s+LUFS")


def integrated_lufs(path):
    """EBU R128 integrated loudness of the clip, or None when unmeasurable.
    Pure silence reports -70 LUFS (the gate floor)."""
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path,
                            "-map", "0:a:0", "-af", "ebur128", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=300)
        found = _LUFS_RE.findall(r.stderr or "")
        if not found:
            return None
        v = found[-1]
        return -70.0 if "inf" in v else float(v)
    except Exception:
        return None


def last_frame_luma(path):
    """{"mean","max"} luma (0-255) of a frame ~0.1s before the end, or None."""
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-sseof", "-0.25", "-i", path,
                            "-frames:v", "1", "-vf", "scale=32:32,format=gray",
                            "-f", "rawvideo", "-"], capture_output=True, timeout=60)
        data = r.stdout
        if not data:
            return None
        return {"mean": round(sum(data) / len(data), 2), "max": max(data)}
    except Exception:
        return None


def qa_clip(path, start, end, words, min_secs, max_secs, expect_audio=True):
    """Measure and judge one rendered clip. ``words`` are whole-source words
    (absolute times); ``start``/``end`` the clip's position in the source."""
    meta = probe(path)
    metrics = {"duration": meta["duration"], "has_audio": meta["has_audio"]}
    if expect_audio and meta["has_audio"]:
        metrics["lufs"] = integrated_lufs(path)
    if meta["has_video"] is not False:
        metrics["last_frame_luma"] = last_frame_luma(path)
    if words:
        metrics["ending_problem"] = ending_problem(words, float(start), float(end))
        metrics["double_takes"] = find_double_takes(words_in_range(words, float(start), float(end)))
    return evaluate(metrics, min_secs, max_secs, expect_audio=expect_audio)


def pick_survivors(results):
    """Indices of clips to keep. ``results`` = [(index, qa, predicted_score)].

    Everything that passed; when nothing passed, the single best failure
    (fewest QA failures, then highest predicted score) so a job never ends
    with zero clips because of the gate."""
    passed = [i for i, qa, _ in results if qa and qa.get("passed")]
    if passed or not results:
        return passed

    def rank(item):
        _, qa, score = item
        try:
            score = float(score or 0)
        except (TypeError, ValueError):
            score = 0.0
        return (len((qa or {}).get("failures") or []), -score)

    return [min(results, key=rank)[0]]
