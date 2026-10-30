"""Acoustic refinement of cut points: land each cut on the quietest ~10ms.

Word timestamps put a cut in the right gap; this puts it at the quietest
instant of that gap so the splice has no click and keeps the word's decay.
The search ranges come from clip_selection.quiet_search_window (in-point
-0.10/+0.05s around the first word's start, out-point -0.06/+0.20s around the
last word's end, never crossing a neighbouring word). Audio is read with a
fast-seek ffmpeg decode of just that window, so it costs ~50ms per cut even on
an hour-long source. Idea ported from vibe-editing's snap_silence.py (MIT).
"""
import subprocess

from clip_selection import quiet_search_window

SAMPLE_RATE = 16000


def read_window(src, t_lo, dur, sr=SAMPLE_RATE):
    """Mono float32 PCM of [t_lo, t_lo+dur] via ffmpeg, or None on failure."""
    try:
        import numpy as np
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, t_lo):.3f}", "-i", str(src),
             "-t", f"{max(0.02, dur):.3f}", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, check=True, timeout=30)
        return np.frombuffer(p.stdout, dtype=np.float32)
    except Exception:
        return None


def quietest_offset(samples, sr=SAMPLE_RATE, win=0.01):
    """Offset (seconds from the window start) of the centre of the quietest
    ``win``-second frame, scanning with 50% hop. None for an empty buffer."""
    import numpy as np
    y = np.asarray(samples, dtype=np.float64)
    if y.size == 0:
        return None
    fl = max(1, int(win * sr))
    if y.size <= fl:
        return y.size / 2.0 / sr
    hop = max(1, fl // 2)
    best_k, best_e = 0, None
    for k in range(0, y.size - fl + 1, hop):
        e = float(np.mean(y[k:k + fl] ** 2))
        if best_e is None or e < best_e:
            best_e, best_k = e, k
    return (best_k + fl / 2.0) / sr


def quietest(src, t_lo, t_hi, sr=SAMPLE_RATE):
    """Absolute time of the quietest ~10ms in [t_lo, t_hi] of ``src``, or None."""
    if t_hi - t_lo < 0.02:
        return None
    y = read_window(src, t_lo, t_hi - t_lo, sr)
    if y is None or len(y) == 0:
        return None
    off = quietest_offset(y, sr)
    return None if off is None else t_lo + off


def refine_clip_cuts(src, clip, video_duration):
    """Refine a snapped clip's start/end in place using the source audio.

    ``clip`` carries the snap anchors (``snap`` dict from snap_clip_bounds).
    Clips without anchors (vision path, fallbacks) are left alone. Returns
    True when anything moved."""
    snap = clip.get("snap") or {}
    moved = False
    if snap.get("word_start") is not None:
        win = quiet_search_window(snap["word_start"], "in", snap.get("prev_end"),
                                  video_duration=video_duration)
        t = quietest(src, *win) if win else None
        if t is not None:
            clip["start"] = round(max(0.0, t), 3)
            moved = True
    if snap.get("word_end") is not None:
        win = quiet_search_window(snap["word_end"], "out", next_start=snap.get("next_start"),
                                  video_duration=video_duration)
        t = quietest(src, *win) if win else None
        if t is not None:
            clip["end"] = round(min(float(video_duration), t) if video_duration else t, 3)
            moved = True
    if not (float(clip["end"]) > float(clip["start"])):
        # Never let refinement produce a degenerate clip: back to the word cut.
        clip["start"], clip["end"] = snap.get("cut_start", clip["start"]), snap.get("cut_end", clip["end"])
        return False
    return moved
