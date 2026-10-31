"""Clip tightening: drop fillers, dead air and double takes inside a clip.

Runs per clip between the bound refinement (audio_snap) and the render. The
plan is a list of KEEP intervals on the clip's own timeline; the cut step
(ffmpeg_utils.cut_clip with ``keep_ranges``) renders them in ONE encode with a
trim+concat filtergraph that uses the same intervals for video and audio, so
it replaces the plain cut rather than adding a lossy pass. Reframing, captions
and QA then work on the tightened clip, with the clip's words remapped onto
the new timeline (``remap_words``).

What is removed (only between two KEPT words — the clip's own edges were
already placed by clip_selection/audio_snap):

  fillers      "um", "uh", "erm", "ah", "uhm", "hmm" (+ spelling variants);
               "you know" / "I mean" only when flanked by pauses >= 200ms;
               "like" only standalone between pauses >= 200ms (and not a
               question); "right"/"okay"/"so" are never removed — between
               pauses they are usually the other speaker's answer
  double takes a 3-5-word phrase immediately repeated (clip_qa's rule); the
               FIRST take goes, the second (cleaner) take stays
  dead air     any gap >= 0.55s between kept words that is also QUIET in
               the audio (>= 90% of 20ms frames below max(-40 dBFS, noise
               floor + 15 dB)); a gap holding a laugh / backchannel / an
               unlabelled word tail shrinks to its quiet part or stays

and every removed stretch leaves ~0.15s of air (never less than a 0.12s
breath) at the join. Every join must also sit in quiet audio (+/-40ms), so a
filler glued to a neighbouring word, or a join landing in a word's tail, is
skipped rather than chopped.

No visible jump cuts: a removal inside a continuous shot happens only when the
last kept frame before the join and the first one after it look alike (mean
abs grey difference on 64x36 thumbnails <= MOTION_MAX_DIFF, and no more than
1.5x this clip's own p90 frame-to-frame motion), or when the join lands on a
scene change. Nothing is cut within SCENE_GUARD seconds of a scene
change unless the change itself falls inside the removed span.

Never over-tightens: at most TIGHTEN_MAX_REMOVED_FRACTION (0.25) of the clip,
and never below CLIP_MIN_SECONDS (removals are skipped instead). Join points
land on the quietest ~10ms near the planned point (audio_snap's measure), then
on the video frame grid so audio and video pieces have identical lengths and
sync cannot drift across joins.

Grok STT (transcription.py's primary engine) keeps fillers and repeats in its
words; Whisper usually drops fillers from the text, but the word gaps still
expose the dead air they leave.

Thresholds and the motion gate follow vibe-editing's detect_fillers.py /
reqc.py jump-cut scan (MIT), reimplemented here.
"""
import math
import os
import re
import subprocess
from fractions import Fraction

FILLER_TOKENS = frozenset({
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "erm", "ah", "ahh", "ahhh",
    "hmm", "hmmm", "hm", "mm", "mmm",
})
# Fillers only when standalone between pauses (otherwise they carry meaning).
# "right" / "okay" / "so" are NOT here: in a two-person interview a lone
# "Right." or "Okay." between pauses is usually the other speaker's answer,
# and we have no speaker labels to tell. A lone "like" is a hesitation unless
# it is a question ("Like?") or answers one.
ISOLATED_FILLERS = frozenset({"like"})
PHRASE_FILLERS = (("you", "know"), ("i", "mean"))

PAUSE_FLANK = 0.20          # pause that makes a word/phrase "standalone"
DEAD_AIR_MIN = 0.55         # gaps at least this long are trimmed
KEEP_GAP = 0.15             # air left at every join
MIN_BREATH = 0.12           # never less air than this between sentences
MIN_REMOVE = 0.10           # removals shorter than this are not worth a join
MIN_SHRUNK_REMOVE = 0.25    # ...and dead air shrunk by the audio check below this
SNAP_SLACK = 0.05           # quietest-point search either side of a planned join
WORD_GUARD = 0.03           # joins never land closer than this to a kept word
MOTION_MAX_DIFF = 14.0      # mean abs grey diff (0-255) on 64x36 thumbnails: hard ceiling
# The ceiling alone let visible head jumps through on handheld phone footage
# (joins at 10-14 against a natural frame-to-frame p90 of ~6). A join must
# also look like ordinary motion for THIS clip: <= MOTION_NATURAL_FACTOR x
# the p90 of consecutive-frame diffs, but never stricter than MOTION_FLOOR
# (static shots have near-zero natural motion and sensor noise alone ~2-4).
MOTION_NATURAL_FACTOR = 1.5
MOTION_FLOOR = 6.0
SCENE_GUARD = 0.30          # no joins this close to a scene change
SCENE_MIN_DIFF = 28.0       # thumbnail diff that counts as a hard cut
THUMB_W, THUMB_H = 64, 36
# Audio checks (see plan): 20ms RMS frames; "quiet" = below
# max(QUIET_ABS_DB, noise floor + QUIET_OVER_FLOOR_DB).
QUIET_STEP = 0.02
QUIET_ABS_DB = -40.0
QUIET_OVER_FLOOR_DB = 15.0
QUIET_SPAN_FRACTION = 0.90
JOIN_QUIET_MARGIN = 0.04

_TERMINAL_RE = re.compile(r"[.!?…]['\")\]]*$")


def enabled(cli_flag=True):
    """TIGHTEN_CLIPS (default on) AND the per-job flag."""
    return bool(cli_flag) and os.environ.get("TIGHTEN_CLIPS", "1").strip().lower() not in (
        "0", "false", "off", "no")


def max_removed_fraction():
    try:
        v = float(os.environ.get("TIGHTEN_MAX_REMOVED_FRACTION", "0.25"))
    except ValueError:
        v = 0.25
    return min(max(v, 0.0), 0.5)


def join_fade_seconds():
    """8-12ms fade on both sides of every join (TIGHTEN_JOIN_FADE_MS, default 10)."""
    try:
        ms = float(os.environ.get("TIGHTEN_JOIN_FADE_MS", "10"))
    except ValueError:
        ms = 10.0
    return min(max(ms, 8.0), 12.0) / 1000.0


def _tok(text):
    return re.sub(r"[^\w']", "", str(text or "").lower())


def _norm(words):
    """[{'word','start','end'}] sorted by start; untimed entries dropped."""
    out = []
    for w in words or []:
        if not isinstance(w, dict):
            continue
        try:
            s = float(w.get("start", w.get("s")))
            e = float(w.get("end", w.get("e")))
        except (TypeError, ValueError):
            continue
        out.append({"word": str(w.get("word", w.get("w", ""))), "start": s, "end": max(s, e)})
    out.sort(key=lambda w: w["start"])
    return out


# --- what to remove ----------------------------------------------------------

def find_removable_words(words):
    """{index: reason} of words to remove: 'filler' or 'double_take'.

    ``words`` = the clip's words (sorted). Pause tests use the neighbouring
    words' timestamps; the first/last word have no neighbour on one side and
    count as flanked there (they are never removed anyway — see plan)."""
    n = len(words)
    toks = [_tok(w["word"]) for w in words]

    def gap_before(i):
        return words[i]["start"] - words[i - 1]["end"] if i > 0 else 99.0

    def gap_after(i):
        return words[i + 1]["start"] - words[i]["end"] if i + 1 < n else 99.0

    out = {}
    for i, t in enumerate(toks):
        if t in FILLER_TOKENS:
            out[i] = "filler"
        elif (t in ISOLATED_FILLERS and gap_before(i) >= PAUSE_FLANK and gap_after(i) >= PAUSE_FLANK
              and "?" not in words[i]["word"] and not (i and "?" in words[i - 1]["word"])):
            out[i] = "filler"
    for phrase in PHRASE_FILLERS:
        k = len(phrase)
        for i in range(n - k + 1):
            if tuple(toks[i:i + k]) == phrase and gap_before(i) >= PAUSE_FLANK \
                    and gap_after(i + k - 1) >= PAUSE_FLANK:
                for j in range(i, i + k):
                    out[j] = "filler"
    # Double takes: same rule as clip_qa.find_double_takes (longest first, a
    # phrase of one repeated token is emphasis); the first take is removed.
    covered = set()
    for size in (5, 4, 3):
        for i in range(n - 2 * size + 1):
            a, b = toks[i:i + size], toks[i + size:i + 2 * size]
            if a == b and all(a) and len(set(a)) > 1 and not covered.intersection(range(i, i + 2 * size)):
                covered.update(range(i, i + 2 * size))
                for j in range(i, i + size):
                    out[j] = "double_take"
    return out


def candidate_removals(words, removable):
    """Removal candidates between consecutive KEPT words.

    Returns [{"r0","r1","kind","prev_end","next_start","lo_limit","hi_limit"}]
    in source seconds. ``lo_limit``/``hi_limit`` bound where the quietest-point
    refinement may move r0/r1 without cutting into a kept or removed word's
    wrong side."""
    kept = [i for i in range(len(words)) if i not in removable]
    out = []
    for a, b in zip(kept, kept[1:]):
        wa, wb = words[a], words[b]
        gone = list(range(a + 1, b))
        span = wb["start"] - wa["end"]
        if span <= 0:
            continue
        if gone:
            kinds = {removable[j] for j in gone}
            kind = "double_take" if "double_take" in kinds else "filler"
        elif span >= DEAD_AIR_MIN:
            kind = "dead_air"
        else:
            continue
        keep = max(KEEP_GAP, MIN_BREATH)
        r0 = wa["end"] + keep / 2
        r1 = wb["start"] - keep / 2
        if gone:
            # The removed words must go entirely, even when that leaves less air.
            r0 = min(r0, words[gone[0]]["start"])
            r1 = max(r1, words[gone[-1]]["end"])
            r0 = max(r0, wa["end"] + WORD_GUARD) if words[gone[0]]["start"] > wa["end"] + WORD_GUARD else r0
            r1 = min(r1, wb["start"] - WORD_GUARD) if words[gone[-1]]["end"] < wb["start"] - WORD_GUARD else r1
        if r1 - r0 < MIN_REMOVE:
            continue
        out.append({
            "r0": r0, "r1": r1, "kind": kind,
            "prev_end": wa["end"], "next_start": wb["start"],
            # r0 may move within [prev_end+guard, first removed word start]
            "r0_hi": words[gone[0]]["start"] if gone else wb["start"] - WORD_GUARD,
            "r1_lo": words[gone[-1]]["end"] if gone else wa["end"] + WORD_GUARD,
        })
    return out


# --- planning ----------------------------------------------------------------

def _fps_float(fps):
    if not fps:
        return None
    try:
        return float(Fraction(fps)) if not isinstance(fps, float) else fps
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def frame_grid(t, fps):
    """``t`` (clip seconds) rounded to the nearest frame boundary."""
    if not fps:
        return t
    fps = Fraction(fps).limit_denominator(1001000)
    return float(Fraction(round(Fraction(t).limit_denominator(10 ** 6) * fps)) / fps)


def grid_ceil(t, fps):
    if not fps:
        return t
    f = Fraction(fps).limit_denominator(1001000)
    return float(math.ceil(Fraction(t).limit_denominator(10 ** 6) * f - Fraction(1, 10 ** 6)) / f)


def grid_floor(t, fps):
    if not fps:
        return t
    f = Fraction(fps).limit_denominator(1001000)
    return float(math.floor(Fraction(t).limit_denominator(10 ** 6) * f + Fraction(1, 10 ** 6)) / f)


def _longest_quiet_run(mask, t0, step):
    """(start, end) seconds of the longest all-quiet run in ``mask`` or None."""
    best, cur = None, None
    for k, q in enumerate(mask):
        if q:
            cur = k if cur is None else cur
            if best is None or (k + 1 - cur) > (best[1] - best[0]):
                best = (cur, k + 1)
        else:
            cur = None
    return None if best is None else (t0 + best[0] * step, t0 + best[1] * step)


def plan(words, clip_start, clip_end, *, min_clip_seconds, max_fraction=None,
         fps=None, frame_diff=None, scene_cuts=(), quietest=None, motion_threshold=None,
         quiet=None):
    """Plan the tightening of [clip_start, clip_end] (source seconds).

    ``words`` source words (absolute). ``frame_diff(t_a, t_b)`` -> grey diff
    between the frames at absolute source times (None = unmeasurable, the
    removal is skipped). ``scene_cuts`` absolute times of hard cuts inside the
    clip. ``quietest(lo, hi)`` -> absolute time of the quietest ~10ms, or None.
    ``quiet(lo, hi)`` -> per-QUIET_STEP booleans "this 20ms is quiet" over
    [lo, hi] (see quiet_fn); None skips the audio checks (unit tests only —
    plan_for_clip always measures).

    Audio checks (word gaps alone are not enough: a backchannel "mm-hm", a
    laugh or a trailing word Whisper/Grok did not label sits in many "gaps"):
      * dead air must be >= QUIET_SPAN_FRACTION quiet, else it shrinks to the
        longest quiet run inside it (keeping KEEP_GAP/2 of air next to the
        sound) or is skipped;
      * every join must sit in quiet audio (JOIN_QUIET_MARGIN either side) —
        a filler glued to a word, or a cut landing in a word's tail, is skipped.

    Returns {"keep": [(s, e)] clip-relative, "removed": [(s, e, kind)]
    absolute, "removed_seconds", "cuts", "skipped_for_motion",
    "skipped_for_audio", "skipped_for_cap", "skipped_double_takes",
    "duration", "fps"}."""
    clip_start, clip_end = float(clip_start), float(clip_end)
    dur = clip_end - clip_start
    motion_threshold = MOTION_MAX_DIFF if motion_threshold is None else motion_threshold
    max_fraction = max_removed_fraction() if max_fraction is None else max_fraction
    result = {"keep": [(0.0, dur)], "removed": [], "removed_seconds": 0.0, "cuts": 0,
              "skipped_for_motion": 0, "skipped_for_audio": 0, "skipped_for_cap": 0,
              "skipped_double_takes": 0, "duration": dur, "fps": str(fps) if fps else None}
    ws = [w for w in _norm(words) if clip_start - 1e-6 <= w["start"] and w["end"] <= clip_end + 0.05]
    if dur <= 0 or len(ws) < 2:
        return result
    budget = min(max_fraction * dur, dur - float(min_clip_seconds))
    if budget < MIN_REMOVE:
        return result

    cands = candidate_removals(ws, find_removable_words(ws))
    fpsf = _fps_float(fps)
    frame = (1.0 / fpsf) if fpsf else 1.0 / 30
    # The cut ends on the last whole frame (ffmpeg_utils.tightened_pieces).
    dur_grid = grid_floor(dur, fps) if fps else dur

    def join_quiet(t):
        if quiet is None:
            return True
        m = quiet(t - JOIN_QUIET_MARGIN, t + JOIN_QUIET_MARGIN)
        return bool(m) and sum(m) >= 0.75 * len(m)

    for c in cands:
        # Land each join on the quietest point near its plan, inside its limits.
        if quietest:
            lo, hi = max(c["prev_end"] + WORD_GUARD, c["r0"] - SNAP_SLACK), min(c["r0_hi"], c["r0"] + SNAP_SLACK)
            t = quietest(lo, hi) if hi - lo >= 0.02 else None
            if t is not None:
                c["r0"] = t
            lo, hi = max(c["r1_lo"], c["r1"] - SNAP_SLACK), min(c["next_start"] - WORD_GUARD, c["r1"] + SNAP_SLACK)
            t = quietest(lo, hi) if hi - lo >= 0.02 else None
            if t is not None:
                c["r1"] = t
        # Frame grid (clip-relative) so every piece is a whole number of
        # frames — but never rounded INTO a kept word.
        q0 = frame_grid(c["r0"] - clip_start, fps)
        q1 = frame_grid(c["r1"] - clip_start, fps)
        if q0 < c["prev_end"] - clip_start:
            q0 = grid_ceil(c["prev_end"] - clip_start, fps)
        if q1 > c["next_start"] - clip_start:
            q1 = grid_floor(c["next_start"] - clip_start, fps)
        c["q0"], c["q1"] = q0, q1

    # Priority: double takes, fillers, then the longest dead air first.
    order = {"double_take": 0, "filler": 1, "dead_air": 2}
    cands.sort(key=lambda c: (order[c["kind"]], -(c["q1"] - c["q0"])))
    chosen = []
    for c in cands:
        ok = _choose(c, result, clip_start, dur_grid, frame, fps, quiet, join_quiet,
                     scene_cuts, frame_diff, motion_threshold, budget)
        if ok:
            budget -= c["q1"] - c["q0"]
            chosen.append(c)
        elif c["kind"] == "double_take":
            result["skipped_double_takes"] += 1

    if not chosen:
        return result
    chosen.sort(key=lambda c: c["q0"])
    keep, cursor = [], 0.0
    for c in chosen:
        if c["q0"] > cursor:
            keep.append((cursor, c["q0"]))
        cursor = max(cursor, c["q1"])
    if cursor < dur_grid:
        keep.append((cursor, dur_grid))
    removed = round(sum(c["q1"] - c["q0"] for c in chosen), 3)
    result.update(keep=[(round(s, 9), round(e, 9)) for s, e in keep],
                  removed=[(round(clip_start + c["q0"], 3), round(clip_start + c["q1"], 3), c["kind"])
                           for c in chosen],
                  removed_seconds=removed, cuts=len(chosen),
                  duration=round(sum(e - s for s, e in keep), 9))
    return result


def _choose(c, result, clip_start, dur, frame, fps, quiet, join_quiet, scene_cuts,
            frame_diff, motion_threshold, budget):
    """Gate one candidate (audio, scene, motion, cap); may shrink it. Counts
    the reason in ``result`` when it is skipped."""
    if c["q1"] - c["q0"] < MIN_REMOVE or c["q0"] <= 0 or c["q1"] >= dur:
        return False
    a, b = clip_start + c["q0"], clip_start + c["q1"]
    if quiet is not None and c["kind"] == "dead_air":
        m = quiet(a, b)
        if not m:
            result["skipped_for_audio"] += 1
            return False
        loud_a, loud_b = not join_quiet(a), not join_quiet(b)
        if sum(m) < QUIET_SPAN_FRACTION * len(m) or loud_a or loud_b:
            # Shrink to the longest quiet run, keeping KEEP_GAP/2 of air next
            # to whatever sound bounds it (a laugh, a word's unlabelled tail).
            run = _longest_quiet_run(m, a, QUIET_STEP)
            if run is None:
                result["skipped_for_audio"] += 1
                return False
            ra, rb = run
            ra = ra + KEEP_GAP / 2 if (ra > a + 1e-6 or loud_a) else ra
            rb = rb - KEEP_GAP / 2 if (rb < b - 1e-6 or loud_b) else rb
            c["q0"] = grid_ceil(ra - clip_start, fps)
            c["q1"] = grid_floor(rb - clip_start, fps)
            if c["q1"] - c["q0"] < MIN_SHRUNK_REMOVE:
                result["skipped_for_audio"] += 1
                return False
            a, b = clip_start + c["q0"], clip_start + c["q1"]
    if not (join_quiet(a) and join_quiet(b)):
        result["skipped_for_audio"] += 1
        return False
    # Scene gate: a cut inside the span = the join IS the cut (fine); a cut
    # just outside it would leave a few-frame flash of one shot.
    inside = [s for s in scene_cuts if a - frame <= s <= b + frame]
    near = [s for s in scene_cuts if not (a - frame <= s <= b + frame)
            and (a - SCENE_GUARD < s < a or b < s < b + SCENE_GUARD)]
    if near:
        result["skipped_for_motion"] += 1
        return False
    if not inside:
        diff = frame_diff(a - frame, b) if frame_diff else None
        if diff is None or diff > motion_threshold:
            result["skipped_for_motion"] += 1
            return False
    size = c["q1"] - c["q0"]
    if size > budget:
        if c["kind"] != "dead_air" or budget < MIN_REMOVE:
            result["skipped_for_cap"] += 1
            return False
        # Partial trim: shrink symmetrically around the middle (stays inside
        # the already-checked span, so the gates above still hold).
        mid = (c["q0"] + c["q1"]) / 2
        q0 = grid_ceil(mid - budget / 2, fps)
        q1 = grid_floor(mid + budget / 2, fps)
        if q1 - q0 < MIN_REMOVE:
            result["skipped_for_cap"] += 1
            return False
        if frame_diff and not inside:
            diff = frame_diff(clip_start + q0 - frame, clip_start + q1)
            if diff is None or diff > motion_threshold:
                result["skipped_for_motion"] += 1
                return False
        c["q0"], c["q1"] = q0, q1
    return True


# --- new timeline ------------------------------------------------------------

def remap_time(t, keep):
    """Clip-relative time ``t`` -> tightened-timeline time. A time inside a
    removed span lands on the join. ``keep`` = clip-relative keep intervals."""
    out = 0.0
    for s, e in keep:
        if t < s:
            return out
        if t <= e:
            return out + (t - s)
        out += e - s
    return out


def remap_words(words, clip_start, clip_end, keep, context=False):
    """The clip's words on the tightened timeline.

    Words whose midpoint falls in a removed span are dropped (they were cut).
    With ``context`` the source words before/after the clip are kept too
    (shifted rigidly, before -> negative times) so clip_qa's ending checks can
    still look at what follows the out-point."""
    clip_start, clip_end = float(clip_start), float(clip_end)
    new_dur = sum(e - s for s, e in keep)
    out = []
    for w in _norm(words):
        mid = (w["start"] + min(w["end"], clip_end)) / 2 - clip_start
        if w["start"] < clip_start - 1e-6 or w["start"] >= clip_end:
            if not context:
                continue
            shift = -clip_start if w["start"] < clip_start else new_dur - clip_end
            out.append({**w, "start": round(w["start"] + shift, 3), "end": round(w["end"] + shift, 3)})
            continue
        if not any(s - 1e-6 <= mid <= e + 1e-6 for s, e in keep):
            continue
        start = remap_time(w["start"] - clip_start, keep)
        if w["end"] > clip_end:
            # Runs past the out-point: shift its end rigidly so clip_qa still
            # sees the chopped / false ending on the new timeline.
            end = w["end"] - clip_end + new_dur
        else:
            end = remap_time(w["end"] - clip_start, keep)
        out.append({**w, "start": round(start, 3), "end": round(max(start, end), 3)})
    return out


def words_transcript(words, language=None):
    """A transcript dict (segments/words) around already-timed words, for the
    caption burners and the per-clip metadata."""
    from transcription import build_segments
    segs = build_segments([{"word": w["word"] if str(w["word"]).startswith(" ") else f" {w['word']}",
                            "start": w["start"], "end": w["end"]} for w in words])
    return {"language": language, "timeline": "clip", "segments": segs,
            "text": " ".join(s["text"] for s in segs).strip()}


def summary(p):
    return {"removed_seconds": round(float(p.get("removed_seconds") or 0), 2),
            "cuts": int(p.get("cuts") or 0),
            "skipped_for_motion": int(p.get("skipped_for_motion") or 0),
            "skipped_for_audio": int(p.get("skipped_for_audio") or 0),
            "skipped_double_takes": int(p.get("skipped_double_takes") or 0)}


# --- measurements on the source ---------------------------------------------

def _rate(value):
    try:
        f = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return f if 1 <= f <= 240 else None


def video_fps(path):
    """The rate the tightened cut normalises to: the AVERAGE frame rate
    (r_frame_rate is the stream's base rate — 30 for a 27.5 fps VFR screen
    recording). Falls back to r_frame_rate when avg is unknown. A Fraction or
    None. VFR is fine: the cut and the thumbnails both resample to this rate."""
    try:
        import json
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=60).stdout
        st = (json.loads(out or "{}").get("streams") or [{}])[0]
        return _rate(st.get("avg_frame_rate")) or _rate(st.get("r_frame_rate"))
    except Exception:
        return None


def read_thumbnails(src, start, end, fps=None):
    """[start, end] as 64x36 grey frames (numpy uint8 [n, 36, 64]), or None.
    With ``fps`` resampled exactly like the tightened cut (CFR from t=0), so
    index k is the frame the cut shows at k/fps."""
    try:
        import numpy as np
        rate = ""
        if fps:
            f = Fraction(fps).limit_denominator(1001000)
            rate = f"fps=fps={f.numerator}/{f.denominator}:start_time=0,"
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", str(src), "-an", "-vf", f"{rate}scale={THUMB_W}:{THUMB_H}:flags=area,format=gray",
             "-f", "rawvideo", "-"], capture_output=True, timeout=900)
        n = len(p.stdout) // (THUMB_W * THUMB_H)
        if p.returncode != 0 or n < 2:
            return None
        return np.frombuffer(p.stdout[:n * THUMB_W * THUMB_H], dtype=np.uint8).reshape(n, THUMB_H, THUMB_W)
    except Exception:
        return None


def scene_cuts_from_thumbs(thumbs, fps, clip_start):
    """Absolute times of hard cuts: a frame-to-frame diff that is both large
    and an outlier against the clip's normal motion."""
    import numpy as np
    if thumbs is None or len(thumbs) < 3:
        return []
    d = np.mean(np.abs(np.diff(thumbs.astype(np.int16), axis=0)), axis=(1, 2))
    thr = max(SCENE_MIN_DIFF, float(np.median(d)) * 4 + 10)
    return [clip_start + (k + 1) / float(fps) for k in np.nonzero(d >= thr)[0]]


def motion_threshold_from_thumbs(thumbs):
    """Join-diff threshold for this clip (see MOTION_NATURAL_FACTOR)."""
    import numpy as np
    if thumbs is None or len(thumbs) < 3:
        return MOTION_FLOOR
    d = np.mean(np.abs(np.diff(thumbs.astype(np.int16), axis=0)), axis=(1, 2))
    natural = float(np.percentile(d, 90)) * MOTION_NATURAL_FACTOR
    return min(MOTION_MAX_DIFF, max(MOTION_FLOOR, natural))


def thumb_diff_fn(thumbs, fps, clip_start):
    """frame_diff(t_a, t_b) over pre-read thumbnails (absolute source times)."""
    import numpy as np
    n = len(thumbs)

    def diff(ta, tb):
        ia = int(round((ta - clip_start) * float(fps)))
        ib = int(round((tb - clip_start) * float(fps)))
        if not (0 <= ia < n and 0 <= ib < n):
            return None
        return float(np.mean(np.abs(thumbs[ia].astype(np.int16) - thumbs[ib].astype(np.int16))))
    return diff


def quietest_fn(src, clip_start, clip_end, samples=None):
    """quietest(lo, hi) over the clip's audio, read once (or ``samples``,
    starting at clip_start - 0.5s)."""
    import audio_snap
    pad = 0.5
    t0 = max(0.0, clip_start - pad)
    y = samples if samples is not None else audio_snap.read_window(src, t0, clip_end - t0 + pad)
    if y is None or len(y) == 0:
        return None
    sr = audio_snap.SAMPLE_RATE

    def quietest(lo, hi):
        a, b = int((lo - t0) * sr), int((hi - t0) * sr)
        if b - a < int(0.02 * sr) or a < 0 or b > len(y):
            return None
        off = audio_snap.quietest_offset(y[a:b], sr)
        return None if off is None else lo + off
    return quietest


def quiet_fn(src, clip_start, clip_end, samples=None, sr=None):
    """quiet(lo, hi) -> [bool per 20ms frame] over the clip's audio, read once.

    Quiet = 20ms RMS below max(QUIET_ABS_DB, noise floor + QUIET_OVER_FLOOR_DB),
    the floor being the clip's 10th-percentile frame. ``samples`` (mono
    float32 starting at clip_start - 0.5s) skips the read."""
    import numpy as np
    import audio_snap
    pad = 0.5
    t0 = max(0.0, clip_start - pad)
    y = samples if samples is not None else audio_snap.read_window(src, t0, clip_end - t0 + pad)
    sr = sr or audio_snap.SAMPLE_RATE
    if y is None or len(y) == 0:
        return None
    hop = int(QUIET_STEP * sr)
    n = len(y) // hop
    if n < 3:
        return None
    frames = np.asarray(y[:n * hop], dtype=np.float64).reshape(n, hop)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    a0, a1 = int((clip_start - t0) / QUIET_STEP), int((clip_end - t0) / QUIET_STEP)
    floor = float(np.percentile(db[max(0, a0):max(a0 + 1, min(n, a1))], 10))
    thr = max(QUIET_ABS_DB, floor + QUIET_OVER_FLOOR_DB)
    is_quiet = db < thr

    def quiet(lo, hi):
        i0, i1 = int(round((lo - t0) / QUIET_STEP)), int(round((hi - t0) / QUIET_STEP))
        i0, i1 = max(0, i0), min(n, max(i1, i0 + 1))
        return [bool(q) for q in is_quiet[i0:i1]]
    return quiet


def plan_for_clip(src, clip_start, clip_end, words, min_clip_seconds, log=print):
    """Measure the source and plan one clip. Never raises: on any problem the
    plan is a no-op (the clip renders untightened)."""
    try:
        fps = video_fps(src)
        if not fps:
            return plan([], clip_start, clip_end, min_clip_seconds=min_clip_seconds)
        # Cheap pre-check: nothing to remove -> skip the frame decode.
        ws = [w for w in _norm(words) if clip_start - 1e-6 <= w["start"] and w["end"] <= clip_end + 0.05]
        if not candidate_removals(ws, find_removable_words(ws)):
            return plan([], clip_start, clip_end, min_clip_seconds=min_clip_seconds)
        thumbs = read_thumbnails(src, clip_start, clip_end, fps=fps)
        if thumbs is None:
            log("   ⚠️ Tightening skipped: could not read frames for the motion check.")
            return plan([], clip_start, clip_end, min_clip_seconds=min_clip_seconds)
        import audio_snap
        t0 = max(0.0, clip_start - 0.5)
        samples = audio_snap.read_window(src, t0, clip_end - t0 + 0.5)
        quiet = quiet_fn(src, clip_start, clip_end, samples=samples) if samples is not None else None
        if quiet is None:
            log("   ⚠️ Tightening skipped: could not read the audio for the silence check.")
            return plan([], clip_start, clip_end, min_clip_seconds=min_clip_seconds)
        return plan(words, clip_start, clip_end, min_clip_seconds=min_clip_seconds, fps=fps,
                    frame_diff=thumb_diff_fn(thumbs, fps, clip_start),
                    scene_cuts=scene_cuts_from_thumbs(thumbs, fps, clip_start),
                    quietest=quietest_fn(src, clip_start, clip_end, samples=samples),
                    motion_threshold=motion_threshold_from_thumbs(thumbs), quiet=quiet)
    except Exception as e:
        log(f"   ⚠️ Tightening skipped ({type(e).__name__}: {e}).")
        return plan([], clip_start, clip_end, min_clip_seconds=min_clip_seconds)
