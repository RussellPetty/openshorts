"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""
import re

# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing.
MODEL_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}


def lookup_model_prices(model_name):
    """Longest-prefix match against MODEL_PRICES; None if unknown."""
    name = str(model_name or "").lower()
    best_key = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICES[best_key] if best_key else None


def clip_count_targets(n_windows):
    """How many clips to ask the detail pass for, given the shortlist size.

    Measured on prod 3-ago-2026: 408 of 429 jobs (95%) delivered 3 clips or
    fewer, the mode being ONE, while the prompt was free to return one per
    shortlisted window. Users who received 1-3 clips came back a second day
    0.4% of the time; those who received 4-9 came back 16.1% — so the clip
    count, not the clip quality, is what the retention curve hangs on.

    The old prompt biased hard the other way ("prefer one great clip per
    candidate window") and handed the model two unbounded licences to drop
    clips (the 2-second rule and STANDS ALONE both end in "or skip it"), with
    no floor to stop it collapsing to a single clip. This puts a floor and a
    realistic ceiling on it instead.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` override both for A/B runs
    without a deploy (the reframe-testing harness drives them).
    """
    import os

    n = max(1, int(n_windows or 1))
    # Floor grows with the material: 3 windows -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Ceiling allows a rich window to yield more than one without inviting padding.
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def shortlist_target(video_duration):
    """How many scored windows the detail pass should get.

    Scales with duration so a long video surfaces more candidates without
    exploding the detail call, floored at 3 so a short one still gets a
    choice. ``score_batches`` exists to make this number reachable.
    """
    try:
        seconds = float(video_duration or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return max(3, min(10, int(seconds // 90) + 2))


def score_batches(windows, batch_size):
    """Split ``windows`` into near-equal scoring batches.

    Two things were wrong with the old ``range(0, n, batch_size)`` walk plus a
    prompt that said "choose up to 3 windows from this batch".

    The cap made the shortlist unreachable. Its real ceiling was
    ``3 * n_batches``, not ``shortlist_target``: measured 22-sep-2026 on a
    9:10 source, 9 windows batched 8 + 1 returned 3 + 1 = 4 windows against a
    target of 8. ``clip_count_targets`` then derives the clip floor from that
    halved shortlist, so the detail pass was asked for 4-8 clips instead of
    6-12 and the job delivered 4 (3 in prod on the same video). Every source
    under ~30 minutes was starved the same way, which is the mechanism behind
    "95% of jobs deliver 3 clips or fewer" — the clip count the retention
    curve hangs on (see clip_count_targets). The prompt now scores every
    window it is given and the shortlist is the global top ``target``, so the
    selection happens where the scores can actually be compared.

    Per-batch selection also threw away the ranking it was computing: a batch
    holding five great moments could only contribute three, while a batch of
    filler still contributed its own three.

    The remainder was its own bug: a trailing batch of one window came back
    with that window whatever its score, because a cap of 3 cannot filter a
    batch of 1 — the tail of the video entered the shortlist by arithmetic
    rather than merit (``window_009`` did exactly that on the measured run).
    Near-equal batches keep that from happening and also keep each score
    comparable, since a window judged alone is judged against nothing.
    """
    windows = list(windows or [])
    n = len(windows)
    if not n:
        return []
    size = max(1, int(batch_size or 1))
    n_batches = max(1, -(-n // size))       # ceil
    base, extra = divmod(n, n_batches)

    out = []
    start = 0
    for index in range(n_batches):
        take = base + (1 if index < extra else 0)
        batch = windows[start:start + take]
        start += take
        if batch:
            out.append(batch)
    return out


def trim_to_best(shorts, max_clips):
    """Cut an over-long detail-pass result down to ``max_clips`` BY SCORE.

    The detail pass hands its clips back in transcript order, batch after
    batch, so slicing the list keeps the EARLIEST clips rather than the best
    ones. On a 9-minute walkthrough that quietly threw away everything past
    minute three: the model proposed clips across the whole video, and the
    ones covering the demo, the MCP walkthrough and the close were the tail
    that got dropped. Worse, the failure scales the wrong way — the more
    generous the model is, the more of the video disappears.

    That sabotages the windowing: get_viral_clips builds scoring windows
    precisely because "a single call over the whole transcript clusters picks
    near the start", and a positional slice puts the clustering right back.

    Ranking is by ``predicted_score`` (the detail prompt already asks for it,
    and nothing else was reading it here). Ties keep transcript order, and the
    survivors come back in transcript order too, so clip numbering still runs
    front to back the way every caller downstream expects.
    """
    max_clips = max(1, int(max_clips or 1))
    if len(shorts) <= max_clips:
        return list(shorts)

    def score(item):
        try:
            return float(item[1].get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    indexed = list(enumerate(shorts))
    best = sorted(indexed, key=score, reverse=True)[:max_clips]
    return [item for _, item in sorted(best, key=lambda pair: pair[0])]


def clip_duration_bounds():
    """The clip length band (seconds) the selection prompts and word-snapping
    enforce. ``CLIP_MIN_SECONDS`` / ``CLIP_MAX_SECONDS`` override the classic
    15-60 — set per job by /api/process when the user asks for a specific
    length, or by hand for A/B runs. Values are clamped to platform-sane
    limits and re-ordered so bad input degrades instead of breaking the job.
    """
    import os

    def _read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except ValueError:
            return default

    lo = _read("CLIP_MIN_SECONDS", 15.0)
    hi = _read("CLIP_MAX_SECONDS", 60.0)
    lo = min(max(lo, 5.0), 175.0)
    hi = min(max(hi, 10.0), 180.0)
    if hi < lo + 5.0:  # keep a real band: degenerate ranges starve the model
        hi = min(180.0, lo + 5.0)
    return round(lo, 3), round(hi, 3)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def dedupe_overlapping(shorts, ratio=0.5):
    """Drop clips that overlap an already-kept clip by ``ratio`` of the
    shorter one, keeping the higher ``predicted_score`` (earlier on a tie).

    The detail prompt's DIVERSITY rule is the only thing that stopped two
    clips from sharing the same seconds, and a rule is not a guarantee: two
    picks from one window can cover the same moment with different edges,
    and the user then downloads the same short twice. Survivors come back in
    transcript order, like ``trim_to_best``.
    """
    def _score(c):
        try:
            return float(c.get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _span(c):
        try:
            return float(c.get("start", 0)), float(c.get("end", 0))
        except (TypeError, ValueError, AttributeError):
            return 0.0, 0.0

    indexed = list(enumerate(shorts))
    kept = []
    for idx, clip in sorted(indexed, key=lambda p: (-_score(p[1]), p[0])):
        s, e = _span(clip)
        clash = False
        for _, other in kept:
            os_, oe = _span(other)
            overlap = min(e, oe) - max(s, os_)
            shorter = max(1e-6, min(e - s, oe - os_))
            if overlap > 0 and overlap / shorter >= ratio:
                clash = True
                break
        if not clash:
            kept.append((idx, clip))
    return [c for _, c in sorted(kept, key=lambda p: p[0])]



# --- Word-level boundary helpers ---------------------------------------------
# LLMs are bad at millisecond arithmetic; the word timestamps are ground truth.
# The rules below are the "never end mid-thought" gate: a clip starts where a
# sentence (or at least a phrase) starts and ends where one ends, and it never
# ends on a word that cannot close a sentence (a "false ending" — the speaker
# keeps going on the same clause after our cut).

_TERMINAL_RE = re.compile(r"[.!?\u2026\u3002\uff01\uff1f][\"'\u201d\u2019)\]]*$")
_PHRASE_RE = re.compile(r"[,;:\u2014\u2013\u3001\uff0c][\"'\u201d\u2019)\]]*$")

# Words that can NEVER end a complete sentence: conjunctions, articles,
# infinitive "to", possessives, bare auxiliaries, subordinators. Deliberately
# not most prepositions ("what we're good at.") — but "with" is in, since an
# unpunctuated clip ending on it was cut mid-clause far more often than not.
# A word carrying terminal punctuation is trusted over this list.
FALSE_ENDING_WORDS = frozenset({
    "and", "but", "or", "nor", "so", "because", "cause", "the", "a", "an", "to", "of",
    "that", "which", "as", "than", "then", "into", "very", "with", "my", "your", "his",
    "her", "their", "our", "its", "is", "was", "are", "were", "been", "be", "am", "if",
    "while", "whose", "until", "unless", "whether", "whereas", "although", "upon",
})
FILLER_WORDS = frozenset({"um", "uh", "uhm", "umm", "uhh", "er", "erm", "ah", "ahh",
                          "hmm", "mm", "mhm", "mmhmm"})
_LOWER_I = frozenset({"i", "i'm", "i've", "i'll", "i'd"})

# A pause this long after a word ends the thought even without punctuation.
SENTENCE_PAUSE = 0.8
PHRASE_PAUSE = 0.35


def normalize_words(words):
    """[{'w','s','e'}] sorted by start, from either that shape or faster-whisper's
    {'word','start','end'}; entries without timestamps are dropped."""
    out = []
    for w in words or []:
        if not isinstance(w, dict):
            continue
        text = w.get("w", w.get("word", ""))
        s = w.get("s", w.get("start"))
        e = w.get("e", w.get("end"))
        try:
            s, e = float(s), float(e)
        except (TypeError, ValueError):
            continue
        out.append({"w": str(text or ""), "s": s, "e": max(s, e)})
    out.sort(key=lambda x: x["s"])
    return out


def norm_token(text):
    return re.sub(r"[^\w']", "", str(text or "").lower())


def is_terminal(text):
    """True when the word carries sentence-final punctuation."""
    return bool(_TERMINAL_RE.search(str(text or "").strip()))


def is_false_ending_word(text):
    """True for a word a clip must never end on (unless it is punctuated)."""
    return not is_terminal(text) and norm_token(text) in FALSE_ENDING_WORDS


def _starts_capitalized(text):
    t = str(text or "").strip().lstrip("\"'\u201c\u2018(\u00bf\u00a1")
    return bool(t) and t[0].isupper() and norm_token(t) not in _LOWER_I


def sentence_end_at(words, i, pause=SENTENCE_PAUSE):
    """True when ``words[i]`` (normalized list) closes a sentence."""
    text = words[i]["w"]
    if is_terminal(text):
        return True
    if norm_token(text) in FALSE_ENDING_WORDS:
        return False
    if i >= len(words) - 1:
        return True  # source ends here: the speaker finished
    gap = words[i + 1]["s"] - words[i]["e"]
    if gap >= pause:
        return True
    # Unpunctuated ASR: a capitalized next word after a breath starts a new sentence.
    return gap >= 0.25 and _starts_capitalized(words[i + 1]["w"])


def sentence_start_at(words, i):
    return i <= 0 or sentence_end_at(words, i - 1)


def phrase_start_at(words, i):
    if sentence_start_at(words, i):
        return True
    prev = words[i - 1]
    return bool(_PHRASE_RE.search(prev["w"].strip())) or words[i]["s"] - prev["e"] >= PHRASE_PAUSE


def ends_complete(words, i):
    """(ok, reason): does a clip ending on ``words[i]`` end on a complete thought?"""
    text = words[i]["w"].strip()
    if is_terminal(text):
        return True, f"ends on {text!r} (sentence-terminal)"
    if norm_token(text) in FALSE_ENDING_WORDS:
        return False, f"ends on {text!r}, which cannot end a sentence (false ending)"
    if sentence_end_at(words, i):
        return True, f"ends on {text!r} before a pause / new sentence"
    nxt = " ".join(w["w"].strip() for w in words[i + 1:i + 6])
    return False, f"ends on {text!r} but the speaker continues: '{nxt}…' (false ending)"


def snap_clip_bounds(start, end, words, video_duration, min_duration=15.0, max_duration=60.0,
                     start_back=6.0, start_fwd=4.0, end_back=6.0, end_fwd=8.0,
                     tolerance=0.5, lenient=False):
    """Move a model-proposed [start, end] onto real sentence boundaries.

    START goes to the nearest sentence start within [start-start_back,
    start+start_fwd] (earlier preferred: moving back keeps context), else a
    phrase start, else any non-filler word start — never mid-word. Leading
    fillers ("um", "uh") are stepped over.

    END goes to a sentence end (terminal punctuation, a long pause, or a new
    capitalized sentence) near the proposed end that keeps the clip inside
    [min_duration, max_duration]; failing that, the clip is EXTENDED to the
    next sentence end that still fits, then shortened to an earlier one. It
    never ends on a FALSE_ENDING_WORDS word.

    Returns a dict (start/end are the cut points with a little of the
    surrounding silence; word_start/word_end, prev_end/next_start are the
    anchors the audio refiner needs), or None when no boundary pair satisfies
    the rules — the caller drops the clip. ``lenient=True`` (used only when
    every clip of a job would otherwise be dropped) settles for the nearest
    non-false-ending word boundaries instead of returning None.
    """
    total = float(video_duration or 0)
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    words = normalize_words(words)
    n = len(words)

    def _plain(s, e, kind):
        s = max(0.0, s)
        e = min(total, e) if total else e
        if not (e > s):
            return None
        return {"start": round(s, 3), "end": round(e, 3), "word_start": None, "word_end": None,
                "prev_end": None, "next_start": None, "start_kind": kind, "end_kind": kind}

    if not n:
        return _plain(start, end, "unsnapped")

    idx = [i for i, w in enumerate(words) if start - start_back <= w["s"] <= start + start_fwd]
    if not idx:
        nearest = min(range(n), key=lambda i: abs(words[i]["s"] - start))
        if abs(words[nearest]["s"] - start) > 15.0:
            return _plain(start, end, "unsnapped") if lenient else None
        idx = [nearest]

    def start_rank(i):
        if norm_token(words[i]["w"]) in FILLER_WORDS:
            tier = 3
        elif sentence_start_at(words, i):
            tier = 0
        elif phrase_start_at(words, i):
            tier = 1
        else:
            tier = 2
        d = words[i]["s"] - start
        return tier, abs(d) * (0.7 if d < 0 else 1.0)

    def skip_fillers(i):
        while i < n - 1 and norm_token(words[i]["w"]) in FILLER_WORDS:
            i += 1
        return i

    def pick_end(si, sentence_only=True):
        t0 = words[si]["s"]
        cands = []
        for j in range(si, n):
            d = words[j]["e"] - t0
            if d > max_duration:
                break
            if d < min_duration - tolerance:
                continue
            if sentence_only and not sentence_end_at(words, j):
                continue
            if not sentence_only and is_false_ending_word(words[j]["w"]):
                continue
            cands.append(j)
        if not cands:
            return None, None
        window = [j for j in cands if end - end_back <= words[j]["e"] <= end + end_fwd]
        if window:
            def cost(j):
                d = words[j]["e"] - end
                return abs(d) * (0.8 if d > 0 else 1.0)  # ties go to keeping the payoff
            return min(window, key=cost), "sentence" if sentence_only else "word"
        later = [j for j in cands if words[j]["e"] > end]
        if later:
            return later[0], ("sentence" if sentence_only else "word") + "-extended"
        return cands[-1], ("sentence" if sentence_only else "word") + "-shortened"

    ordered = sorted(dict.fromkeys(skip_fillers(i) for i in sorted(idx, key=start_rank)),
                     key=start_rank)
    chosen = None
    for si in ordered[:6]:
        ej, end_kind = pick_end(si)
        if ej is not None:
            chosen = (si, ej, end_kind)
            break
    if chosen is None:
        if not lenient:
            return None
        si = ordered[0]
        ej, end_kind = pick_end(si, sentence_only=False)
        if ej is None:
            t0 = words[si]["s"]
            reach = [j for j in range(si, n) if words[j]["e"] - t0 <= max_duration]
            if not reach:
                return _plain(start, end, "unsnapped")
            ej, end_kind = reach[-1], "word-forced"
        chosen = (si, ej, end_kind)

    si, ej, end_kind = chosen
    t0, t1 = words[si]["s"], words[ej]["e"]
    prev_end = words[si - 1]["e"] if si > 0 else None
    next_start = words[ej + 1]["s"] if ej + 1 < n else None
    gap_before = t0 - prev_end if prev_end is not None else t0
    gap_after = (next_start - t1) if next_start is not None else max(0.0, total - t1)
    cut_start = max(0.0, t0 - min(0.12, max(0.0, gap_before) / 2))
    cut_end = t1 + min(0.25, max(0.0, gap_after) / 2)
    if total:
        cut_end = min(total, cut_end)
    if not (0.0 <= cut_start < cut_end) or (total and cut_end > total + 1e-6):
        return None
    if sentence_start_at(words, si):
        start_kind = "sentence"
    elif phrase_start_at(words, si):
        start_kind = "phrase"
    else:
        start_kind = "word"
    return {"start": round(cut_start, 3), "end": round(cut_end, 3),
            "word_start": round(t0, 3), "word_end": round(t1, 3),
            "prev_end": None if prev_end is None else round(prev_end, 3),
            "next_start": None if next_start is None else round(next_start, 3),
            "start_kind": start_kind, "end_kind": end_kind,
            "first_word": words[si]["w"].strip(), "last_word": words[ej]["w"].strip()}


def quiet_search_window(anchor, kind, prev_end=None, next_start=None, video_duration=None):
    """The [lo, hi] range to look for the quietest ~10ms in around a cut.

    In-points look from 100ms before to 50ms after the first word's start
    (land in the leading silence, never drag in the previous word's tail);
    out-points from 60ms before to 200ms after the last word's end (Whisper's
    end labels run early, so this keeps the word's full decay). Neither range
    crosses into a neighbouring word. None when there is no room.
    """
    anchor = float(anchor)
    if kind == "in":
        lo, hi = anchor - 0.10, anchor + 0.05
        if prev_end is not None:
            lo = max(lo, float(prev_end))
    elif kind == "out":
        lo, hi = anchor - 0.06, anchor + 0.20
        if next_start is not None:
            hi = min(hi, float(next_start))
    else:
        raise ValueError(f"kind must be 'in' or 'out', not {kind!r}")
    lo = max(0.0, lo)
    if video_duration:
        hi = min(hi, float(video_duration))
    if hi - lo < 0.02:
        return None
    return round(lo, 4), round(hi, 4)


def timed_transcript(words, start, end, max_words_per_line=14):
    """Compact word-timed transcript of [start, end] for the detail prompt.

    One line per sentence: ``12.3-17.8 The sentence text.`` (start of its
    first word, end of its last). Sentences longer than ``max_words_per_line``
    continue on lines prefixed with ``~``, which are NOT safe start points.
    One decimal is enough — the snapper lands the exact boundary — and keeps
    the overhead to a few tokens per sentence.
    """
    words = normalize_words(words)
    lo, hi = float(start) - 0.05, float(end) + 0.05
    span = [i for i, w in enumerate(words) if w["s"] >= lo and w["e"] <= hi]
    lines, cur = [], []
    for k, i in enumerate(span):
        cur.append(i)
        last = k == len(span) - 1
        if last or sentence_end_at(words, i) or len(cur) >= max_words_per_line:
            first = cur[0]
            text = "".join(words[j]["w"] for j in cur).strip()
            text = " ".join(text.split())
            prefix = "" if sentence_start_at(words, first) else "~"
            lines.append(f"{prefix}{words[first]['s']:.1f}-{words[cur[-1]]['e']:.1f} {text}")
            cur = []
    return "\n".join(lines)


def snap_clip_to_words(start, end, words, video_duration, min_duration=15.0, max_duration=60.0):
    """Compatibility wrapper: (start, end) from ``snap_clip_bounds`` in
    lenient mode, or the clamped input when nothing can be snapped. New code
    should call ``snap_clip_bounds`` directly — it can say "drop this clip"."""
    snapped = snap_clip_bounds(start, end, words, video_duration, min_duration=min_duration,
                               max_duration=max_duration, lenient=True)
    if snapped is None:
        return (round(float(start), 3), round(float(end), 3))
    return snapped["start"], snapped["end"]
