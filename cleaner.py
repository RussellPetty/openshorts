import os
import shutil
import subprocess
import json

from ffmpeg_utils import video_encode_args, QUALITY, METADATA_SCRUB

FILLER_WORDS = {'umm', 'um', 'uh', 'uhh', 'ah', 'ahh', 'uhm'}

# Fade applied on both sides of every join (and at the clip's ends), seconds.
# A hard audio splice between two non-zero samples clicks; ~10ms of fade is
# inaudible as a fade but removes the click. Fades rather than acrossfade on
# purpose: acrossfade overlaps the pieces and would shorten the audio against
# the video, drifting lip sync by 10ms per join.
JOIN_FADE = 0.010
# Speech kept before the first word / after the last when trimming the clip's
# leading and trailing dead air.
EDGE_PAD = 0.15


def transcribe_clip(video_path):
    """Word-level transcript of a clip via the shared transcription module
    (transcription.py: Grok STT tiers, faster-whisper fallback)."""
    import transcription
    result = transcription.transcribe(video_path, word_timestamps=True)
    words = []
    for segment in result.get('segments', []):
        for w in segment.get('words') or []:
            if w.get('start') is None or w.get('end') is None:
                continue
            words.append({'word': w['word'], 'start': float(w['start']), 'end': float(w['end'])})
    return words


def find_cuts(words, total_duration, max_silence_gap=0.5, silence_pad=0.15):
    """
    Return list of (start, end) ranges to cut.
    - Filler words (umm, um, uh, etc.) are cut entirely.
    - Silence gaps longer than max_silence_gap are trimmed, leaving silence_pad on each side.
    - Dead air before the first word and after the last is trimmed, leaving EDGE_PAD.
    """
    cuts = []

    for w in words:
        clean = w['word'].strip().lower().strip('.,!?-')
        if clean in FILLER_WORDS:
            cuts.append((w['start'], w['end']))

    sorted_words = sorted(words, key=lambda x: x['start'])
    for i in range(len(sorted_words) - 1):
        gap_start = sorted_words[i]['end']
        gap_end = sorted_words[i + 1]['start']
        if gap_end - gap_start > max_silence_gap:
            cut_start = gap_start + silence_pad
            cut_end = gap_end - silence_pad
            if cut_end > cut_start:
                cuts.append((cut_start, cut_end))

    if sorted_words and total_duration:
        lead_end = sorted_words[0]['start'] - EDGE_PAD
        if lead_end > 0.05:
            cuts.append((0.0, lead_end))
        tail_start = max(w['end'] for w in sorted_words) + EDGE_PAD
        if total_duration - tail_start > 0.05:
            cuts.append((tail_start, total_duration))

    return cuts


def invert_ranges(cuts, total_duration, min_keep=0.05):
    """Convert cut ranges to keep ranges over [0, total_duration]. Slivers
    shorter than ``min_keep`` are dropped (they only add a join)."""
    if not cuts:
        return [(0.0, total_duration)]

    # Sort and merge overlapping cuts
    merged = []
    for s, e in sorted(cuts, key=lambda x: x[0]):
        s, e = max(0.0, s), min(total_duration, e)
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    keep = []
    cursor = 0.0
    for s, e in merged:
        if s - cursor >= min_keep:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if total_duration - cursor >= min_keep:
        keep.append((cursor, total_duration))

    return keep


def _fade_chain(duration, fade=JOIN_FADE):
    """afade in+out for one kept piece (skipped when the piece is too short)."""
    if duration <= 4 * fade:
        return ""
    return (f",afade=t=in:st=0:d={fade:.3f}"
            f",afade=t=out:st={duration - fade:.4f}:d={fade:.3f}")


def build_filter(keep_ranges, has_audio=True, fade=JOIN_FADE):
    """The trim+concat filtergraph for ``keep_ranges``."""
    filter_parts = []
    for i, (s, e) in enumerate(keep_ranges):
        filter_parts.append(f"[0:v]trim={s:.4f}:{e:.4f},setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            filter_parts.append(f"[0:a]atrim={s:.4f}:{e:.4f},asetpts=PTS-STARTPTS"
                                f"{_fade_chain(e - s, fade)}[a{i}]")

    n = len(keep_ranges)
    if has_audio:
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")
    else:
        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")
    return ";".join(filter_parts)


def apply_cuts(input_path, output_path, keep_ranges, has_audio=True):
    """Run FFmpeg trim+concat to keep only the specified time ranges."""
    if not keep_ranges:
        raise ValueError("No keep ranges — all content would be removed")

    maps = ['-map', '[outv]'] + (['-map', '[outa]', '-c:a', 'aac'] if has_audio else [])
    tmp = f"{output_path}.part.mp4"
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter_complex', build_filter(keep_ranges, has_audio),
        *maps,
        *video_encode_args(QUALITY), '-pix_fmt', 'yuv420p',
        *METADATA_SCRUB, '-movflags', '+faststart',
        tmp
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.replace(tmp, output_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def probe_media(path):
    """(duration_seconds or None, has_audio) via ffprobe."""
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=duration',
            '-of', 'json', path], timeout=60).decode()
        data = json.loads(out or '{}')
    except Exception:
        return None, False
    has_audio = any(s.get('codec_type') == 'audio' for s in data.get('streams') or [])
    try:
        duration = float(data.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        duration = None
    return duration, has_audio


def clean_clip(input_path, output_path, max_silence_gap=0.5, silence_pad=0.15):
    """Transcribe clip, find fillers + dead space, cut them out.

    A clip with no audio, or no recognisable speech, has nothing to clean: it
    is copied through unchanged so the endpoint still returns a playable URL.
    """
    total_duration, has_audio = probe_media(input_path)
    if not has_audio:
        print(f"✂️  {input_path} has no audio track — nothing to clean, copying as-is.")
        shutil.copyfile(input_path, output_path)
        return output_path

    print(f"✂️  Transcribing clip for cleaning: {input_path}")
    words = transcribe_clip(input_path)
    print(f"   Found {len(words)} words")
    if not words:
        shutil.copyfile(input_path, output_path)
        return output_path
    if not total_duration:
        # Unknown container duration: never invent one (the old code assumed
        # an hour). The last word's end plus the edge pad is all we can trust.
        total_duration = max(w['end'] for w in words) + EDGE_PAD

    cuts = find_cuts(words, total_duration, max_silence_gap, silence_pad)
    keep_ranges = invert_ranges(cuts, total_duration)
    print(f"   Cutting {len(cuts)} segments → keeping {len(keep_ranges)} ranges")
    if not keep_ranges:
        shutil.copyfile(input_path, output_path)
        return output_path

    apply_cuts(input_path, output_path, keep_ranges, has_audio=True)
    return output_path
