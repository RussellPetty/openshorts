import os
import subprocess
import json

FILLER_WORDS = {'umm', 'um', 'uh', 'uhh', 'ah', 'ahh', 'uhm'}


def transcribe_clip(video_path):
    """Transcribe a video file with word-level timestamps using Faster-Whisper."""
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video_path, word_timestamps=True)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({'word': w.word, 'start': w.start, 'end': w.end})
    return words


def find_cuts(words, total_duration, max_silence_gap=0.5, silence_pad=0.15):
    """
    Return list of (start, end) ranges to cut.
    - Filler words (umm, um, uh, etc.) are cut entirely.
    - Silence gaps longer than max_silence_gap are trimmed, leaving silence_pad on each side.
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

    return cuts


def invert_ranges(cuts, total_duration):
    """Convert cut ranges to keep ranges over [0, total_duration]."""
    if not cuts:
        return [(0.0, total_duration)]

    # Sort and merge overlapping cuts
    merged = []
    for s, e in sorted(cuts, key=lambda x: x[0]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    keep = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            keep.append((cursor, s))
        cursor = e
    if cursor < total_duration:
        keep.append((cursor, total_duration))

    return keep


def apply_cuts(input_path, output_path, keep_ranges):
    """Run FFmpeg trim+concat to keep only the specified time ranges."""
    if not keep_ranges:
        raise ValueError("No keep ranges — all content would be removed")

    filter_parts = []
    for i, (s, e) in enumerate(keep_ranges):
        filter_parts.append(f"[0:v]trim={s:.4f}:{e:.4f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim={s:.4f}:{e:.4f},asetpts=PTS-STARTPTS[a{i}]")

    n = len(keep_ranges)
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter_complex', ";".join(filter_parts),
        '-map', '[outv]', '-map', '[outa]',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        output_path
    ]
    subprocess.run(cmd, check=True)


def clean_clip(input_path, output_path, max_silence_gap=0.5, silence_pad=0.15):
    """Transcribe clip, find fillers + dead space, cut them out."""
    # Probe duration
    try:
        probe = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=duration', '-of', 'csv=s=x:p=0', input_path
        ]).decode().strip()
        total_duration = float(probe)
    except Exception:
        total_duration = 3600.0

    print(f"✂️  Transcribing clip for cleaning: {input_path}")
    words = transcribe_clip(input_path)
    print(f"   Found {len(words)} words")

    cuts = find_cuts(words, total_duration, max_silence_gap, silence_pad)
    keep_ranges = invert_ranges(cuts, total_duration)
    print(f"   Cutting {len(cuts)} segments → keeping {len(keep_ranges)} ranges")

    apply_cuts(input_path, output_path, keep_ranges)
    return output_path
