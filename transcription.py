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
"""
import os
import threading

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


def transcribe(media_path, word_timestamps=True, log=None, echo_segments=False):
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
    }
