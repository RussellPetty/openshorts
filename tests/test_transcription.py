"""Shared faster-whisper setup (transcription.py)."""
import sys
import types

import pytest

import subtitles
import transcription


def test_defaults(monkeypatch):
    for k in ("WHISPER_MODEL", "WHISPER_DEVICE", "WHISPER_COMPUTE", "WHISPER_CPU_THREADS",
              "WHISPER_BEAM_SIZE"):
        monkeypatch.delenv(k, raising=False)
    assert transcription.get_whisper_config() == {
        "model_size": "small", "device": "cpu", "compute_type": "int8", "cpu_threads": 0}
    p = transcription.transcribe_params()
    assert p["vad_filter"] is True and p["condition_on_previous_text"] is False
    assert p["beam_size"] == 5 and p["word_timestamps"] is True
    assert transcription.transcribe_params(word_timestamps=False)["word_timestamps"] is False


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "2")
    monkeypatch.setenv("WHISPER_CPU_THREADS", "junk")
    assert transcription.get_whisper_config()["model_size"] == "base"
    assert transcription.get_whisper_config()["cpu_threads"] == 0
    assert transcription.transcribe_params()["beam_size"] == 2


def test_subtitles_reexports_the_shared_helpers():
    assert subtitles.merge_continuation_words is transcription.merge_continuation_words
    assert subtitles.get_whisper_config is transcription.get_whisper_config


def test_merge_is_idempotent():
    words = [{"word": " YouTube", "start": 0, "end": 1}, {"word": "-Kanal.", "start": 1, "end": 2}]
    once = transcription.merge_continuation_words(words)
    assert transcription.merge_continuation_words(once) == once


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end, self.probability = word, start, end, 0.9


class _Seg:
    def __init__(self, text, start, end, words):
        self.text, self.start, self.end, self.words = text, start, end, words


def test_transcribe_loads_model_once_and_shapes_output(monkeypatch):
    loads, calls = [], []

    class FakeModel:
        def __init__(self, size, device, compute_type, cpu_threads):
            loads.append((size, device, compute_type, cpu_threads))

        def transcribe(self, path, **params):
            calls.append(params)
            segs = [_Seg(" Hello YouTube-Kanal.", 0.0, 2.0,
                         [_Word(" Hello", 0.0, 0.5), _Word(" YouTube", 0.6, 1.2),
                          _Word("-Kanal.", 1.2, 2.0)])]
            info = types.SimpleNamespace(language="de", language_probability=0.98, duration=30.0)
            return iter(segs), info

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(transcription, "_models", {})
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    r1 = transcription.transcribe("/a.mp4")
    transcription.transcribe("/b.mp4", word_timestamps=False)
    assert loads == [("small", "cpu", "int8", 0)]
    assert calls[0]["vad_filter"] and not calls[1]["word_timestamps"]
    assert r1["language"] == "de" and r1["duration"] == 30.0
    assert [w["word"] for w in r1["segments"][0]["words"]] == [" Hello", " YouTube-Kanal."]
    assert r1["text"] == "Hello YouTube-Kanal."


def test_cleaner_uses_shared_transcription(monkeypatch):
    import cleaner
    monkeypatch.setattr(transcription, "transcribe", lambda p, word_timestamps=True, **k: {
        "segments": [{"words": [{"word": " a", "start": 0.1, "end": 0.2},
                                {"word": " b", "start": None, "end": None}]}]})
    assert cleaner.transcribe_clip("/x.mp4") == [{"word": " a", "start": 0.1, "end": 0.2}]


def test_no_merge_for_languages_without_spaces():
    words = [{"word": "你", "start": 0, "end": 0.2}, {"word": "好", "start": 0.2, "end": 0.4}]
    assert len(transcription.merge_for_language(words, "zh")) == 2
    assert len(transcription.merge_for_language(words, "en")) == 1
