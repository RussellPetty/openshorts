"""Every prompt template must format with exactly the kwargs its caller passes.

The silent-video prompt needed {min_clips},{max_clips},{min_secs},{max_secs}
but main.get_visual_clips passed only video_duration + language: KeyError on
every silent/low-speech job, swallowed into the "first 60 seconds" fallback.
"""
import string

import pytest

import gemini_worker


def _fields(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def test_visual_prompt_builder_formats():
    prompt = gemini_worker.build_visual_prompt(600, "en")
    assert "{min_clips}" not in prompt and "{max_secs" not in prompt
    assert "600" in prompt


def test_visual_prompt_contains_the_clip_band(monkeypatch):
    monkeypatch.setenv("CLIP_MIN_SECONDS", "20")
    monkeypatch.setenv("CLIP_MAX_SECONDS", "45")
    prompt = gemini_worker.build_visual_prompt(900, "es", min_clips=3, max_clips=5)
    assert "3–5" in prompt
    assert "20 to 45 seconds" in prompt
    assert "in es" in prompt


def test_visual_template_needs_only_what_the_builder_passes():
    assert _fields(gemini_worker.VISUAL_PROMPT_TEMPLATE) == {
        "video_duration", "language", "min_clips", "max_clips", "min_secs", "max_secs"}


def test_old_two_kwarg_call_would_fail():
    # Documents the bug the builder fixes.
    with pytest.raises(KeyError):
        gemini_worker.VISUAL_PROMPT_TEMPLATE.format(video_duration=60, language="en")


def test_score_template_fields():
    assert _fields(gemini_worker.SCORE_PROMPT_TEMPLATE) == {
        "language", "video_duration", "windows_json"}
    gemini_worker.SCORE_PROMPT_TEMPLATE.format(language="en", video_duration=60, windows_json="[]")


def test_detail_template_fields_and_rules():
    assert _fields(gemini_worker.DETAIL_PROMPT_TEMPLATE) == {
        "language", "video_duration", "min_clips", "max_clips", "min_secs", "max_secs",
        "windows_json"}
    p = gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
        language="en", video_duration=600, min_clips=4, max_clips=8, min_secs=15.0,
        max_secs=60.0, windows_json="[]")
    assert "15 to 60 seconds" in p
    for rule in ("NEVER OPEN ON A LITERAL QUESTION", "NEVER END RIGHT BEFORE THE EXPLANATION",
                 "interior cuts", "TIMED TRANSCRIPT", "punchline"):
        assert rule in p
