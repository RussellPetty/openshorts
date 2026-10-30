"""app.py helpers: progress parsing, ClipResult mapping, retention sweep,
failed-job leftovers, and the transcribe/cleaner wiring to transcription.py."""
import os
import time

import pytest

app = pytest.importorskip("app")  # needs fastapi/redis client libs


class TestParseProgress:
    def test_metadata_line_is_not_finalizing(self):
        assert app.parse_progress("   Saved metadata to output/x/y_metadata.json") is None
        assert app.parse_progress("   Wrote metadata: output/x/y_metadata.json") is None

    def test_clip_lines_are_finalizing(self):
        assert app.parse_progress("   ✅ Clip saved to output/x/c.mp4")[0] == 90
        assert app.parse_progress("   ✅ Clip 3 ready: output/x/c.mp4")[0] == 90

    def test_other_stages(self):
        assert app.parse_progress("🎙️  Transcribing with faster-whisper")[0] == 30
        assert app.parse_progress("🎬 Processing Clip 1: 3s - 40s")[0] == 70


class TestClipResult:
    def test_new_fields_and_youtube_description(self):
        clip = {"start": 1.5, "end": 40.0, "video_title_for_youtube_short": "Title",
                "video_description_for_tiktok": "tt #a", "video_description_for_instagram": "ig #b",
                "viral_hook_text": "97% miss this", "predicted_score": 87.4,
                "why": "Opens on a number.", "qa": {"passed": True, "failures": []}}
        r = app.clip_result_from_metadata("job", "t_clip_1.mp4", clip)
        assert r.video_url.endswith("/job/t_clip_1.mp4")
        assert r.title == "Title" and r.description_youtube == "ig #b"
        assert r.viral_hook_text == "97% miss this" and r.predicted_score == 87
        assert r.why == "Opens on a number." and r.qa["passed"] is True
        assert (r.start, r.end) == (1.5, 40.0)

    def test_explicit_youtube_description_wins(self):
        r = app.clip_result_from_metadata("j", "f.mp4", {
            "video_title_for_youtube_short": "T", "video_description_for_youtube": "yt",
            "video_description_for_instagram": "ig"})
        assert r.description_youtube == "yt"

    def test_fallback_clip_metadata(self):
        r = app.clip_result_from_metadata("j", "f.mp4", {
            "start": 0, "end": 60, "video_title_for_youtube_short": "My video",
            "video_description_for_tiktok": "", "video_description_for_instagram": ""})
        assert r.description_youtube == "My video"
        assert r.viral_hook_text is None and r.predicted_score is None and r.qa is None

    def test_bad_score_is_none(self):
        r = app.clip_result_from_metadata("j", "f.mp4", {"predicted_score": "n/a"})
        assert r.predicted_score is None


class TestRetentionSweep:
    def _age(self, path, seconds):
        old = time.time() - seconds
        for root, dirs, files in os.walk(path):
            for n in files + dirs:
                os.utime(os.path.join(root, n), (old, old))
        os.utime(path, (old, old))

    def test_old_dirs_removed_new_and_active_kept(self, tmp_path):
        out = tmp_path / "output"
        for name in ("old", "new", "active"):
            (out / name).mkdir(parents=True)
            (out / name / "clip.mp4").write_bytes(b"x")
        self._age(out / "old", 50 * 3600)
        self._age(out / "active", 50 * 3600)
        removed = app.sweep_expired_outputs(retention=48 * 3600, base_dirs=(str(out),),
                                            active={"active"})
        assert removed == 1
        assert not (out / "old").exists() and (out / "new").exists() and (out / "active").exists()

    def test_recent_file_inside_old_dir_keeps_it(self, tmp_path):
        out = tmp_path / "output"
        (out / "job").mkdir(parents=True)
        (out / "job" / "a.mp4").write_bytes(b"x")
        self._age(out / "job", 50 * 3600)
        (out / "job" / "b_edited.mp4").write_bytes(b"x")  # edited yesterday-ish: now
        assert app.sweep_expired_outputs(retention=48 * 3600, base_dirs=(str(out),), active=set()) == 0

    def test_missing_base_dir(self, tmp_path):
        assert app.sweep_expired_outputs(base_dirs=(str(tmp_path / "nope"),), active=set()) == 0


def test_failed_job_leftovers_keep_only_published_clips(tmp_path):
    for name in ("Talk_clip_1.mp4", "Talk_clip_2_edited.mp4", "Talk.mp4", "Talk_metadata.json",
                 ".qa_hold_3_ab12cd.mp4", ".rendering_1234.mp4", "temp_x_Talk_clip_4.mp4"):
        (tmp_path / name).write_bytes(b"x")
    app.remove_job_leftovers(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["Talk_clip_1.mp4", "Talk_clip_2_edited.mp4"]


def test_api_transcribe_uses_shared_transcription(monkeypatch):
    import transcription
    seen = {}

    def fake(path, word_timestamps=True, **kw):
        seen["args"] = (path, word_timestamps)
        return {"text": "hi there", "language": "en", "segments": [], "duration": 12.5}

    monkeypatch.setattr(transcription, "transcribe", fake)
    assert app._transcribe("/x.mp4") == {"text": "hi there", "language": "en", "duration": 12.5}
    assert seen["args"] == ("/x.mp4", False)
