"""Post-render QA gate: pure verdicts, double takes, endings, survivor
selection, and real measurements on ffmpeg-synthesized clips."""
import os
import shutil
import subprocess

import pytest

import clip_qa
from clip_qa import evaluate, ending_problem, find_double_takes, pick_survivors


def _w(text, s, e):
    return {"w": text, "s": s, "e": e}


def _seq(text, start=0.0, step=0.4, dur=0.3):
    return [_w(" " + tok, round(start + i * step, 3), round(start + i * step + dur, 3))
            for i, tok in enumerate(text.split())]


GOOD = {"duration": 30.0, "has_audio": True, "lufs": -14.2,
        "last_frame_luma": {"mean": 90.0, "max": 220}, "ending_problem": None,
        "double_takes": []}


class TestDoubleTakes:
    def test_restart_detected(self):
        hits = find_double_takes(_seq("so we need to we need to fix the roof"))
        assert [p for _, p in hits] == ["we need to"]

    def test_longest_phrase_reported_once(self):
        hits = find_double_takes(_seq("the best way to go the best way to go is north"))
        assert [p for _, p in hits] == ["the best way to go"]

    def test_punctuation_and_case_ignored(self):
        hits = find_double_takes(_seq("And I said, I think that. I think that is fine"))
        assert [p for _, p in hits] == ["i think that"]

    def test_single_word_emphasis_is_not_a_double_take(self):
        assert find_double_takes(_seq("no no no no no no way")) == []

    def test_clean_speech(self):
        assert find_double_takes(_seq("rates dropped a full point this week and buyers noticed")) == []


class TestEndingProblem:
    def test_word_straddling_the_cut_is_chopped(self):
        words = _seq("this is the whole point.")
        # last word "point." spans 1.6-1.9; cut at 1.7
        problem = ending_problem(words, 0.0, 1.7)
        assert problem and "chopped" in problem

    def test_small_overrun_tolerated(self):
        words = _seq("this is the whole point.")
        assert ending_problem(words, 0.0, 1.85) is None

    def test_false_ending_word(self):
        words = _seq("we bought the house and then sold it.")
        # clip covers through "and" (1.6-1.9), cut at 2.0
        problem = ending_problem(words, 0.0, 2.0)
        assert problem and "false ending" in problem

    def test_no_words(self):
        assert ending_problem([], 0, 10) is None


class TestEvaluate:
    def test_good_clip_passes(self):
        assert evaluate(dict(GOOD), 15, 60)["passed"]

    @pytest.mark.parametrize("patch,needle", [
        ({"duration": 10.0}, "duration"),
        ({"duration": 70.0}, "duration"),
        ({"duration": None}, "duration"),
        ({"has_audio": False}, "no audio stream"),
        ({"lufs": -70.0}, "silent"),
        ({"lufs": -2.0}, "too loud"),
        ({"last_frame_luma": {"mean": 2.0, "max": 10}}, "black"),
        ({"ending_problem": "ends on 'and'"}, "ending"),
        ({"double_takes": [(3.2, "we need to")]}, "double take"),
    ])
    def test_each_failure(self, patch, needle):
        verdict = evaluate(dict(GOOD, **patch), 15, 60)
        assert not verdict["passed"]
        assert any(needle in f for f in verdict["failures"])

    def test_duration_slack(self):
        assert evaluate(dict(GOOD, duration=14.5), 15, 60)["passed"]
        assert evaluate(dict(GOOD, duration=60.8), 15, 60)["passed"]

    def test_dark_but_not_black_frame_passes(self):
        assert evaluate(dict(GOOD, last_frame_luma={"mean": 6.0, "max": 120}), 15, 60)["passed"]

    def test_no_audio_expected(self):
        assert evaluate(dict(GOOD, has_audio=False, lufs=None), 15, 60, expect_audio=False)["passed"]

    def test_missing_measurements_never_fail(self):
        assert evaluate({"duration": 30.0}, 15, 60)["passed"]


class TestPickSurvivors:
    def test_passing_clips_kept(self):
        ok, bad = {"passed": True, "failures": []}, {"passed": False, "failures": ["x"]}
        assert pick_survivors([(0, ok, 50), (1, bad, 99), (2, ok, 10)]) == [0, 2]

    def test_never_drops_everything(self):
        one = {"passed": False, "failures": ["a"]}
        two = {"passed": False, "failures": ["a", "b"]}
        assert pick_survivors([(0, two, 99), (1, one, 10), (2, one, 70)]) == [2]

    def test_empty(self):
        assert pick_survivors([]) == []


# --- real measurements (need ffmpeg) -----------------------------------------

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg not installed")


def _make(path, seconds, video="color=c=gray:s=64x112:r=10", audio="sine=frequency=440",
          tail_black=0.0):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"{video}:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"{audio}:d={seconds}"]
    vf = []
    if tail_black:
        vf = ["-vf", f"drawbox=c=black:t=fill:enable='gte(t,{seconds - tail_black})'"]
    cmd += vf + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest"]
    if audio:
        cmd += ["-c:a", "aac"]
    cmd.append(path)
    subprocess.run(cmd, check=True)
    return path


@needs_ffmpeg
class TestMeasurements:
    def test_good_clip_passes(self, tmp_path):
        p = _make(str(tmp_path / "good.mp4"), 16)
        verdict = clip_qa.qa_clip(p, 0, 16, [], 15, 60)
        assert verdict["passed"], verdict["failures"]
        m = verdict["metrics"]
        assert 15.5 < m["duration"] < 16.5
        assert m["lufs"] is not None and m["lufs"] > -40
        assert m["last_frame_luma"]["mean"] > 50

    def test_silent_clip_fails(self, tmp_path):
        p = _make(str(tmp_path / "silent.mp4"), 16, audio="anullsrc=r=44100:cl=mono")
        verdict = clip_qa.qa_clip(p, 0, 16, [], 15, 60)
        assert not verdict["passed"]
        assert any("silent" in f for f in verdict["failures"])

    def test_missing_audio_stream_fails(self, tmp_path):
        p = _make(str(tmp_path / "noaudio.mp4"), 16, audio=None)
        verdict = clip_qa.qa_clip(p, 0, 16, [], 15, 60)
        assert any("no audio stream" in f for f in verdict["failures"])
        assert clip_qa.qa_clip(p, 0, 16, [], 15, 60, expect_audio=False)["passed"]

    def test_black_last_frame_fails(self, tmp_path):
        p = _make(str(tmp_path / "black.mp4"), 16, tail_black=1.0)
        verdict = clip_qa.qa_clip(p, 0, 16, [], 15, 60)
        assert any("black" in f for f in verdict["failures"])

    def test_too_short_fails(self, tmp_path):
        p = _make(str(tmp_path / "short.mp4"), 5)
        verdict = clip_qa.qa_clip(p, 0, 5, [], 15, 60)
        assert any("duration" in f for f in verdict["failures"])

    def test_transcript_checks_run(self, tmp_path):
        p = _make(str(tmp_path / "talk.mp4"), 16)
        words = _seq("we need to we need to fix the roof and", start=1.0)
        verdict = clip_qa.qa_clip(p, 0, 16, words, 15, 60)
        fails = " | ".join(verdict["failures"])
        assert "double take" in fails and "false ending" in fails

    def test_unreadable_file(self, tmp_path):
        p = tmp_path / "junk.mp4"
        p.write_bytes(b"not a video")
        verdict = clip_qa.qa_clip(str(p), 0, 16, [], 15, 60)
        assert not verdict["passed"]


@needs_ffmpeg
class TestAudioSnap:
    def test_quietest_finds_the_gap(self, tmp_path):
        np = pytest.importorskip("numpy")
        import audio_snap
        sr = 16000
        y = np.sin(np.arange(sr) * 2 * np.pi * 440 / sr).astype(np.float32)
        y[int(0.40 * sr):int(0.43 * sr)] = 0.0   # 30ms gap at 0.40-0.43
        off = audio_snap.quietest_offset(y, sr)
        assert 0.40 <= off <= 0.43

    def test_quietest_on_a_real_file(self, tmp_path):
        pytest.importorskip("numpy")
        import audio_snap
        p = str(tmp_path / "gap.wav")
        # 1s tone, 50ms of silence at 0.5s
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=1",
                        "-af", "volume=enable='between(t,0.5,0.55)':volume=0", p], check=True)
        t = audio_snap.quietest(p, 0.4, 0.65)
        assert t is not None and 0.49 <= t <= 0.56

    def test_refine_moves_cuts_into_silence_and_keeps_order(self, tmp_path):
        pytest.importorskip("numpy")
        import audio_snap
        p = str(tmp_path / "gap2.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=3",
                        "-af", "volume=enable='between(t,0.95,1.0)+between(t,2.05,2.15)':volume=0",
                        p], check=True)
        clip = {"start": 0.9, "end": 2.3,
                "snap": {"word_start": 1.0, "word_end": 2.05, "prev_end": None,
                         "next_start": None, "cut_start": 0.9, "cut_end": 2.3}}
        assert audio_snap.refine_clip_cuts(p, clip, 3.0)
        assert 0.94 <= clip["start"] <= 1.01
        assert 2.04 <= clip["end"] <= 2.16
        assert clip["start"] < clip["end"]

    def test_refine_without_anchors_is_a_noop(self):
        import audio_snap
        clip = {"start": 1.0, "end": 5.0}
        assert audio_snap.refine_clip_cuts("/nonexistent.mp4", clip, 10.0) is False
        assert (clip["start"], clip["end"]) == (1.0, 5.0)


@needs_ffmpeg
def test_cut_clip_fades_do_not_break_the_cut(tmp_path):
    import ffmpeg_utils
    src = _make(str(tmp_path / "src.mp4"), 6)
    out = str(tmp_path / "cut.mp4")
    ffmpeg_utils.cut_clip(src, out, 1.0, 4.0, 1)
    meta = clip_qa.probe(out)
    assert meta["has_audio"] and 2.8 < meta["duration"] < 3.3
