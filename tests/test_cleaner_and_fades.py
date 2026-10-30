"""/api/v2/edit cleaner (edge trims, join fades, no-audio) and cut fades."""
import shutil
import subprocess

import pytest

import cleaner
import ffmpeg_utils


def _w(word, s, e):
    return {"word": word, "start": s, "end": e}


class TestFindCuts:
    def test_trims_leading_and_trailing_dead_air(self):
        words = [_w(" hi", 2.0, 2.4), _w(" there", 2.5, 2.9)]
        keep = cleaner.invert_ranges(cleaner.find_cuts(words, 10.0), 10.0)
        assert keep[0][0] == pytest.approx(2.0 - cleaner.EDGE_PAD)
        assert keep[-1][1] == pytest.approx(2.9 + cleaner.EDGE_PAD)

    def test_fillers_and_long_gaps_cut(self):
        words = [_w(" so", 0.1, 0.3), _w(" um", 0.4, 0.7), _w(" yes", 0.8, 1.0),
                 _w(" ok", 3.0, 3.2)]
        keep = cleaner.invert_ranges(cleaner.find_cuts(words, 3.4), 3.4)
        kept = sum(e - s for s, e in keep)
        assert kept < 3.4 - 0.3 - 1.5  # filler + most of the 2s gap gone

    def test_slivers_dropped(self):
        assert cleaner.invert_ranges([(0.0, 1.0), (1.02, 2.0)], 2.0) == []

    def test_no_cuts_keeps_everything(self):
        assert cleaner.invert_ranges([], 5.0) == [(0.0, 5.0)]


class TestBuildFilter:
    def test_audio_pieces_get_fades(self):
        f = cleaner.build_filter([(0.0, 1.0), (2.0, 3.5)])
        assert f.count("afade=t=in") == 2 and f.count("afade=t=out") == 2
        assert "concat=n=2:v=1:a=1" in f

    def test_tiny_piece_skips_fades(self):
        f = cleaner.build_filter([(0.0, 0.03)])
        assert "afade" not in f

    def test_no_audio_graph(self):
        f = cleaner.build_filter([(0.0, 1.0), (2.0, 3.0)], has_audio=False)
        assert "[0:a]" not in f and "concat=n=2:v=1:a=0[outv]" in f


class TestCutFades:
    def test_fade_filters(self, monkeypatch):
        monkeypatch.delenv("CUT_AUDIO_FADE_MS", raising=False)
        assert ffmpeg_utils.audio_fade_filters(10.0) == [
            "afade=t=in:st=0:d=0.008", "afade=t=out:st=9.992:d=0.008"]

    def test_fades_disabled(self, monkeypatch):
        monkeypatch.setenv("CUT_AUDIO_FADE_MS", "0")
        assert ffmpeg_utils.audio_fade_filters(10.0) == []

    def test_encode_args_order_loudnorm_then_fades(self, monkeypatch):
        monkeypatch.delenv("CUT_AUDIO_FADE_MS", raising=False)
        monkeypatch.delenv("AUDIO_NORMALIZE", raising=False)
        args = ffmpeg_utils.audio_encode_args(10.0)
        chain = args[args.index("-af") + 1]
        assert chain.startswith(ffmpeg_utils.LOUDNORM_FILTER) and "afade=t=out" in chain

    def test_encode_args_without_duration_unchanged(self, monkeypatch):
        monkeypatch.delenv("AUDIO_NORMALIZE", raising=False)
        assert ffmpeg_utils.audio_encode_args() == ["-af", ffmpeg_utils.LOUDNORM_FILTER, "-c:a", "aac"]


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@needs_ffmpeg
def test_clean_clip_without_audio_is_copied(tmp_path):
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=64x64:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    out = tmp_path / "v_edited.mp4"
    cleaner.clean_clip(str(src), str(out))
    assert out.read_bytes() == src.read_bytes()


@needs_ffmpeg
def test_apply_cuts_with_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    src = tmp_path / "a.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=4",
                    "-f", "lavfi", "-i", "sine=f=440:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    out = tmp_path / "a_edited.mp4"
    cleaner.apply_cuts(str(src), str(out), [(0.0, 1.0), (2.0, 3.0)])
    dur, has_audio = cleaner.probe_media(str(out))
    assert has_audio and 1.8 < dur < 2.3
    assert not (tmp_path / "a_edited.mp4.part.mp4").exists()
