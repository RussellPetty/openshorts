"""Clip tightening (tighten.py) + the trim/concat cut in ffmpeg_utils."""
import json
import os
import shutil
import subprocess

import pytest

import tighten
import ffmpeg_utils


def W(text, s, e):
    return {"word": f" {text}", "start": s, "end": e}


# --- filler detection ---------------------------------------------------------

def test_single_token_fillers_always_removed():
    ws = [W("So", 0.0, 0.2), W("um,", 0.25, 0.5), W("we", 0.55, 0.7), W("Hmm.", 0.8, 1.0),
          W("erm", 1.1, 1.3), W("go", 1.35, 1.5)]
    rem = tighten.find_removable_words(ws)
    assert rem == {1: "filler", 3: "filler", 4: "filler"}


def test_isolated_fillers_need_pauses_on_both_sides():
    ws = [W("I", 0.0, 0.2), W("like", 0.25, 0.4), W("pizza.", 0.45, 0.8),  # mid-sentence verb
          W("Like,", 1.2, 1.4), W("it", 1.7, 1.8), W("works.", 1.85, 2.2),  # standalone
          W("So", 2.25, 2.4), W("what?", 2.45, 2.7)]                       # no pause after
    rem = tighten.find_removable_words(ws)
    assert rem == {3: "filler"}


def test_phrase_fillers_only_when_flanked_by_pauses():
    flanked = [W("Yes.", 0.0, 0.3), W("You", 0.6, 0.7), W("know,", 0.72, 0.9),
               W("it", 1.2, 1.3), W("works.", 1.35, 1.6)]
    assert tighten.find_removable_words(flanked) == {1: "filler", 2: "filler"}
    inline = [W("do", 0.0, 0.2), W("you", 0.25, 0.35), W("know", 0.37, 0.5), W("him?", 0.55, 0.8)]
    assert tighten.find_removable_words(inline) == {}
    i_mean = [W("And", 0.0, 0.2), W("I", 0.25, 0.3), W("mean,", 0.32, 0.5), W("we", 0.8, 0.9),
              W("know.", 0.95, 1.2)]
    assert tighten.find_removable_words(i_mean) == {}  # no pause before "I mean"


def test_double_take_removes_the_first_take():
    toks = "we need to we need to fix it".split()
    ws = [W(t, i * 0.3, i * 0.3 + 0.25) for i, t in enumerate(toks)]
    rem = tighten.find_removable_words(ws)
    assert rem == {0: "double_take", 1: "double_take", 2: "double_take"}
    # one repeated token is emphasis, not a restart
    ws = [W("no", i * 0.3, i * 0.3 + 0.25) for i in range(6)]
    assert tighten.find_removable_words(ws) == {}


def test_grok_fixture_finds_fillers():
    fx = os.path.join(os.path.dirname(__file__), "fixtures", "grok_stt_sample.json")
    try:
        body = json.load(open(fx))
    except OSError:
        pytest.skip("fixture not present")
    ws = [W(w["text"], w["start"], w["end"]) for w in body["words"]]
    rem = tighten.find_removable_words(ws)
    removed_text = {tighten._tok(ws[i]["word"]) for i in rem}
    assert removed_text <= tighten.FILLER_TOKENS | tighten.ISOLATED_FILLERS | {"you", "know", "i", "mean"} \
        | {tighten._tok(w["word"]) for i, w in enumerate(ws) if rem.get(i) == "double_take"}


# --- planning -----------------------------------------------------------------

def _speech(gaps, word=0.2, step=0.25, n=8, t0=0.5):
    """n-word bursts separated by the given gap lengths."""
    ws, t = [], t0
    for g in list(gaps) + [None]:
        for _ in range(n):
            ws.append(W("la", round(t, 3), round(t + word, 3)))
            t += step
        t = t - step + word + (g or 0)
    return ws


def test_dead_air_trimmed_to_keep_gap():
    ws = _speech([1.5, 0.3, 2.0])
    end = ws[-1]["end"] + 0.1
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    assert p["cuts"] == 2  # the 0.3s pause stays
    for a, b, kind in p["removed"]:
        assert kind == "dead_air"
    assert p["removed_seconds"] == pytest.approx(1.5 + 2.0 - 2 * tighten.KEEP_GAP, abs=0.07)
    # keep intervals are on the frame grid
    for s, e in p["keep"]:
        assert abs(s * 30 - round(s * 30)) < 1e-5 and abs(e * 30 - round(e * 30)) < 1e-5
    assert p["duration"] == pytest.approx(end - p["removed_seconds"], abs=1e-3)


def test_cap_on_removed_fraction_and_partial_dead_air():
    ws = _speech([3.0, 3.0])
    end = ws[-1]["end"] + 0.1
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.25, fps=30,
                     frame_diff=lambda a, b: 0.0)
    assert p["removed_seconds"] <= 0.25 * end + 1e-6
    assert p["removed_seconds"] > 0.2 * end  # used the budget, trimming partially


def test_never_below_min_clip_seconds():
    ws = _speech([2.0, 2.0])
    end = ws[-1]["end"] + 0.1
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=end - 0.05, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    assert p["cuts"] == 0 and p["keep"] == [(0.0, end)]
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=end - 2.0, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    assert p["duration"] >= end - 2.0 - 1e-6


def test_motion_gate_skips_dissimilar_frames():
    ws = _speech([1.5, 2.0])
    end = ws[-1]["end"] + 0.1
    first_gap_end = ws[8]["start"]
    diff = lambda a, b: 40.0 if b < first_gap_end + 0.1 else 3.0  # noqa: E731
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30, frame_diff=diff)
    assert p["cuts"] == 1 and p["skipped_for_motion"] == 1
    assert p["removed"][0][0] > first_gap_end
    # unmeasurable frames -> no removal
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: None)
    assert p["cuts"] == 0 and p["skipped_for_motion"] == 2


def test_scene_change_inside_span_allows_cut_near_one_blocks_it():
    ws = _speech([1.5])
    end = ws[-1]["end"] + 0.1
    gap = (ws[7]["end"], ws[8]["start"])
    mid = sum(gap) / 2
    # a hard cut in the middle of the removed span: the join IS the cut
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 99.0, scene_cuts=[mid])
    assert p["cuts"] == 1
    # a hard cut 0.1s after the removed span would leave a 3-frame flash
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0, scene_cuts=[gap[1] - 0.02])
    assert p["cuts"] == 0 and p["skipped_for_motion"] == 1


def test_filler_removed_with_quietest_refinement_inside_limits():
    ws = [W("We", 1.0, 1.3), W("um", 1.6, 1.9), W("go", 2.4, 2.7), W("now.", 2.75, 3.0)]
    calls = []

    def quietest(lo, hi):
        calls.append((lo, hi))
        return (lo + hi) / 2

    p = tighten.plan(ws, 0.0, 4.0, min_clip_seconds=1, max_fraction=0.5, fps=25,
                     frame_diff=lambda a, b: 0.0, quietest=quietest)
    assert p["cuts"] == 1 and p["removed"][0][2] == "filler"
    a, b, _ = p["removed"][0]
    assert 1.3 < a <= 1.6 + 0.02 and 1.9 - 0.02 <= b < 2.4  # filler gone, words intact
    assert calls and all(hi > lo for lo, hi in calls)


def test_edges_are_never_touched():
    ws = [W("um", 0.2, 0.5), W("hello", 2.0, 2.4), W("there.", 2.5, 2.9), W("uh", 4.0, 4.3)]
    p = tighten.plan(ws, 0.0, 4.5, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    for a, b, _ in p["removed"]:
        assert a > 0.5 and b < 4.0


# --- remapping ------------------------------------------------------------------

def test_remap_words_onto_new_timeline():
    keep = [(0.0, 2.0), (3.0, 5.0)]  # 1s removed at 2-3 (clip-relative)
    words = [W("before", 9.0, 9.5), W("a", 10.2, 10.5), W("um", 12.2, 12.6),
             W("b", 13.5, 13.9), W("after", 15.2, 15.5)]
    out = tighten.remap_words(words, 10.0, 15.0, keep)
    assert [w["word"].strip() for w in out] == ["a", "b"]
    assert out[0]["start"] == pytest.approx(0.2) and out[1]["start"] == pytest.approx(2.5)
    ctx = tighten.remap_words(words, 10.0, 15.0, keep, context=True)
    assert [w["word"].strip() for w in ctx] == ["before", "a", "b", "after"]
    assert ctx[0]["start"] == pytest.approx(-1.0)
    assert ctx[-1]["start"] == pytest.approx(4.0 + 0.2)  # new_dur 4 + 0.2 past the end


def test_remapped_double_take_no_longer_flagged_by_qa():
    import clip_qa
    toks = "so we need to we need to fix it now".split()
    words = [W(t, 10 + i * 0.3, 10 + i * 0.3 + 0.25) for i, t in enumerate(toks)]
    assert clip_qa.find_double_takes(words)
    p = tighten.plan(words, 10.0, 13.0, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    assert p["cuts"] == 1 and p["removed"][0][2] == "double_take"
    remapped = tighten.remap_words(words, 10.0, 13.0, p["keep"])
    assert [w["word"].strip() for w in remapped] == "so we need to fix it now".split()
    assert not clip_qa.find_double_takes(remapped)


def test_words_transcript_shape():
    t = tighten.words_transcript([W("Hi.", 0.0, 0.3), W("Yo", 0.5, 0.7)], language="en")
    assert t["timeline"] == "clip" and len(t["segments"]) == 2
    assert t["segments"][0]["words"][0]["word"] == " Hi."


def test_env_switches(monkeypatch):
    monkeypatch.setenv("TIGHTEN_CLIPS", "0")
    assert not tighten.enabled(True)
    monkeypatch.setenv("TIGHTEN_CLIPS", "1")
    assert tighten.enabled(True) and not tighten.enabled(False)
    monkeypatch.setenv("TIGHTEN_MAX_REMOVED_FRACTION", "0.1")
    assert tighten.max_removed_fraction() == 0.1
    monkeypatch.setenv("TIGHTEN_JOIN_FADE_MS", "50")
    assert tighten.join_fade_seconds() == 0.012


def test_filtergraph_uses_same_intervals_for_video_and_audio():
    g = ffmpeg_utils.tightened_filtergraph([(0.0, 1.0), (2.0, 3.5)], 0.01, 2.5)
    assert "[0:v]trim=start=2.000000000:end=3.500000000" in g
    assert "[0:a]atrim=start=2.000000000:end=3.500000000" in g
    assert "concat=n=2:v=1:a=1" in g and g.count("afade=t=in") >= 2
    # with fps the video is cut by frame index: whole frames, no float edge cases
    g = ffmpeg_utils.tightened_filtergraph([(0.0, 1.0), (2.0, 3.5)], 0.01, 2.5, fps=30000 / 1001)
    assert "trim=start_frame=60:end_frame=105" in g


# --- real ffmpeg ----------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg not installed")

BURSTS = [(0.5, 3.0), (4.5, 7.0), (7.3, 10.0), (12.0, 15.0), (15.2, 19.0)]


def _probe_streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,duration", "-of", "json", path],
                         capture_output=True, text=True).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def _onsets_audio(path):
    """Times where the sine starts again (silencedetect silence_end)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-af",
                        "silencedetect=n=-40dB:d=0.1", "-f", "null", "-"],
                       capture_output=True, text=True)
    import re
    return [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]


def _luma_series(path, fps=30):
    """Mean luma sampled at ``fps`` on the file's own timestamps."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vf",
                        f"fps={fps}:start_time=0,scale=8:8,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    b = r.stdout
    return [sum(b[i:i + 64]) / 64 for i in range(0, len(b) - 63, 64)]


@needs_ffmpeg
def test_real_cut_trims_gaps_and_keeps_sync(tmp_path):
    src = str(tmp_path / "src.mp4")
    cond = "+".join(f"between(t,{a},{b})" for a, b in BURSTS)
    # The picture flips dark->bright 1.0s into each burst (and back at its
    # middle-ish), far from any join, so the joins sit on identical frames
    # (motion gate passes) and a sync error shows as a luma edge that is not
    # exactly 1.0s after its audio onset.
    vcond = "+".join(f"between(T,{a + 1.0},{b - 0.6})" for a, b in BURSTS)
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", f"nullsrc=s=320x180:r=30:d=20,format=gray,"
                                         f"geq=lum='if({vcond},200,40)'",
                    "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*440*t)*({cond})':s=48000:d=20",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
                   check=True)
    words = []
    for a, b in BURSTS:
        t = a
        while t + 0.2 <= b + 1e-6:
            words.append(W("la", round(t, 3), round(t + 0.2, 3)))
            t += 0.25
    p = tighten.plan_for_clip(src, 0.0, 20.0, words, min_clip_seconds=10, log=lambda *_: None)
    # the 1.5s and 2.0s gaps go, the 0.3s and 0.2s pauses stay
    assert p["cuts"] == 2 and p["skipped_for_motion"] == 0
    # (the tone's tail runs ~50ms past the last labelled word, so the quiet
    # check keeps a little more air than the word gaps alone would)
    assert 2.9 <= p["removed_seconds"] <= 1.5 + 2.0 - 2 * tighten.KEEP_GAP + 0.1

    out = str(tmp_path / "out.mp4")
    ffmpeg_utils.cut_clip(src, out, 0.0, 20.0, 1, keep_ranges=p["keep"], join_fade=0.01)
    d = _probe_streams(out)
    assert abs(d["video"] - d["audio"]) <= 1 / 30 + 1e-3
    assert d["video"] == pytest.approx(p["duration"], abs=1 / 30 + 0.03)

    # every long silence is gone; onsets land where the plan says
    expected = [tighten.remap_time(a, p["keep"]) for a, _ in BURSTS[1:]]
    all_onsets = _onsets_audio(out)
    # (silencedetect also reports the clip start and the trailing silence's end)
    onsets = [min(all_onsets, key=lambda o: abs(o - want)) for want in expected]
    for got, want in zip(onsets, expected):
        assert got == pytest.approx(want, abs=0.06)
    # the brightness edge 1.0s into each burst stays 1.0s after the AUDIO
    # onset measured in the output: A/V in sync to within a frame
    lum = _luma_series(out)
    rises = [i / 30 for i in range(1, len(lum)) if lum[i - 1] < 120 <= lum[i]]
    assert len(rises) == len(BURSTS)
    for got_onset, rise in zip(onsets, rises[1:]):
        assert rise - got_onset == pytest.approx(1.0, abs=1 / 30 + 0.03)


def test_motion_threshold_adapts_to_natural_motion():
    import numpy as np
    static = np.zeros((10, 36, 64), dtype=np.uint8)
    assert tighten.motion_threshold_from_thumbs(static) == tighten.MOTION_FLOOR
    shaky = np.stack([np.full((36, 64), (k % 2) * 8, dtype=np.uint8) for k in range(10)])
    assert tighten.motion_threshold_from_thumbs(shaky) == pytest.approx(12.0)
    wild = np.stack([np.full((36, 64), (k % 2) * 40, dtype=np.uint8) for k in range(10)])
    assert tighten.motion_threshold_from_thumbs(wild) == tighten.MOTION_MAX_DIFF
    ws = _speech([1.5])
    end = ws[-1]["end"] + 0.1
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 10.0, motion_threshold=6.0)
    assert p["cuts"] == 0 and p["skipped_for_motion"] == 1


def _make_source(path, vfr=False):
    cond = "+".join(f"between(t,{a},{b})" for a, b in BURSTS)
    vcond = "+".join(f"between(T,{a + 1.0},{b - 0.6})" for a, b in BURSTS)
    vf = f"nullsrc=s=320x180:r=30:d=20,format=gray,geq=lum='if({vcond},200,40)'"
    extra = []
    if vfr:
        # Drop every other frame in 1-4s and 8-11s: a VFR stream whose
        # r_frame_rate (30) is not its average rate (meeting/screen recordings).
        vf += ",select='not((between(t,1,4)+between(t,8,11))*mod(n,2))'"
        extra = ["-fps_mode", "vfr"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", vf,
                    "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*440*t)*({cond})':s=48000:d=20",
                    *extra, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", path], check=True)


def _burst_words():
    words = []
    for a, b in BURSTS:
        t = a
        while t + 0.2 <= b + 1e-6:
            words.append(W("la", round(t, 3), round(t + 0.2, 3)))
            t += 0.25
    return words


@needs_ffmpeg
def test_vfr_source_stays_in_sync_across_joins(tmp_path):
    src = str(tmp_path / "vfr.mp4")
    _make_source(src, vfr=True)
    fps = tighten.video_fps(src)
    assert fps is not None and abs(float(fps) - 30) > 0.3  # average, not r_frame_rate
    p = tighten.plan_for_clip(src, 0.0, 20.0, _burst_words(), min_clip_seconds=10,
                              log=lambda *_: None)
    assert p["cuts"] == 2
    out = str(tmp_path / "out.mp4")
    ffmpeg_utils.cut_clip(src, out, 0.0, 20.0, 1, keep_ranges=p["keep"], join_fade=0.01,
                          fps=p["fps"])
    d = _probe_streams(out)
    assert abs(d["video"] - d["audio"]) <= 1 / float(fps) + 1e-3
    assert d["audio"] == pytest.approx(p["duration"], abs=0.01)
    all_onsets = _onsets_audio(out)
    lum = _luma_series(out, fps=100)
    rises = [i / 100 for i in range(1, len(lum)) if lum[i - 1] < 120 <= lum[i]]
    assert len(rises) == len(BURSTS)
    # every picture edge (1.0s into its burst) is still 1.0s after the audio
    # onset, before AND after both joins
    for rise in rises[1:]:
        onset = min(all_onsets, key=lambda o: abs(o - (rise - 1.0)))
        assert rise - onset == pytest.approx(1.0, abs=1 / float(fps) + 0.03)


def test_pieces_snap_to_grid_and_drop_empty():
    pieces = ffmpeg_utils.tightened_pieces([(0.0, 1.0), (1.01, 1.015), (2.0, 59.9)], fps=30)
    assert [(sf, ef) for _, _, sf, ef in pieces] == [(0, 30), (60, 1797)]
    assert pieces[-1][1] == pytest.approx(1797 / 30)  # audio ends on the last frame too
    g = ffmpeg_utils.tightened_filtergraph([(0.0, 1.0), (2.0, 59.9)], 0.01, fps="55/2")
    assert g.startswith("[0:v]fps=fps=55/2:start_time=0,split=2[s0][s1]")
    assert "atrim=end=" in g


def test_plan_final_end_on_frame_grid():
    ws = _speech([1.5])
    end = ws[-1]["end"] + 0.123
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    assert abs(p["keep"][-1][1] * 30 - round(p["keep"][-1][1] * 30)) < 1e-6


# --- audio checks (review: speech inside "gaps") -----------------------------------

def _quiet_from(loud_spans, step=tighten.QUIET_STEP):
    def quiet(lo, hi):
        n = max(1, int(round((hi - lo) / step)))
        return [not any(a <= lo + (k + 0.5) * step < b for a, b in loud_spans) for k in range(n)]
    return quiet


def test_dead_air_with_laugh_shrinks_to_quiet_part():
    ws = _speech([2.0])
    gap = (ws[7]["end"], ws[8]["start"])
    laugh = (gap[0] + 0.3, gap[0] + 0.6)  # unlabelled sound in the first part
    end = ws[-1]["end"] + 0.1
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0, quiet=_quiet_from([laugh]))
    assert p["cuts"] == 1
    a, b, _ = p["removed"][0]
    assert a >= laugh[1] + tighten.KEEP_GAP / 2 - 0.04  # laugh kept, with air after it


def test_dead_air_full_of_sound_is_skipped():
    ws = _speech([1.5])
    gap = (ws[7]["end"], ws[8]["start"])
    sound = [(gap[0] + k * 0.2, gap[0] + k * 0.2 + 0.12) for k in range(8)]  # chatter
    end = ws[-1]["end"] + 0.1
    p = tighten.plan(ws, 0.0, end, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0, quiet=_quiet_from(sound))
    assert p["cuts"] == 0 and p["skipped_for_audio"] == 1


def test_filler_glued_to_word_is_skipped_when_join_not_quiet():
    ws = [W("We", 1.0, 1.3), W("um", 1.31, 1.6), W("go", 1.61, 1.9), W("now.", 2.0, 2.3)]
    speech = [(1.0, 1.9), (2.0, 2.3)]  # continuous voicing through the filler
    p = tighten.plan(ws, 0.0, 3.0, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0, quiet=_quiet_from(speech))
    assert p["cuts"] == 0 and p["skipped_for_audio"] == 1
    # with pauses around it the same filler goes
    ws = [W("We", 1.0, 1.3), W("um", 1.6, 1.9), W("go", 2.4, 2.7), W("now.", 2.75, 3.0)]
    speech = [(1.0, 1.3), (1.6, 1.9), (2.4, 3.0)]
    p = tighten.plan(ws, 0.0, 4.0, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0, quiet=_quiet_from(speech))
    assert p["cuts"] == 1 and p["removed"][0][2] == "filler"


def test_grid_never_rounds_into_kept_words():
    # word ends 1 ms after a frame boundary: nearest-frame rounding would cut it
    ws = [W("We", 1.0, 1.334), W("um", 1.6, 1.9), W("go", 2.233, 2.6), W("now.", 2.7, 3.0)]
    p = tighten.plan(ws, 0.0, 4.0, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 0.0)
    for a, b, _ in p["removed"]:
        assert a >= 1.334 - 1e-3 and b <= 2.233 + 1e-3


def test_quiet_fn_threshold_from_noise_floor():
    import numpy as np
    sr = 16000
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(sr * 4) * 10 ** (-60 / 20)).astype(np.float32)  # -60 dB floor
    y[int(2.5 * sr):int(3.0 * sr)] += 0.1  # ~-20 dB "speech"
    q = tighten.quiet_fn(None, 0.5, 4.0, samples=y, sr=sr)  # samples start at clip_start-0.5
    assert all(q(0.6, 1.5))
    assert not any(q(2.55, 2.95))  # absolute source times


# --- review items 3 / 6 / 8 ----------------------------------------------------------

def test_remap_keeps_word_running_past_the_end():
    keep = [(0.0, 2.0), (3.0, 5.0)]
    words = [W("a", 10.2, 10.5), W("chopped", 14.8, 15.4), W("after", 15.6, 15.9)]
    ctx = tighten.remap_words(words, 10.0, 15.0, keep, context=True)
    chopped = [w for w in ctx if w["word"].strip() == "chopped"][0]
    assert chopped["start"] == pytest.approx(3.8) and chopped["end"] == pytest.approx(4.4)
    import clip_qa
    assert clip_qa.ending_problem(ctx, 0.0, 4.0) is not None  # still caught


def test_right_okay_so_are_never_fillers():
    ws = [W("Is", 0.0, 0.2), W("that", 0.25, 0.4), W("true?", 0.45, 0.7),
          W("Right.", 1.2, 1.4), W("Okay.", 1.9, 2.1), W("So", 2.6, 2.8), W("yes.", 3.3, 3.5)]
    assert tighten.find_removable_words(ws) == {}


def test_like_as_question_is_kept():
    ws = [W("It", 0.0, 0.2), W("was", 0.25, 0.4), W("Like?", 0.8, 1.0), W("Yes.", 1.4, 1.6)]
    assert tighten.find_removable_words(ws) == {}


def test_double_take_warning_vs_failure():
    import clip_qa
    m = {"duration": 30.0, "double_takes": [(3.0, "we need to")]}
    hard = clip_qa.evaluate(m, 15, 60)
    assert not hard["passed"] and hard["failures"]
    soft = clip_qa.evaluate(m, 15, 60, double_takes_fatal=False)
    assert soft["passed"] and soft["warnings"] == ["double take at 3.0s: 'we need to'"]


def test_skipped_double_take_is_recorded():
    toks = "so we need to we need to fix it now".split()
    words = [W(t, 10 + i * 0.3, 10 + i * 0.3 + 0.25) for i, t in enumerate(toks)]
    p = tighten.plan(words, 10.0, 13.0, min_clip_seconds=1, max_fraction=0.5, fps=30,
                     frame_diff=lambda a, b: 99.0)
    assert p["cuts"] == 0 and p["skipped_double_takes"] == 1
