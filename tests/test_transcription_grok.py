"""Grok STT tiers in transcription.py: gateway -> xAI API keys -> Whisper."""
import json
import os

import httpx
import pytest

import transcription

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "grok_stt_sample.json")

SAMPLE = {"text": "In Syg was kind of a, a big draw for me. Oh, that's awesome.",
          "language": "en", "duration": 7.7,
          "words": [{"text": t, "start": s, "end": e} for t, s, e in [
              ("In", 0.08, 0.2), ("Syg", 0.3, 0.58), ("was", 0.76, 0.9), ("kind", 1.0, 1.24),
              ("of", 1.26, 1.36), ("a,", 1.44, 1.46), ("a", 1.76, 1.78), ("big", 1.98, 2.24),
              ("draw", 2.46, 2.68), ("for", 2.74, 2.86), ("me.", 2.9, 2.96),
              ("Oh,", 5.84, 6.0), ("that's", 6.1, 6.4), ("awesome.", 6.5, 7.68)]]}


def _sample():
    try:
        with open(FIXTURE) as f:
            return json.load(f)
    except OSError:
        return SAMPLE


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("LLM_GATEWAY_API_KEY", "LLM_GATEWAY_URL", "GROK_API_KEY", "GROK_API_KEY_FALLBACK",
              "TRANSCRIBE_ENGINE", "GROK_STT_MODEL", "GROK_GATEWAY_MAX_CONCURRENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROK_STT_SLOT_DIR", str(tmp_path / "slots"))
    return monkeypatch


def _fake_media(monkeypatch, duration=60.0):
    monkeypatch.setattr(transcription, "media_duration", lambda p: duration)

    def fake_extract(src, dst, start=0.0, duration=None, bitrate_kbps=40):
        with open(dst, "wb") as f:
            f.write(b"ID3fake-mp3")
    monkeypatch.setattr(transcription, "extract_audio", fake_extract)


def _transport(monkeypatch, handler):
    calls = []

    def wrapped(request):
        calls.append(request)
        return handler(request, len(calls))
    monkeypatch.setattr(transcription, "_client",
                        lambda **kw: httpx.Client(transport=httpx.MockTransport(wrapped)))
    return calls


# --- parsing -------------------------------------------------------------------

def test_fixture_to_segments_and_words():
    body = _sample()
    words = transcription.stitch_chunk_words([((0.0, 60.0, 0.0, float("inf")), body)])
    assert len(words) == len(body["words"])
    assert words[0]["word"].startswith(" ") and words[0]["start"] == body["words"][0]["start"]
    segs = transcription.build_segments(words)
    assert sum(len(s["words"]) for s in segs) == len(words)
    for s in segs:
        assert s["end"] - s["start"] <= transcription.SEGMENT_MAX_SECONDS + 3
        assert s["text"] == " ".join(w["word"].strip() for w in s["words"])
        # a segment only ends mid-sentence at a pause or the length cap
    assert any(s["text"].endswith(".") for s in segs)
    # disfluent repeat kept for tightening
    assert "a, a" in " ".join(s["text"] for s in segs)


def test_segments_split_on_long_gap_and_length():
    ws = [{"word": " a", "start": 0.0, "end": 0.2}, {"word": " b", "start": 1.5, "end": 1.7}]
    assert len(transcription.build_segments(ws)) == 2
    ws = [{"word": " w", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(40)]
    segs = transcription.build_segments(ws)
    assert all(s["end"] - s["start"] <= 15.0 for s in segs) and len(segs) >= 2


# --- chunking ------------------------------------------------------------------

def test_single_chunk_when_small():
    assert transcription.plan_chunks(1200) == [(0.0, 1200.0, 0.0, float("inf"))]


def test_chunks_overlap_and_own_disjoint_ranges():
    ch = transcription.plan_chunks(3 * 3600)
    assert len(ch) == 9
    assert ch[0][:3] == (0.0, 1203.0, 0.0) and ch[1][0] == 1197.0 and ch[1][2] == 1200.0
    assert ch[-1][3] == float("inf")
    for (a, b), (c, d) in zip([(x[2], x[3]) for x in ch], [(x[2], x[3]) for x in ch[1:]]):
        assert b == c


def test_stitch_offsets_and_dedupes_seams():
    seam = 1200.0
    c0 = (0.0, seam + 3, 0.0, seam)
    c1 = (seam - 3, 2000.0, seam, float("inf"))
    # the word at 1199.5-1200.4 appears in both chunks; owned by chunk 0
    b0 = {"words": [{"text": "x", "start": 1190.0, "end": 1190.3},
                    {"text": "seam", "start": 1199.5, "end": 1200.4},
                    {"text": "dup", "start": 1201.0, "end": 1201.2}]}
    b1 = {"words": [{"text": "seam", "start": 2.5, "end": 3.4},   # 1199.5 abs
                    {"text": "dup", "start": 4.0, "end": 4.2},    # 1201.0 abs
                    {"text": "y", "start": 10.0, "end": 10.2}]}
    words = transcription.stitch_chunk_words([(c0, b0), (c1, b1)])
    assert [w["word"].strip() for w in words] == ["x", "seam", "dup", "y"]
    assert words[2]["start"] == pytest.approx(1201.0) and words[3]["start"] == pytest.approx(1207.0)


# --- tiers ----------------------------------------------------------------------

def test_targets_order_and_dedupe(env):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("GROK_API_KEY", "k1")
    env.setenv("GROK_API_KEY_FALLBACK", "k1")
    t = transcription.stt_targets()
    assert [x["tier"] for x in t] == ["grok-proxy", "grok-api"]
    assert t[0]["url"] == "https://llm.broker-marketplace.com/v1/grok/stt"
    assert t[1]["url"] == "https://api.x.ai/v1/stt"
    env.setenv("LLM_GATEWAY_URL", "http://gw.internal")
    assert transcription.stt_targets()[0]["url"] == "http://gw.internal/v1/grok/stt"


def test_proxy_serves_and_fields_precede_file(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    _fake_media(env)
    calls = _transport(env, lambda req, n: httpx.Response(200, json=_sample()))
    r = transcription.transcribe(str(tmp_path / "x.mp4"))
    assert r["engine"] == "grok-proxy" and r["language"] == "en" and r["segments"]
    req = calls[0]
    assert req.headers["Authorization"] == "Bearer gw"
    body = req.read()
    assert body.index(b'name="model"') < body.index(b'name="file"')
    assert b"grok-voice-transcribe-2.0" in body


def test_proxy_429_retried_then_succeeds(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    _fake_media(env)
    sleeps = []
    env.setattr(transcription.time, "sleep", lambda s: sleeps.append(s))
    calls = _transport(env, lambda req, n: httpx.Response(429) if n < 3
                       else httpx.Response(200, json=_sample()))
    r = transcription.transcribe(str(tmp_path / "x.mp4"))
    assert r["engine"] == "grok-proxy" and len(calls) == 3
    # longer, jittered waits for the gateway's concurrency cap
    assert 3.0 <= sleeps[0] <= 4.5 and 8.0 <= sleeps[1] <= 12.0


def test_proxy_4xx_not_retried_falls_to_api_key(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("GROK_API_KEY", "k1")
    _fake_media(env)
    env.setattr(transcription.time, "sleep", lambda s: None)

    def handler(req, n):
        if "llm.broker-marketplace.com" in str(req.url):
            return httpx.Response(400, text="secret transcript text")
        return httpx.Response(200, json=_sample())
    calls = _transport(env, handler)
    logs = []
    r = transcription.transcribe(str(tmp_path / "x.mp4"), log=logs.append)
    assert r["engine"] == "grok-api"
    assert [c.url.host for c in calls] == ["llm.broker-marketplace.com", "api.x.ai"]
    assert calls[1].headers["Authorization"] == "Bearer k1"
    joined = "\n".join(logs)
    assert "HTTP 400" in joined and "secret transcript" not in joined and "gw" not in joined.split()


def test_api_5xx_and_network_retried_then_second_key(env, tmp_path):
    env.setenv("GROK_API_KEY", "k1")
    env.setenv("GROK_API_KEY_FALLBACK", "k2")
    _fake_media(env)
    sleeps = []
    env.setattr(transcription.time, "sleep", lambda s: sleeps.append(s))

    def handler(req, n):
        if req.headers["Authorization"] == "Bearer k1":
            if n % 2:
                raise httpx.ConnectError("boom https://api.x.ai/?k=secret")
            return httpx.Response(503)
        return httpx.Response(200, json=_sample())
    calls = _transport(env, handler)
    logs = []
    r = transcription.transcribe(str(tmp_path / "x.mp4"), log=logs.append)
    assert r["engine"] == "grok-api"
    k1 = [c for c in calls if c.headers["Authorization"] == "Bearer k1"]
    assert len(k1) == 4 and sleeps[:3] == [1.0, 2.0, 4.0]  # 1 try + 3 retries
    assert "secret" not in "\n".join(logs)


def test_api_400_not_retried(env, tmp_path):
    env.setenv("GROK_API_KEY", "k1")
    _fake_media(env)
    env.setattr(transcription, "_transcribe_whisper",
                lambda *a, **k: {"engine": "whisper-small", "segments": []})
    calls = _transport(env, lambda req, n: httpx.Response(401))
    r = transcription.transcribe(str(tmp_path / "x.mp4"))
    assert len(calls) == 1 and r["engine"] == "whisper-small"


def test_everything_fails_falls_back_to_whisper(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("GROK_API_KEY", "k1")
    _fake_media(env)
    env.setattr(transcription.time, "sleep", lambda s: None)
    _transport(env, lambda req, n: httpx.Response(500))
    seen = {}

    def fake_whisper(path, word_timestamps=True, log=None, echo_segments=False):
        seen["path"] = path
        return {"engine": "whisper-small", "segments": [], "text": "", "language": "en"}
    env.setattr(transcription, "_transcribe_whisper", fake_whisper)
    logs = []
    r = transcription.transcribe(str(tmp_path / "x.mp4"), log=logs.append)
    assert r["engine"] == "whisper-small" and seen["path"].endswith("x.mp4")
    assert any("falling back to local Whisper" in m for m in logs)


def test_invalid_json_moves_on(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("GROK_API_KEY", "k1")
    _fake_media(env)
    _transport(env, lambda req, n: httpx.Response(200, text="<html>") if n == 1
               else httpx.Response(200, json=_sample()))
    assert transcription.transcribe(str(tmp_path / "x.mp4"))["engine"] == "grok-api"


def test_whisper_engine_skips_grok(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("TRANSCRIBE_ENGINE", "whisper")
    calls = _transport(env, lambda req, n: httpx.Response(200, json=_sample()))
    env.setattr(transcription, "_transcribe_whisper",
                lambda *a, **k: {"engine": "whisper-small", "segments": []})
    assert transcription.transcribe(str(tmp_path / "x.mp4"))["engine"] == "whisper-small"
    assert not calls


def test_no_keys_goes_straight_to_whisper(env, tmp_path):
    env.setattr(transcription, "_transcribe_whisper",
                lambda *a, **k: {"engine": "whisper-small", "segments": []})
    assert transcription.transcribe(str(tmp_path / "x.mp4"))["engine"] == "whisper-small"


def test_long_audio_chunks_concurrently_and_offsets(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    _fake_media(env, duration=2 * 3600.0)
    starts = []
    orig = transcription.extract_audio

    def rec(src, dst, start=0.0, duration=None, bitrate_kbps=40):
        starts.append(start)
        orig(src, dst, start, duration, bitrate_kbps)
    env.setattr(transcription, "extract_audio", rec)
    body = {"language": "en", "words": [{"text": "hi.", "start": 5.0, "end": 5.3}]}
    _transport(env, lambda req, n: httpx.Response(200, json=body))
    r = transcription.transcribe(str(tmp_path / "x.mp4"))
    n = len(transcription.plan_chunks(2 * 3600.0))
    assert n > 1 and len(starts) == n
    got = sorted(w["start"] for s in r["segments"] for w in s["words"])
    assert got == pytest.approx(sorted(a + 5.0 for a in starts))


def test_chunks_by_duration_even_when_small():
    # 25 min fits in 30 MiB but is longer than one 20-min chunk
    ch = transcription.plan_chunks(25 * 60)
    assert len(ch) == 2 and ch[0][3] == 1200.0


def test_seam_dedupe_tolerates_timing_disagreement():
    seam = 1200.0
    c0 = (0.0, seam + 3, 0.0, seam)
    c1 = (seam - 3, 2000.0, seam, float("inf"))
    b0 = {"words": [{"text": "Hello", "start": 1199.97, "end": 1200.2}]}
    b1 = {"words": [{"text": "hello,", "start": 3.02, "end": 3.3},   # 1200.02 abs
                    {"text": "world", "start": 3.4, "end": 3.6}]}
    words = transcription.stitch_chunk_words([(c0, b0), (c1, b1)])
    assert [w["word"].strip() for w in words] == ["Hello", "world"]
    assert all("_chunk" not in w for w in words)


@pytest.mark.parametrize("body,ok", [
    ({"text": "hi", "words": [{"text": "hi", "start": 0, "end": 0.2}]}, True),
    ({"text": "", "words": []}, True),            # real silence
    ({"text": "hi there"}, False),                # no words key
    ({"text": "hi there", "words": []}, False),   # text but no timings
    ({"words": None}, False),
    ([], False),
])
def test_valid_body(body, ok):
    assert transcription.valid_stt_body(body) is ok


def test_missing_words_moves_to_next_tier(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("GROK_API_KEY", "k1")
    _fake_media(env)
    _transport(env, lambda req, n: httpx.Response(200, json={"text": "hello"}) if n == 1
               else httpx.Response(200, json=_sample()))
    assert transcription.transcribe(str(tmp_path / "x.mp4"))["engine"] == "grok-api"


def test_paid_read_timeout_not_retried(env, tmp_path):
    env.setenv("GROK_API_KEY", "k1")
    env.setenv("GROK_API_KEY_FALLBACK", "k2")
    _fake_media(env)
    env.setattr(transcription.time, "sleep", lambda s: None)

    def handler(req, n):
        if req.headers["Authorization"] == "Bearer k1":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=_sample())
    calls = _transport(env, handler)
    assert transcription.transcribe(str(tmp_path / "x.mp4"))["engine"] == "grok-api"
    assert sum(c.headers["Authorization"] == "Bearer k1" for c in calls) == 1


def test_stage_deadline_falls_back_to_whisper(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    _fake_media(env, duration=60.0)
    clock = {"t": 1000.0}
    env.setattr(transcription.time, "monotonic", lambda: clock["t"])

    def fake_sleep(s):
        clock["t"] += s
    env.setattr(transcription.time, "sleep", fake_sleep)

    def handler(req, n):
        clock["t"] += 60  # each 429 costs a minute
        return httpx.Response(429)
    calls = _transport(env, handler)
    env.setattr(transcription, "_transcribe_whisper",
                lambda *a, **k: {"engine": "whisper-small", "segments": []})
    logs = []
    r = transcription.transcribe(str(tmp_path / "x.mp4"), log=logs.append)
    assert r["engine"] == "whisper-small"
    assert len(calls) <= 3  # budget = 0.5*60+120 = 150s
    assert any("time budget" in m for m in logs)


def test_stage_deadline_formula():
    t0 = transcription.time.monotonic()
    assert transcription.stage_deadline(60) - t0 == pytest.approx(150, abs=1)
    assert transcription.stage_deadline(10 * 3600) - t0 == pytest.approx(900, abs=1)


def test_gateway_slots_cap_concurrency(env, tmp_path):
    env.setenv("GROK_GATEWAY_MAX_CONCURRENT", "1")
    a, b = transcription._GatewaySlot(), transcription._GatewaySlot()
    assert a.acquire(0, lambda s: None)
    assert not b.acquire(0, lambda s: None)  # held (flock is per open file)
    a.release()
    assert b.acquire(0, lambda s: None)
    b.release()


def test_gateway_busy_moves_to_paid_api(env, tmp_path):
    env.setenv("LLM_GATEWAY_API_KEY", "gw")
    env.setenv("GROK_API_KEY", "k1")
    env.setenv("GROK_GATEWAY_MAX_CONCURRENT", "1")
    env.setattr(transcription, "GATEWAY_SLOT_WAIT_SECONDS", 0.0)
    _fake_media(env)
    held = transcription._GatewaySlot()
    assert held.acquire(0, lambda s: None)
    try:
        calls = _transport(env, lambda req, n: httpx.Response(200, json=_sample()))
        assert transcription.transcribe(str(tmp_path / "x.mp4"))["engine"] == "grok-api"
        assert [c.url.host for c in calls] == ["api.x.ai"]
    finally:
        held.release()
