"""The OpenAI-compatible moment-picker backend (llm_backend.py).

Self-hosters asked for a pipeline that never calls Google. The transcript
passes go to any /chat/completions server when LLM_BASE_URL is set; these
tests pin the contract main.py relies on: same (parsed, cost) shape as the
Gemini stage, schema validation, and the response_format fallback ladder.
"""
import json

import httpx
import pytest

import gemini_worker
import llm_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def _serve(handler, monkeypatch):
    """Route llm_backend's httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 40},
    })


# --- activation -----------------------------------------------------------

def test_inactive_without_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_backend.active() is False
    assert llm_backend.describe() is None


def test_base_url_alone_activates_and_describes(local):
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1"}


def test_explicit_gemini_provider_wins_over_base_url(local, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_backend.active() is False


# --- generate_json ---------------------------------------------------------

def test_returns_validated_payload_and_zero_cost(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _completion({"windows": [{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]})

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:14b"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["auth"] == "Bearer ollama"  # documented placeholder when no key is set
    assert parsed["windows"][0]["score"] == 88
    # llm.test is a remote host: unknown price, NOT labelled as a free local model.
    assert cost["total_cost"] == 0.0 and cost["local"] is False
    assert cost["price_known"] is False and cost["provider"] == "openai-compatible"
    assert cost["input_tokens"] == 120 and cost["output_tokens"] == 40


def test_falls_back_to_json_object_when_schema_mode_is_rejected(local, monkeypatch):
    formats = []

    def handler(request):
        body = json.loads(request.content)
        fmt = (body.get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert formats == ["json_schema", "json_object"]
    assert parsed == {"windows": []}


def test_code_fenced_json_is_accepted(local, monkeypatch):
    """Small models wrap the object in ```json fences even when told not to."""
    def handler(request):
        text = "```json\n" + json.dumps({"windows": []}) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}


def test_schema_violation_raises_instead_of_leaking_into_the_pipeline(local, monkeypatch):
    _serve(lambda r: _completion({"windows": [{"id": "w0"}]}), monkeypatch)
    with pytest.raises(Exception) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "validation error" in str(exc.value)


def test_server_errors_surface_with_status_and_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(503, text="loading model"), monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "503" in str(exc.value) and "loading model" in str(exc.value)


# --- main.py routing (needs the heavy deps; skipped on minimal CI) ------------

def test_stage_routes_to_local_backend_and_retries_transient(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_generate(prompt, schema, model=None, endpoint=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("ConnectError: connection refused")
        return {"windows": [{"id": "w0", "start": 0, "end": 1, "score": 50, "reason": ""}]}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    parsed, cost = main._run_llm_stage(("llm", "qwen2.5:14b"), "prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 2
    assert parsed["windows"][0]["score"] == 50
    assert cost["total_cost"] == 0.0




# --- Broker Marketplace gateway lane (Claude Haiku 5.5 first) ----------------

def test_gateway_lane_off_without_key(monkeypatch):
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    assert llm_backend.gateway_endpoint() is None


def test_gateway_lane_posts_haiku_in_ask_mode_without_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "gw-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_MODEL", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "bt-key")  # would add reasoning_effort=low on Baseten
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _completion({"windows": [{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]})

    _serve(handler, monkeypatch)
    endpoint = llm_backend.gateway_endpoint()
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse, endpoint=endpoint)
    assert parsed["windows"][0]["id"] == "w0"
    assert seen["url"] == "https://llm.broker-marketplace.com/v1/chat/completions"
    assert seen["auth"] == "Bearer gw-key"
    assert seen["body"]["model"] == "claude-code/claude-haiku-5-5-medium"
    assert seen["body"]["mode"] == "ask"
    assert "reasoning_effort" not in seen["body"]


def test_llm_lanes_are_deepseek_then_haiku(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "gw-key")
    monkeypatch.setenv("BASETEN_API_KEY", "bt-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    lanes = llm_backend.llm_lanes()
    assert [lane[1] for lane in lanes] == [
        "deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast", "claude-code/claude-haiku-5-5-medium"]
    assert lanes[0][2] is None and lanes[1][2]["gateway"] is True


def test_llm_lanes_without_gateway_key_is_deepseek_only(monkeypatch):
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "bt-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert [lane[1] for lane in llm_backend.llm_lanes()] == ["deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast"]



# --- cost labels -----------------------------------------------------------

@pytest.mark.parametrize("url,gateway,label,local", [
    ("https://inference.baseten.co/v1/chat/completions", False, "baseten", False),
    ("https://llm.broker-marketplace.com/v1/chat/completions", True, "broker-marketplace-gateway", False),
    ("http://localhost:11434/v1/chat/completions", False, "local-openai-compatible", True),
    ("http://127.0.0.1:8080/v1/chat/completions", False, "local-openai-compatible", True),
])
def test_endpoint_provider_labels(url, gateway, label, local):
    assert llm_backend._endpoint_provider(url, gateway) == (label, local)


# --- response_format rejection is matched on the field, not on "format" --------

def test_unrelated_400_mentioning_format_is_not_a_format_rejection():
    resp = httpx.Response(400, text='{"error": "invalid message format in messages[1]"}')
    assert llm_backend._is_format_rejection(resp) is False


def test_response_format_400_is_a_format_rejection():
    resp = httpx.Response(400, text='{"error": "response_format json_schema is not supported"}')
    assert llm_backend._is_format_rejection(resp) is True
    assert llm_backend._is_format_rejection(httpx.Response(500, text="response_format")) is False


def test_400_with_bad_message_format_surfaces_instead_of_ladder(local, monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="invalid date format")

    _serve(handler, monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 1 and "400" in str(exc.value)


# --- transient-error classification ------------------------------------------

class _CodeErr(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("LLM server 503 from https://x/v1/chat/completions: loading"), True),
    (RuntimeError("LLM server 502 from https://x: bad gateway"), True),
    (RuntimeError("LLM server 400 from https://x: max_tokens must be <= 500"), False),
    (RuntimeError("LLM server 401 from https://x: bad key (request 5021)"), False),
    (RuntimeError("503 UNAVAILABLE. {'error': 'overloaded'}"), True),
    (RuntimeError("429 RESOURCE_EXHAUSTED"), True),
    (_CodeErr("whatever", 500), True),
    (_CodeErr("whatever", 404), False),
    (httpx.ConnectError("connection refused"), True),
    (httpx.ReadTimeout("read timed out"), True),
    (ValueError("Gemini returned an empty response body."), True),
    (ValueError("schema mismatch at item 500"), False),
    (RuntimeError("window 502 not found"), False),
])
def test_is_transient_error(exc, expected):
    assert llm_backend.is_transient_error(exc) is expected


def test_status_code_from_attributes_and_text():
    assert llm_backend.error_status_code(_CodeErr("x", 429)) == 429
    assert llm_backend.error_status_code(RuntimeError("HTTP Error 504: Gateway Timeout")) == 504
    assert llm_backend.error_status_code(RuntimeError("tokens: 500 used")) is None
