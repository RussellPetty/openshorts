"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

# Broker Marketplace runs the moment picker on DeepSeek V4.1 Flash via
# Fireworks first (FIREWORKS_API_KEY), then Claude Haiku 5.5 through the
# Broker Marketplace LLM gateway (LLM_GATEWAY_API_KEY), then Gemini. Haiku-first
# was tried 2026-10-09 and picked fewer, more clustered, weaker clips than
# DeepSeek on the same episode (7 vs 11-12).
# LLM_BASE_URL / LLM_MODEL / LLM_API_KEY still override the Fireworks lane.
GATEWAY_DEFAULT_URL = "https://llm.broker-marketplace.com/v1"
GATEWAY_DEFAULT_MODEL = "claude-code/claude-haiku-5-5-medium"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_MAX_TOKENS = 32000
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def gateway_endpoint() -> Optional[dict]:
    """The Broker Marketplace gateway lane (Claude Haiku 5.5), or None when
    LLM_GATEWAY_API_KEY is unset. Effort rides in the model id's tier, so this
    lane never sends reasoning_effort, and ``mode: ask`` keeps Claude Code a
    plain LLM (no tools)."""
    key = (os.environ.get("LLM_GATEWAY_API_KEY") or "").strip()
    if not key:
        return None
    return {
        "base_url": (os.environ.get("LLM_GATEWAY_URL") or GATEWAY_DEFAULT_URL).strip().rstrip("/"),
        "api_key": key,
        "model": (os.environ.get("LLM_GATEWAY_MODEL") or GATEWAY_DEFAULT_MODEL).strip(),
        "gateway": True,
    }


def llm_lanes() -> list:
    """OpenAI-compatible clip-picker lanes in order, as ``("llm", model,
    endpoint)``: DeepSeek (Fireworks or LLM_BASE_URL), then the gateway's
    Claude Haiku 5.5. main.py appends Gemini after these."""
    lanes = []
    if active():
        lanes.append(("llm", model_name(), None))
    gateway = gateway_endpoint()
    if gateway:
        lanes.append(("llm", gateway["model"], gateway))
    return lanes


def _fireworks_key() -> str:
    return (os.environ.get("FIREWORKS_API_KEY") or "").strip()


def base_url() -> str:
    explicit = (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
    if explicit:
        return explicit
    return FIREWORKS_BASE_URL if _fireworks_key() else ""


def model_name() -> str:
    explicit = (os.environ.get("LLM_MODEL") or "").strip()
    if explicit:
        return explicit
    if base_url() == FIREWORKS_BASE_URL:
        return FIREWORKS_DEFAULT_MODEL
    return DEFAULT_MODEL


def _reasoning_effort() -> Optional[str]:
    """``LLM_REASONING_EFFORT`` (none|low|medium|high), default "low" on
    Fireworks. Unbounded reasoning on the detail pass spent the whole output
    budget thinking and returned an empty body (prod, 5-oct-2026)."""
    raw = (os.environ.get("LLM_REASONING_EFFORT") or "").strip().lower()
    if raw:
        return None if raw == "default" else raw
    return "low" if base_url() == FIREWORKS_BASE_URL else None


def _max_tokens() -> int:
    try:
        return int(os.environ.get("LLM_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    except ValueError:
        return DEFAULT_MAX_TOKENS


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or _fireworks_key() or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  endpoint: Optional[dict] = None) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    gateway = bool(endpoint and endpoint.get("gateway"))
    url = f"{(endpoint or {}).get('base_url') or base_url()}/chat/completions"
    model = model or (endpoint or {}).get("model") or model_name()
    headers = ({"Authorization": f"Bearer {endpoint['api_key']}", "Content-Type": "application/json"}
               if endpoint else _headers())
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    def _request(effort):
        last_rejection: Optional[str] = None
        with _client() as client:
            for fmt in _response_formats(schema):
                body = {"model": model, "messages": messages, "temperature": 0.2,
                        "stream": False, "max_tokens": _max_tokens()}
                if effort and not gateway:
                    body["reasoning_effort"] = effort
                if gateway:
                    body["mode"] = "ask"
                if fmt is not None:
                    body["response_format"] = fmt
                resp = client.post(url, json=body, headers=headers)
                if fmt is not None and _is_format_rejection(resp):
                    # The server does not know this response_format flavour; the
                    # next loop iteration asks for a looser one.
                    last_rejection = resp.text[:200]
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
                return resp.json()
        raise RuntimeError(
            f"LLM server rejected every response_format variant: {last_rejection}")

    def _content(data):
        choices = data.get("choices") or []
        if not choices:
            return "", None
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return text, choices[0].get("finish_reason")

    effort = None if gateway else _reasoning_effort()
    data = _request(effort)
    text, finish = _content(data)
    if finish == "length" and effort and effort != "none":
        # Ran out of output budget (usually on reasoning) before finishing the
        # JSON: ask once more with reasoning off rather than fail the stage.
        print(f"⚠️ LLM hit max_tokens (reasoning_effort={effort}); retrying with reasoning off")
        data = _request("none")
        text, finish = _content(data)
    if not text.strip():
        raise ValueError(f"Gemini returned an empty response body. (finish_reason={finish})")
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost
