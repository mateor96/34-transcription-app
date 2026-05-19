"""Tests for app.services.lmstudio — SSE streaming, error mapping, health check."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.services.exceptions import ProviderModelError, ProviderUnavailableError
from app.services.lmstudio import LMStudioService, _extract_error_message

URL = "http://localhost:1234"


# ── _extract_error_message ───────────────────────────────────────────────────

def test_extract_error_dict_with_message():
    body = '{"error": {"message": "something broke", "code": 400}}'
    assert _extract_error_message(body) == "something broke"


def test_extract_error_string_value():
    body = '{"error": "context window exceeded"}'
    assert _extract_error_message(body) == "context window exceeded"


def test_extract_error_dict_without_message_key():
    body = '{"error": {"type": "x"}}'
    out = _extract_error_message(body)
    assert "type" in out  # falls back to str() of dict


def test_extract_error_invalid_json_returns_raw():
    body = "not json — just bad request text"
    assert _extract_error_message(body) == "not json — just bad request text"


def test_extract_error_empty_string():
    assert _extract_error_message("") == "no detail"


def test_extract_error_truncates_long_body():
    body = "x" * 1000
    assert len(_extract_error_message(body)) <= 300


# ── provider_name ────────────────────────────────────────────────────────────

def test_provider_name_no_model():
    assert LMStudioService(model="").provider_name() == "lmstudio"


def test_provider_name_with_model():
    assert LMStudioService(model="qwen/qwen3-vl-30b").provider_name() == "lmstudio/qwen/qwen3-vl-30b"


def test_default_base_url():
    assert LMStudioService().base_url == "http://localhost:1234"


def test_base_url_strips_trailing_slash():
    assert LMStudioService(base_url="http://host:1234/").base_url == "http://host:1234"


# ── stream_summarize: happy path ─────────────────────────────────────────────

@respx.mock
async def test_stream_summarize_yields_tokens():
    sse = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        b'data: [DONE]\n'
    )
    respx.post(f"{URL}/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse))
    svc = LMStudioService(base_url=URL)
    tokens = [t async for t in svc.stream_summarize("hello")]
    assert tokens == ["Hello", " world"]


@respx.mock
async def test_stream_summarize_skips_done_marker():
    sse = (
        b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
        b'data: [DONE]\n'
        b'data: {"choices":[{"delta":{"content":"AFTER"}}]}\n'  # should never be yielded
    )
    respx.post(f"{URL}/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse))
    svc = LMStudioService(base_url=URL)
    tokens = [t async for t in svc.stream_summarize("hello")]
    assert tokens == ["x"]
    assert "AFTER" not in tokens


@respx.mock
async def test_stream_summarize_ignores_malformed_chunks():
    sse = (
        b'data: not json\n'
        b'data: {"choices":[{"delta":{"content":"valid"}}]}\n'
        b'data: {"choices":[{}]}\n'  # delta missing
        b'data: {}\n'  # no choices
    )
    respx.post(f"{URL}/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse))
    svc = LMStudioService(base_url=URL)
    tokens = [t async for t in svc.stream_summarize("hello")]
    assert tokens == ["valid"]


@respx.mock
async def test_stream_summarize_ignores_keepalive_and_blank_lines():
    sse = (
        b'\n'
        b': keepalive\n'
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        b'\n'
    )
    respx.post(f"{URL}/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse))
    svc = LMStudioService(base_url=URL)
    tokens = [t async for t in svc.stream_summarize("hello")]
    assert tokens == ["ok"]


# ── stream_summarize: request payload ─────────────────────────────────────────

@respx.mock
async def test_stream_summarize_omits_model_field_when_unset():
    route = respx.post(f"{URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b'data: [DONE]\n')
    )
    svc = LMStudioService(base_url=URL, model="")
    [_ async for _ in svc.stream_summarize("hello")]
    body = route.calls.last.request.read()
    import json
    payload = json.loads(body)
    assert "model" not in payload
    assert payload["stream"] is True


@respx.mock
async def test_stream_summarize_passes_model_field_when_set():
    route = respx.post(f"{URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b'data: [DONE]\n')
    )
    svc = LMStudioService(base_url=URL, model="some-model")
    [_ async for _ in svc.stream_summarize("hello")]
    body = route.calls.last.request.read()
    import json
    payload = json.loads(body)
    assert payload["model"] == "some-model"


@respx.mock
async def test_stream_summarize_includes_transcript_in_prompt():
    route = respx.post(f"{URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b'data: [DONE]\n')
    )
    svc = LMStudioService(base_url=URL)
    [_ async for _ in svc.stream_summarize("UNIQUE_TRANSCRIPT_MARKER")]
    body = route.calls.last.request.read().decode()
    assert "UNIQUE_TRANSCRIPT_MARKER" in body


# ── stream_summarize: error paths ─────────────────────────────────────────────

@respx.mock
async def test_stream_summarize_400_surfaces_lmstudio_message():
    respx.post(f"{URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            content=b'{"error": {"message": "Multiple models loaded"}}',
        )
    )
    svc = LMStudioService(base_url=URL)
    with pytest.raises(ProviderModelError, match="Multiple models loaded"):
        [_ async for _ in svc.stream_summarize("hello")]


@respx.mock
async def test_stream_summarize_404_raises_model_error():
    respx.post(f"{URL}/v1/chat/completions").mock(
        return_value=httpx.Response(404, content=b'{"error": "model not found"}')
    )
    svc = LMStudioService(base_url=URL, model="missing")
    with pytest.raises(ProviderModelError, match="Model not found"):
        [_ async for _ in svc.stream_summarize("hello")]


@respx.mock
async def test_stream_summarize_500_raises_model_error():
    respx.post(f"{URL}/v1/chat/completions").mock(
        return_value=httpx.Response(500, content=b'{"error": "server crashed"}')
    )
    svc = LMStudioService(base_url=URL)
    with pytest.raises(ProviderModelError):
        [_ async for _ in svc.stream_summarize("hello")]


@respx.mock
async def test_stream_summarize_connect_error_raises_unavailable():
    respx.post(f"{URL}/v1/chat/completions").mock(side_effect=httpx.ConnectError("no route"))
    svc = LMStudioService(base_url=URL)
    with pytest.raises(ProviderUnavailableError, match="Cannot connect"):
        [_ async for _ in svc.stream_summarize("hello")]


@respx.mock
async def test_stream_summarize_timeout_raises_unavailable():
    respx.post(f"{URL}/v1/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    svc = LMStudioService(base_url=URL)
    with pytest.raises(ProviderUnavailableError, match="timed out"):
        [_ async for _ in svc.stream_summarize("hello")]


# ── check_health ─────────────────────────────────────────────────────────────

@respx.mock
async def test_check_health_returns_true_when_models_endpoint_ok():
    respx.get(f"{URL}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    assert await LMStudioService(base_url=URL).check_health() is True


@respx.mock
async def test_check_health_returns_false_on_500():
    respx.get(f"{URL}/v1/models").mock(return_value=httpx.Response(500))
    assert await LMStudioService(base_url=URL).check_health() is False


@respx.mock
async def test_check_health_returns_false_on_connect_error():
    respx.get(f"{URL}/v1/models").mock(side_effect=httpx.ConnectError("no route"))
    assert await LMStudioService(base_url=URL).check_health() is False
