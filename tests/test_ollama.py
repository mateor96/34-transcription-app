"""Tests for app.services.ollama — NDJSON streaming, error mapping, health check."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.services.exceptions import ProviderModelError, ProviderUnavailableError
from app.services.ollama import OllamaService

URL = "http://localhost:11434"


# ── basic properties ─────────────────────────────────────────────────────────

def test_provider_name_includes_model():
    assert OllamaService(model="llama3:8b").provider_name() == "ollama/llama3:8b"


def test_default_base_url():
    assert OllamaService().base_url == "http://localhost:11434"


def test_default_model():
    assert OllamaService().model == "llama3.2:3b"


def test_base_url_strips_trailing_slash():
    assert OllamaService(base_url="http://host:11434/").base_url == "http://host:11434"


# ── stream_chat: happy path ─────────────────────────────────────────────

@respx.mock
async def test_stream_chat_yields_ndjson_tokens():
    ndjson = (
        json.dumps({"message": {"content": "Hello"}, "done": False}).encode() + b"\n"
        + json.dumps({"message": {"content": " world"}, "done": False}).encode() + b"\n"
        + json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n"
    )
    respx.post(f"{URL}/api/chat").mock(return_value=httpx.Response(200, content=ndjson))
    svc = OllamaService(base_url=URL)
    tokens = [t async for t in svc.stream_chat("hello")]
    assert tokens == ["Hello", " world"]


@respx.mock
async def test_stream_chat_stops_on_done_flag():
    ndjson = (
        json.dumps({"message": {"content": "x"}, "done": True}).encode() + b"\n"
        + json.dumps({"message": {"content": "AFTER"}, "done": False}).encode() + b"\n"
    )
    respx.post(f"{URL}/api/chat").mock(return_value=httpx.Response(200, content=ndjson))
    svc = OllamaService(base_url=URL)
    tokens = [t async for t in svc.stream_chat("hello")]
    assert tokens == ["x"]


@respx.mock
async def test_stream_chat_skips_invalid_json_lines():
    ndjson = (
        b"this is not json\n"
        + json.dumps({"message": {"content": "valid"}, "done": False}).encode() + b"\n"
        + json.dumps({"done": True}).encode() + b"\n"
    )
    respx.post(f"{URL}/api/chat").mock(return_value=httpx.Response(200, content=ndjson))
    svc = OllamaService(base_url=URL)
    tokens = [t async for t in svc.stream_chat("hello")]
    assert tokens == ["valid"]


@respx.mock
async def test_stream_chat_skips_blank_lines():
    ndjson = (
        b"\n"
        + json.dumps({"message": {"content": "x"}, "done": False}).encode() + b"\n"
        + b"\n"
        + json.dumps({"done": True}).encode() + b"\n"
    )
    respx.post(f"{URL}/api/chat").mock(return_value=httpx.Response(200, content=ndjson))
    svc = OllamaService(base_url=URL)
    tokens = [t async for t in svc.stream_chat("hello")]
    assert tokens == ["x"]


# ── stream_chat: request payload ─────────────────────────────────────────

@respx.mock
async def test_stream_chat_sends_model_in_payload():
    route = respx.post(f"{URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({"message": {"content": "ok"}, "done": False}).encode() + b"\n"
            + json.dumps({"done": True}).encode() + b"\n",
        )
    )
    svc = OllamaService(base_url=URL, model="llama3:8b")
    [_ async for _ in svc.stream_chat("hello")]
    body = json.loads(route.calls.last.request.read())
    assert body["model"] == "llama3:8b"
    assert body["stream"] is True


@respx.mock
async def test_stream_chat_includes_transcript():
    route = respx.post(f"{URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({"message": {"content": "ok"}, "done": False}).encode() + b"\n"
            + json.dumps({"done": True}).encode() + b"\n",
        )
    )
    svc = OllamaService(base_url=URL)
    [_ async for _ in svc.stream_chat("UNIQUE_MARKER")]
    body = route.calls.last.request.read().decode()
    assert "UNIQUE_MARKER" in body


# ── stream_chat: error paths ─────────────────────────────────────────────

@respx.mock
async def test_stream_chat_404_with_pull_hint():
    respx.post(f"{URL}/api/chat").mock(return_value=httpx.Response(404, content=b"not found"))
    svc = OllamaService(base_url=URL, model="missing")
    with pytest.raises(ProviderModelError, match="ollama pull missing"):
        [_ async for _ in svc.stream_chat("hello")]


@respx.mock
async def test_stream_chat_400_surfaces_body():
    respx.post(f"{URL}/api/chat").mock(return_value=httpx.Response(400, content=b'{"error":"bad params"}'))
    svc = OllamaService(base_url=URL)
    with pytest.raises(ProviderModelError, match="bad params"):
        [_ async for _ in svc.stream_chat("hello")]


@respx.mock
async def test_stream_chat_connect_error_raises_unavailable():
    respx.post(f"{URL}/api/chat").mock(side_effect=httpx.ConnectError("no route"))
    svc = OllamaService(base_url=URL)
    with pytest.raises(ProviderUnavailableError, match="ollama serve"):
        [_ async for _ in svc.stream_chat("hello")]


@respx.mock
async def test_stream_chat_timeout_raises_unavailable():
    respx.post(f"{URL}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    svc = OllamaService(base_url=URL)
    with pytest.raises(ProviderUnavailableError, match="timed out"):
        [_ async for _ in svc.stream_chat("hello")]


# ── check_health ─────────────────────────────────────────────────────────────

@respx.mock
async def test_check_health_returns_true_on_200():
    respx.get(f"{URL}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    assert await OllamaService(base_url=URL).check_health() is True


@respx.mock
async def test_check_health_returns_false_on_connect_error():
    respx.get(f"{URL}/api/tags").mock(side_effect=httpx.ConnectError("no route"))
    assert await OllamaService(base_url=URL).check_health() is False


@respx.mock
async def test_check_health_returns_false_on_500():
    respx.get(f"{URL}/api/tags").mock(return_value=httpx.Response(500))
    assert await OllamaService(base_url=URL).check_health() is False
