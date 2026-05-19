"""Tests for app.services.factory — transcript formatting and provider dispatch."""
from __future__ import annotations

import pytest

from app.services.anthropic_svc import AnthropicService
from app.services.exceptions import ProviderAuthError
from app.services.factory import (
    SUMMARIZATION_PROMPT,
    format_transcript,
    get_summarizer,
)
from app.services.gemini import GeminiService
from app.services.lmstudio import LMStudioService
from app.services.ollama import OllamaService
from app.services.openai_svc import OpenAIService


# ── format_transcript ────────────────────────────────────────────────────────

def test_format_transcript_empty():
    assert format_transcript([], {}) == ""


def test_format_transcript_basic():
    segs = [{"speaker": "SPEAKER_00", "start": 5.2, "end": 6.1, "text": "Hello"}]
    out = format_transcript(segs, {})
    assert out == "[00:05] Speaker 00: Hello"


def test_format_transcript_uses_speaker_names():
    segs = [{"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "Hi"}]
    out = format_transcript(segs, {"SPEAKER_00": "Mateo"})
    assert "Mateo:" in out
    assert "Speaker 00" not in out


def test_format_transcript_falls_back_to_speaker_label_for_unmapped():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "A"},
        {"speaker": "SPEAKER_01", "start": 2, "end": 3, "text": "B"},
    ]
    out = format_transcript(segs, {"SPEAKER_00": "Mateo"})
    lines = out.split("\n")
    assert "Mateo: A" in lines[0]
    assert "Speaker 01: B" in lines[1]


def test_format_transcript_skips_empty_text():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "Hi"},
        {"speaker": "SPEAKER_00", "start": 2, "end": 3, "text": "  "},
        {"speaker": "SPEAKER_00", "start": 4, "end": 5, "text": "Bye"},
    ]
    out = format_transcript(segs, {})
    assert out.count("\n") == 1
    assert "Hi" in out and "Bye" in out


def test_format_transcript_time_formatting():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0,     "end": 1, "text": "A"},
        {"speaker": "SPEAKER_00", "start": 65.7,  "end": 66, "text": "B"},
        {"speaker": "SPEAKER_00", "start": 3601.4, "end": 3602, "text": "C"},  # > 1hr; MM:SS keeps minute count
    ]
    out = format_transcript(segs, {})
    lines = out.split("\n")
    assert lines[0].startswith("[00:00]")
    assert lines[1].startswith("[01:05]")
    assert lines[2].startswith("[60:01]")


def test_format_transcript_handles_missing_keys():
    """The renderer should be resilient to malformed segments."""
    segs = [{"text": "lone text"}]  # missing speaker, start, end
    out = format_transcript(segs, {})
    assert "Speaker 00: lone text" in out


def test_summarization_prompt_has_transcript_placeholder():
    assert "{transcript}" in SUMMARIZATION_PROMPT


# ── get_summarizer ───────────────────────────────────────────────────────────

def test_get_summarizer_lmstudio():
    svc = get_summarizer({"provider": "lmstudio"})
    assert isinstance(svc, LMStudioService)


def test_get_summarizer_ollama():
    svc = get_summarizer({"provider": "ollama"})
    assert isinstance(svc, OllamaService)


def test_get_summarizer_uses_provided_base_url():
    svc = get_summarizer({"provider": "lmstudio", "base_url": "http://custom:9999"})
    assert svc.base_url == "http://custom:9999"


def test_get_summarizer_lmstudio_default_base_url():
    svc = get_summarizer({"provider": "lmstudio", "base_url": ""})
    assert svc.base_url == "http://localhost:1234"


def test_get_summarizer_ollama_default_base_url():
    svc = get_summarizer({"provider": "ollama", "base_url": ""})
    assert svc.base_url == "http://localhost:11434"


def test_get_summarizer_defaults_to_lmstudio_when_no_provider():
    svc = get_summarizer({})
    assert isinstance(svc, LMStudioService)


def test_get_summarizer_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_summarizer({"provider": "made-up-provider"})


def test_get_summarizer_gemini_requires_key():
    with pytest.raises(ProviderAuthError):
        get_summarizer({"provider": "gemini", "api_key": ""})


def test_get_summarizer_anthropic_requires_key():
    with pytest.raises(ProviderAuthError):
        get_summarizer({"provider": "anthropic", "api_key": ""})


def test_get_summarizer_openai_requires_key():
    with pytest.raises(ProviderAuthError):
        get_summarizer({"provider": "openai", "api_key": ""})


def test_get_summarizer_anthropic_with_key():
    svc = get_summarizer({"provider": "anthropic", "api_key": "sk-test"})
    assert isinstance(svc, AnthropicService)


def test_get_summarizer_openai_with_key():
    svc = get_summarizer({"provider": "openai", "api_key": "sk-test"})
    assert isinstance(svc, OpenAIService)


def test_get_summarizer_gemini_with_key():
    svc = get_summarizer({"provider": "gemini", "api_key": "abc"})
    assert isinstance(svc, GeminiService)
