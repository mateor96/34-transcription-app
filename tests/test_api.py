"""Tests for app.main — every HTTP endpoint exposed by the FastAPI app.

Uses FastAPI's TestClient against the in-process app with monkeypatched
storage paths (see conftest.py). Heavy collaborators (LLM providers, the
transcription pipeline) are stubbed at module level via monkeypatch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncGenerator

import httpx
import pytest
import respx

from app import db as db_module
from app import main as main_module
from app.services.exceptions import ProviderUnavailableError

from .conftest import SAMPLE_SEGMENTS


# ── helpers ──────────────────────────────────────────────────────────────────

async def _insert_entry(entry_id: str, filename: str = "f.mp3", audio_ext: str | None = None) -> None:
    await db_module.save_transcription(entry_id, filename, SAMPLE_SEGMENTS, audio_ext=audio_ext)


# ── /archive list and entry ───────────────────────────────────────────────────

class TestArchive:
    async def test_archive_list_empty(self, client):
        r = client.get("/archive")
        assert r.status_code == 200
        assert r.json() == []

    async def test_archive_list_returns_summary_flag(self, client):
        await _insert_entry("a")
        r = client.get("/archive")
        assert r.json()[0]["has_summary"] == 0

    async def test_archive_get_returns_full_entry(self, client):
        await _insert_entry("a", filename="meeting.mp3", audio_ext=".mp3")
        r = client.get("/archive/a")
        assert r.status_code == 200
        body = r.json()
        assert body["filename"]  == "meeting.mp3"
        assert body["audio_ext"] == ".mp3"
        assert body["segments"]  == SAMPLE_SEGMENTS

    async def test_archive_get_missing_returns_404(self, client):
        r = client.get("/archive/nope")
        assert r.status_code == 404


# ── /archive/{id}/audio ───────────────────────────────────────────────────────

class TestArchiveAudio:
    async def test_legacy_entry_with_no_audio_ext_returns_404(self, client):
        await _insert_entry("a")  # no audio_ext
        r = client.get("/archive/a/audio")
        assert r.status_code == 404

    async def test_returns_404_when_file_missing_on_disk(self, client):
        await _insert_entry("a", audio_ext=".mp3")
        # Note: the file was never written to AUDIO_DIR
        r = client.get("/archive/a/audio")
        assert r.status_code == 404

    async def test_returns_file_with_correct_mime_m4a(self, client, storage: Path):
        await _insert_entry("a", audio_ext=".m4a")
        (db_module.AUDIO_DIR / "a.m4a").write_bytes(b"fake-m4a-bytes")
        r = client.get("/archive/a/audio")
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mp4"
        assert r.content == b"fake-m4a-bytes"

    async def test_returns_file_with_correct_mime_webm(self, client):
        await _insert_entry("a", audio_ext=".webm")
        (db_module.AUDIO_DIR / "a.webm").write_bytes(b"webm-bytes")
        r = client.get("/archive/a/audio")
        assert r.headers["content-type"] == "audio/webm"

    async def test_returns_file_with_correct_mime_mp3(self, client):
        await _insert_entry("a", audio_ext=".mp3")
        (db_module.AUDIO_DIR / "a.mp3").write_bytes(b"mp3-bytes")
        r = client.get("/archive/a/audio")
        assert r.headers["content-type"] == "audio/mpeg"

    async def test_returns_file_with_correct_mime_wav(self, client):
        await _insert_entry("a", audio_ext=".wav")
        (db_module.AUDIO_DIR / "a.wav").write_bytes(b"wav")
        r = client.get("/archive/a/audio")
        assert r.headers["content-type"] == "audio/wav"

    async def test_returns_404_for_missing_entry(self, client):
        r = client.get("/archive/missing-id/audio")
        assert r.status_code == 404


# ── /archive/{id}/download/{fmt} ──────────────────────────────────────────────

class TestArchiveDownload:
    async def test_download_txt(self, client):
        await _insert_entry("a")
        r = client.get("/archive/a/download/txt")
        assert r.status_code == 200
        assert "SPEAKER_00" in r.text

    async def test_download_json(self, client):
        await _insert_entry("a")
        r = client.get("/archive/a/download/json")
        body = r.json()
        assert "speakers" in body
        assert "segments" in body

    async def test_download_srt_has_timestamps(self, client):
        await _insert_entry("a")
        r = client.get("/archive/a/download/srt")
        assert "-->" in r.text

    async def test_download_markdown_has_bold_speaker(self, client):
        await _insert_entry("a")
        r = client.get("/archive/a/download/md")
        assert "**SPEAKER_00**" in r.text

    async def test_download_unknown_format_returns_400(self, client):
        await _insert_entry("a")
        r = client.get("/archive/a/download/xml")
        assert r.status_code == 400

    async def test_download_missing_entry_returns_404(self, client):
        r = client.get("/archive/missing/download/txt")
        assert r.status_code == 404


# ── PATCH /archive/{id}/rename and /names ────────────────────────────────────

class TestArchiveMutations:
    async def test_rename_updates_filename(self, client):
        await _insert_entry("a", filename="old.mp3")
        r = client.patch("/archive/a/rename", json={"filename": "new.mp3"})
        assert r.status_code == 200
        entry = await db_module.get_archive_entry("a")
        assert entry["filename"] == "new.mp3"

    async def test_rename_missing_returns_404(self, client):
        r = client.patch("/archive/missing/rename", json={"filename": "x"})
        assert r.status_code == 404

    async def test_save_speaker_names_persists(self, client):
        await _insert_entry("a")
        r = client.patch("/archive/a/names", json={"SPEAKER_00": "Mateo"})
        assert r.status_code == 200
        entry = await db_module.get_archive_entry("a")
        assert entry["speaker_names"] == {"SPEAKER_00": "Mateo"}

    async def test_save_speaker_names_missing_returns_404(self, client):
        r = client.patch("/archive/missing/names", json={"x": "y"})
        assert r.status_code == 404


# ── DELETE /archive/{id} ──────────────────────────────────────────────────────

class TestArchiveDelete:
    async def test_delete_removes_entry(self, client):
        await _insert_entry("a")
        r = client.delete("/archive/a")
        assert r.status_code == 200
        assert await db_module.get_archive_entry("a") is None

    async def test_delete_also_removes_audio_file(self, client):
        await _insert_entry("a", audio_ext=".mp3")
        audio = db_module.AUDIO_DIR / "a.mp3"
        audio.write_bytes(b"data")
        r = client.delete("/archive/a")
        assert r.status_code == 200
        assert not audio.exists()

    async def test_delete_missing_returns_404(self, client):
        r = client.delete("/archive/missing")
        assert r.status_code == 404


# ── /settings ─────────────────────────────────────────────────────────────────

class TestSettings:
    async def test_get_returns_defaults_when_empty(self, client):
        r = client.get("/settings")
        body = r.json()
        assert body["provider"] == "lmstudio"
        assert body["base_url"] == "http://localhost:1234"
        assert body["api_key_set"] is False
        assert "api_key" not in body  # never expose the raw key

    async def test_get_reports_api_key_set_true_when_saved(self, client):
        await db_module.save_settings("anthropic", "", "claude-sonnet-4-6", "sk-abc")
        r = client.get("/settings")
        body = r.json()
        assert body["api_key_set"] is True
        assert "api_key" not in body  # mask, do not return the actual key

    async def test_post_settings_roundtrip(self, client):
        r = client.post("/settings", json={
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model":    "llama3.2:3b",
            "api_key":  "",
        })
        assert r.status_code == 200
        body = client.get("/settings").json()
        assert body["provider"] == "ollama"
        assert body["model"]    == "llama3.2:3b"

    async def test_post_settings_ignores_unwhitelisted_fields(self, client):
        """Arbitrary keys should not be persisted — protects against accidents."""
        r = client.post("/settings", json={
            "provider":   "lmstudio",
            "base_url":   "http://localhost:1234",
            "model":      "",
            "api_key":    "",
            "DROP_TABLE": "archive",  # ignored
            "evil_field": "value",
        })
        assert r.status_code == 200
        # No crash, settings saved correctly
        body = client.get("/settings").json()
        assert body["provider"] == "lmstudio"

    async def test_test_endpoint_uses_saved_key_when_blank(self, client, monkeypatch):
        """If the client sends an empty api_key, the test should fall back to the saved one."""
        await db_module.save_settings("anthropic", "", "claude-sonnet-4-6", "sk-stored")
        captured = {}

        def fake_get_summarizer(cfg):
            captured["cfg"] = cfg
            class S:
                async def check_health(self): return True
            return S()
        monkeypatch.setattr(main_module, "get_summarizer", fake_get_summarizer)

        r = client.post("/settings/test", json={"provider": "anthropic"})  # no api_key
        assert r.status_code == 200
        assert captured["cfg"]["api_key"] == "sk-stored"

    async def test_test_endpoint_returns_ok_when_provider_healthy(self, client, monkeypatch):
        class S:
            async def check_health(self): return True
        monkeypatch.setattr(main_module, "get_summarizer", lambda cfg: S())
        r = client.post("/settings/test", json={"provider": "lmstudio", "base_url": "http://localhost:1234"})
        body = r.json()
        assert body["ok"] is True
        assert body["message"] == "Connected"

    async def test_test_endpoint_returns_unreachable_on_false_health(self, client, monkeypatch):
        class S:
            async def check_health(self): return False
        monkeypatch.setattr(main_module, "get_summarizer", lambda cfg: S())
        r = client.post("/settings/test", json={"provider": "lmstudio"})
        body = r.json()
        assert body["ok"] is False
        assert "Not reachable" in body["message"]

    async def test_test_endpoint_returns_error_on_provider_exception(self, client, monkeypatch):
        def raiser(cfg):
            raise ProviderUnavailableError("server is down", provider="lmstudio")
        monkeypatch.setattr(main_module, "get_summarizer", raiser)
        r = client.post("/settings/test", json={"provider": "lmstudio"})
        body = r.json()
        assert body["ok"] is False
        assert "server is down" in body["message"]


# ── /models ───────────────────────────────────────────────────────────────────

class TestModels:
    def test_anthropic_models_hardcoded(self, client):
        r = client.post("/models", json={"provider": "anthropic"})
        models = r.json()["models"]
        assert len(models) > 0
        assert any("claude" in m for m in models)

    def test_openai_models_hardcoded(self, client):
        r = client.post("/models", json={"provider": "openai"})
        models = r.json()["models"]
        assert any("gpt" in m for m in models)

    def test_gemini_models_hardcoded(self, client):
        r = client.post("/models", json={"provider": "gemini"})
        models = r.json()["models"]
        assert any("gemini" in m for m in models)

    @respx.mock
    def test_lmstudio_fetches_live_models(self, client):
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [{"id": "qwen3-vl"}, {"id": "gemma-3-4b"}],
            })
        )
        r = client.post("/models", json={"provider": "lmstudio", "base_url": "http://localhost:1234"})
        body = r.json()
        assert body["models"] == ["qwen3-vl", "gemma-3-4b"]
        assert "error" not in body

    @respx.mock
    def test_lmstudio_unreachable_returns_error_message(self, client):
        respx.get("http://localhost:1234/v1/models").mock(side_effect=httpx.ConnectError("nope"))
        r = client.post("/models", json={"provider": "lmstudio", "base_url": "http://localhost:1234"})
        body = r.json()
        assert body["models"] == []
        assert "Cannot reach LM Studio" in body["error"]

    @respx.mock
    def test_ollama_fetches_live_models(self, client):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}],
            })
        )
        r = client.post("/models", json={"provider": "ollama", "base_url": "http://localhost:11434"})
        assert r.json()["models"] == ["llama3.2:3b", "qwen2.5:7b"]

    @respx.mock
    def test_ollama_unreachable_returns_error_message(self, client):
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("nope"))
        r = client.post("/models", json={"provider": "ollama", "base_url": "http://localhost:11434"})
        assert "Cannot reach Ollama" in r.json()["error"]

    def test_unknown_provider_returns_empty_list(self, client):
        r = client.post("/models", json={"provider": "made-up"})
        assert r.json() == {"models": []}


# ── /summarize/{id} ──────────────────────────────────────────────────────────

class FakeSummarizer:
    def __init__(self, tokens=None, raises=None):
        self._tokens = tokens or []
        self._raises = raises

    def provider_name(self) -> str:
        return "fake/test"

    async def stream_summarize(self, text: str) -> AsyncGenerator[str, None]:
        if self._raises is not None:
            raise self._raises
        for t in self._tokens:
            yield t


def _parse_sse(text: str):
    """Yield parsed `data:` event payloads from an SSE response body."""
    for line in text.splitlines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


class TestSummarize:
    async def test_summarize_missing_entry_returns_404(self, client, monkeypatch):
        monkeypatch.setattr(main_module, "get_summarizer", lambda cfg: FakeSummarizer())
        r = client.get("/summarize/missing")
        assert r.status_code == 404

    async def test_summarize_streams_tokens_and_done(self, client, monkeypatch):
        await _insert_entry("a")
        monkeypatch.setattr(
            main_module, "get_summarizer",
            lambda cfg: FakeSummarizer(tokens=["Hello", " world"]),
        )
        r = client.get("/summarize/a")
        events = list(_parse_sse(r.text))
        kinds = [e["type"] for e in events]
        assert kinds == ["token", "token", "done"]
        assert events[-1]["summary"] == "Hello world"

    async def test_summarize_persists_summary_to_db(self, client, monkeypatch):
        await _insert_entry("a")
        monkeypatch.setattr(
            main_module, "get_summarizer",
            lambda cfg: FakeSummarizer(tokens=["A ", "B"]),
        )
        client.get("/summarize/a").text  # drain
        entry = await db_module.get_archive_entry("a")
        assert entry["summary"] == "A B"

    async def test_summarize_provider_error_emits_error_event(self, client, monkeypatch):
        await _insert_entry("a")
        err = ProviderUnavailableError("server is sleeping", provider="lmstudio")
        monkeypatch.setattr(main_module, "get_summarizer", lambda cfg: FakeSummarizer(raises=err))
        r = client.get("/summarize/a")
        events = list(_parse_sse(r.text))
        assert events[-1]["type"] == "error"
        assert "server is sleeping" in events[-1]["message"]

    async def test_summarize_invalid_provider_config_returns_400(self, client, monkeypatch):
        await _insert_entry("a")
        def raiser(cfg):
            raise ValueError("Unknown provider: 'banana'")
        monkeypatch.setattr(main_module, "get_summarizer", raiser)
        r = client.get("/summarize/a")
        assert r.status_code == 400


# ── /summary/{id} ─────────────────────────────────────────────────────────────

class TestGetSummary:
    async def test_returns_saved_summary(self, client):
        await _insert_entry("a")
        await db_module.save_summary("a", "stored summary")
        r = client.get("/summary/a")
        assert r.json() == {"summary": "stored summary"}

    async def test_returns_null_when_no_summary(self, client):
        await _insert_entry("a")
        r = client.get("/summary/a")
        assert r.json() == {"summary": None}

    async def test_returns_404_for_missing_entry(self, client):
        r = client.get("/summary/missing")
        assert r.status_code == 404


# ── Lifespan + sanity ─────────────────────────────────────────────────────────

class TestLifespan:
    def test_lifespan_initializes_db(self, client):
        """Just hitting the client invokes the lifespan which calls init_db.
        If that fails, the test client setup itself would error."""
        r = client.get("/archive")
        assert r.status_code == 200
