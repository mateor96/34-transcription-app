"""Tests for app.db — schema, migrations, CRUD, and audio-file cascade."""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from app import db as db_module
from app.db import (
    delete_archive_entry,
    get_archive_entry,
    get_settings,
    init_db,
    list_archive,
    save_settings,
    save_summary,
    save_transcription,
    update_filename,
    update_speaker_names,
)

from .conftest import SAMPLE_SEGMENTS


# ── init_db ──────────────────────────────────────────────────────────────────

async def test_init_db_creates_archive_and_settings_tables(storage: Path):
    await init_db()
    async with aiosqlite.connect(db_module.DB_PATH) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
            tables = {row[0] for row in await cur.fetchall()}
    assert "archive" in tables
    assert "settings" in tables


async def test_init_db_creates_audio_dir(storage: Path):
    # Remove the audio dir created by the fixture so we can verify init_db creates it
    db_module.AUDIO_DIR.rmdir()
    assert not db_module.AUDIO_DIR.exists()
    await init_db()
    assert db_module.AUDIO_DIR.exists()


async def test_init_db_idempotent(storage: Path):
    await init_db()
    await init_db()  # second call must not error
    cfg = await get_settings()
    assert cfg["provider"] == "lmstudio"


async def test_init_db_migrates_legacy_schema(storage: Path):
    """An archive table missing the new columns should pick them up."""
    db_module.DB_PATH.parent.mkdir(exist_ok=True)
    async with aiosqlite.connect(db_module.DB_PATH) as db:
        await db.execute(
            """CREATE TABLE archive (
                id            TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                duration_s    REAL,
                speaker_count INTEGER,
                result_json   TEXT NOT NULL
            )"""
        )
        await db.commit()

    await init_db()

    async with aiosqlite.connect(db_module.DB_PATH) as db:
        async with db.execute("PRAGMA table_info(archive)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
    assert {"speaker_names", "summary", "audio_ext"}.issubset(cols)


# ── save_transcription / get_archive_entry ───────────────────────────────────

async def test_save_transcription_round_trip(initialized_db):
    await save_transcription("job1", "meeting.mp3", SAMPLE_SEGMENTS, audio_ext=".mp3")
    entry = await get_archive_entry("job1")
    assert entry is not None
    assert entry["filename"]    == "meeting.mp3"
    assert entry["duration_s"]  == 9.5
    assert entry["speaker_count"] == 2
    assert entry["audio_ext"]   == ".mp3"
    assert entry["segments"]    == SAMPLE_SEGMENTS
    assert entry["speaker_names"] == {}


async def test_save_transcription_without_audio_ext(initialized_db):
    await save_transcription("job1", "meeting.mp3", SAMPLE_SEGMENTS)
    entry = await get_archive_entry("job1")
    assert entry["audio_ext"] is None


async def test_save_transcription_empty_segments(initialized_db):
    await save_transcription("job1", "empty.mp3", [])
    entry = await get_archive_entry("job1")
    assert entry["duration_s"] == 0
    assert entry["speaker_count"] == 0
    assert entry["segments"] == []


async def test_get_archive_entry_missing(initialized_db):
    assert await get_archive_entry("nonexistent") is None


# ── list_archive ─────────────────────────────────────────────────────────────

async def test_list_archive_empty(initialized_db):
    assert await list_archive() == []


async def test_list_archive_returns_entries(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    await save_transcription("b", "b.mp3", SAMPLE_SEGMENTS)
    entries = await list_archive()
    assert len(entries) == 2
    ids = {e["id"] for e in entries}
    assert ids == {"a", "b"}


async def test_list_archive_ordered_newest_first(initialized_db):
    """SQLite sorts by created_at desc; verify by inserting with explicit timestamps."""
    async with aiosqlite.connect(db_module.DB_PATH) as db:
        await db.execute(
            "INSERT INTO archive (id, filename, created_at, duration_s, speaker_count, result_json, speaker_names) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old", "old.mp3", "2024-01-01T00:00:00", 1.0, 1, "[]", "{}"),
        )
        await db.execute(
            "INSERT INTO archive (id, filename, created_at, duration_s, speaker_count, result_json, speaker_names) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("new", "new.mp3", "2026-01-01T00:00:00", 1.0, 1, "[]", "{}"),
        )
        await db.commit()
    entries = await list_archive()
    assert [e["id"] for e in entries] == ["new", "old"]


async def test_list_archive_has_summary_flag_false_initially(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    entries = await list_archive()
    assert entries[0]["has_summary"] == 0


async def test_list_archive_has_summary_flag_true_after_save_summary(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    await save_summary("a", "Some summary")
    entries = await list_archive()
    assert entries[0]["has_summary"] == 1


async def test_list_archive_has_summary_false_when_empty_string(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    await save_summary("a", "")
    entries = await list_archive()
    assert entries[0]["has_summary"] == 0


# ── update_speaker_names ─────────────────────────────────────────────────────

async def test_update_speaker_names_persists(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    ok = await update_speaker_names("a", {"SPEAKER_00": "Mateo", "SPEAKER_01": "Alice"})
    assert ok is True
    entry = await get_archive_entry("a")
    assert entry["speaker_names"] == {"SPEAKER_00": "Mateo", "SPEAKER_01": "Alice"}


async def test_update_speaker_names_missing_entry_returns_false(initialized_db):
    assert await update_speaker_names("nope", {"x": "y"}) is False


# ── update_filename ──────────────────────────────────────────────────────────

async def test_update_filename_persists(initialized_db):
    await save_transcription("a", "old.mp3", SAMPLE_SEGMENTS)
    ok = await update_filename("a", "new name.mp3")
    assert ok is True
    entry = await get_archive_entry("a")
    assert entry["filename"] == "new name.mp3"


async def test_update_filename_missing_entry_returns_false(initialized_db):
    assert await update_filename("nope", "x") is False


# ── settings CRUD ────────────────────────────────────────────────────────────

async def test_get_settings_defaults_when_table_empty(initialized_db):
    cfg = await get_settings()
    assert cfg == {
        "provider":      "lmstudio",
        "base_url":      "http://localhost:1234",
        "model":         "",
        "api_key":       "",
        "prompt_style":  "meeting",
        "custom_prompt": "",
    }


async def test_save_settings_creates_row(initialized_db):
    await save_settings("ollama", "http://localhost:11434", "llama3:8b", "")
    cfg = await get_settings()
    assert cfg["provider"] == "ollama"
    assert cfg["base_url"] == "http://localhost:11434"
    assert cfg["model"]    == "llama3:8b"


async def test_save_settings_replaces_existing_row(initialized_db):
    await save_settings("lmstudio", "http://localhost:1234", "", "")
    await save_settings("anthropic", "", "claude-sonnet-4-6", "sk-abc123")
    cfg = await get_settings()
    assert cfg["provider"] == "anthropic"
    assert cfg["model"]    == "claude-sonnet-4-6"
    assert cfg["api_key"]  == "sk-abc123"


async def test_save_settings_only_one_row_exists(initialized_db):
    await save_settings("ollama", "url1", "m1", "")
    await save_settings("openai", "", "m2", "key")
    await save_settings("gemini", "", "m3", "key")
    async with aiosqlite.connect(db_module.DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM settings") as cur:
            row = await cur.fetchone()
    assert row[0] == 1


# ── save_summary ─────────────────────────────────────────────────────────────

async def test_save_summary_updates_existing_entry(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    ok = await save_summary("a", "- Topic\n- Decision")
    assert ok is True
    entry = await get_archive_entry("a")
    assert entry["summary"] == "- Topic\n- Decision"


async def test_save_summary_missing_entry_returns_false(initialized_db):
    assert await save_summary("nope", "anything") is False


async def test_save_summary_overwrites(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    await save_summary("a", "first")
    await save_summary("a", "second")
    entry = await get_archive_entry("a")
    assert entry["summary"] == "second"


# ── delete_archive_entry (with audio cascade) ─────────────────────────────────

async def test_delete_archive_entry_removes_row(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)
    ok = await delete_archive_entry("a")
    assert ok is True
    assert await get_archive_entry("a") is None


async def test_delete_archive_entry_removes_audio_file(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS, audio_ext=".mp3")
    audio_path = db_module.AUDIO_DIR / "a.mp3"
    audio_path.write_bytes(b"fake audio")
    assert audio_path.exists()
    await delete_archive_entry("a")
    assert not audio_path.exists()


async def test_delete_archive_entry_no_audio_does_not_fail(initialized_db):
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS)  # no audio_ext
    ok = await delete_archive_entry("a")
    assert ok is True


async def test_delete_archive_entry_audio_missing_on_disk_does_not_fail(initialized_db):
    """audio_ext is set but the file was already deleted externally."""
    await save_transcription("a", "a.mp3", SAMPLE_SEGMENTS, audio_ext=".mp3")
    # Do NOT create the audio file
    ok = await delete_archive_entry("a")
    assert ok is True


async def test_delete_archive_entry_missing_returns_false(initialized_db):
    assert await delete_archive_entry("nope") is False
