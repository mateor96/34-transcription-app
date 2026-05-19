"""Shared fixtures for the test suite.

The application stores its DB and audio files under fixed paths derived from
the user's home directory. Tests must redirect both to a temp directory so
they never touch the real archive. Because both `app.db` and `app.main` hold
module-level references to these constants, monkeypatching must update both.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import db as db_module
from app import main as main_module
from app import pipeline as pipeline_module


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect DB_PATH and AUDIO_DIR to a tmp directory.

    Every module that imports AUDIO_DIR at module-load time holds its own
    reference that must be patched separately.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    monkeypatch.setattr(db_module,       "DB_PATH",   tmp_path / "test.db")
    monkeypatch.setattr(db_module,       "AUDIO_DIR", audio_dir)
    monkeypatch.setattr(main_module,     "AUDIO_DIR", audio_dir)
    monkeypatch.setattr(pipeline_module, "AUDIO_DIR", audio_dir)
    return tmp_path


@pytest.fixture
async def initialized_db(storage: Path) -> Path:
    """Storage with init_db() already run — tables present, ready for inserts."""
    await db_module.init_db()
    return storage


@pytest.fixture
def client(storage: Path):
    """FastAPI TestClient with isolated storage; the app's lifespan runs init_db."""
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as c:
        yield c


SAMPLE_SEGMENTS = [
    {
        "speaker": "SPEAKER_00",
        "start":   0.5,
        "end":     2.8,
        "text":    "Hello there.",
        "words":   [
            {"word": "Hello", "start": 0.5, "end": 1.0},
            {"word": "there", "start": 1.1, "end": 2.8},
        ],
    },
    {
        "speaker": "SPEAKER_01",
        "start":   3.0,
        "end":     6.4,
        "text":    "Hi, how are you?",
        "words":   [],
    },
    {
        "speaker": "SPEAKER_00",
        "start":   7.0,
        "end":     9.5,
        "text":    "Doing well.",
        "words":   [],
    },
]
