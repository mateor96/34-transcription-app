"""Tests for the in-memory job endpoints — POST /transcribe, GET /progress, GET /result.

These endpoints orchestrate around the `jobs` dict on the main module. The
heavy `run_pipeline` is patched out — we only verify the orchestration logic
(job creation, queue draining, format selection).
"""
from __future__ import annotations

import asyncio
import io
import json

import pytest

from app import main as main_module
from app.export import to_txt


@pytest.fixture(autouse=True)
def clear_jobs():
    """Reset the in-memory job dict before each test."""
    main_module.jobs.clear()
    yield
    main_module.jobs.clear()


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace run_pipeline with a no-op coroutine so /transcribe doesn't actually run."""
    called = {}

    async def fake_run_pipeline(job_id, audio_path, jobs, min_speakers, max_speakers, filename="audio"):
        called["args"] = (job_id, audio_path, min_speakers, max_speakers, filename)
        # Drain the queue so /progress can complete cleanly if called
        await jobs[job_id]["queue"].put({"stage": "done", "pct": 100})
        await jobs[job_id]["queue"].put(None)

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)
    return called


SAMPLE_SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "hi", "words": []},
]


# ── POST /transcribe ──────────────────────────────────────────────────────────

class TestTranscribePost:
    def test_returns_job_id(self, client, fake_pipeline):
        files = {"file": ("audio.mp3", io.BytesIO(b"fake-audio-bytes"), "audio/mpeg")}
        r = client.post("/transcribe", files=files)
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert isinstance(body["job_id"], str)
        assert len(body["job_id"]) > 0

    def test_creates_job_in_dict(self, client, fake_pipeline):
        files = {"file": ("audio.mp3", io.BytesIO(b"x"), "audio/mpeg")}
        r = client.post("/transcribe", files=files)
        job_id = r.json()["job_id"]
        assert job_id in main_module.jobs
        assert main_module.jobs[job_id]["status"] == "queued"

    def test_forwards_speaker_range_to_pipeline(self, client, fake_pipeline):
        files = {"file": ("a.mp3", io.BytesIO(b"x"), "audio/mpeg")}
        client.post("/transcribe?min_speakers=2&max_speakers=4", files=files)
        assert fake_pipeline["args"][2] == 2  # min
        assert fake_pipeline["args"][3] == 4  # max

    def test_forwards_filename_to_pipeline(self, client, fake_pipeline):
        files = {"file": ("Recording_2026.webm", io.BytesIO(b"x"), "audio/webm")}
        client.post("/transcribe", files=files)
        assert fake_pipeline["args"][4] == "Recording_2026.webm"


# ── GET /progress/{job_id} ────────────────────────────────────────────────────

class TestProgress:
    def test_missing_job_returns_404(self, client):
        r = client.get("/progress/missing")
        assert r.status_code == 404

    def test_streams_queued_events(self, client):
        # Seed a job manually with events already on its queue
        async def setup():
            q = asyncio.Queue()
            await q.put({"stage": "processing", "pct": 50})
            await q.put({"stage": "done", "pct": 100})
            await q.put(None)  # sentinel
            main_module.jobs["job1"] = {"status": "processing", "queue": q, "result": None}
        asyncio.get_event_loop().run_until_complete(setup())

        r = client.get("/progress/job1")
        events = []
        for line in r.text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        assert events[0]["stage"] == "processing"
        assert events[1]["stage"] == "done"


# ── GET /result/{job_id}/{fmt} ────────────────────────────────────────────────

class TestResult:
    def test_missing_job_returns_404(self, client):
        r = client.get("/result/missing/json")
        assert r.status_code == 404

    def test_unfinished_job_returns_404(self, client):
        main_module.jobs["job1"] = {"status": "processing", "queue": None, "result": None}
        r = client.get("/result/job1/json")
        assert r.status_code == 404

    def test_done_job_returns_json(self, client):
        main_module.jobs["job1"] = {"status": "done", "queue": None, "result": SAMPLE_SEGMENTS}
        r = client.get("/result/job1/json")
        assert r.status_code == 200
        body = r.json()
        assert body["segments"][0]["text"] == "hi"

    def test_done_job_returns_txt(self, client):
        main_module.jobs["job1"] = {"status": "done", "queue": None, "result": SAMPLE_SEGMENTS}
        r = client.get("/result/job1/txt")
        assert r.status_code == 200
        assert r.text == to_txt(SAMPLE_SEGMENTS)

    def test_done_job_unknown_format_returns_400(self, client):
        main_module.jobs["job1"] = {"status": "done", "queue": None, "result": SAMPLE_SEGMENTS}
        r = client.get("/result/job1/xml")
        assert r.status_code == 400
