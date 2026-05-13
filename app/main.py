import asyncio
import json
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .db import delete_archive_entry, get_archive_entry, init_db, list_archive, update_speaker_names
from .export import to_json, to_markdown, to_srt, to_txt
from .pipeline import run_pipeline


@asynccontextmanager
async def lifespan(app):
    await init_db()
    yield


app = FastAPI(title="Transcription App", lifespan=lifespan)

# In-memory job store: job_id → {status, queue, result}
jobs: dict = {}


@app.post("/transcribe")
async def start_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Query(default=None),
    max_speakers: Optional[int] = Query(default=None),
):
    job_id = str(uuid.uuid4())

    suffix = Path(file.filename).suffix or ".audio"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    jobs[job_id] = {"status": "queued", "queue": asyncio.Queue(), "result": None}
    background_tasks.add_task(
        run_pipeline, job_id, tmp.name, jobs, min_speakers, max_speakers, file.filename
    )

    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    async def stream():
        q = job["queue"]
        while True:
            event = await q.get()
            if event is None:
                break
            yield {"data": json.dumps(event)}

    return EventSourceResponse(stream())


@app.get("/result/{job_id}/{fmt}")
async def get_result(job_id: str, fmt: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "result not ready"}, status_code=404)

    segments = job["result"]
    return _format_response(segments, fmt)


# ── Archive ──────────────────────────────────────────────────────────────────

@app.get("/archive")
async def archive_list():
    return await list_archive()


@app.get("/archive/{entry_id}")
async def archive_get(entry_id: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    return entry


@app.get("/archive/{entry_id}/download/{fmt}")
async def archive_download(entry_id: str, fmt: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _format_response(entry["segments"], fmt)


@app.patch("/archive/{entry_id}/names")
async def archive_save_names(entry_id: str, names: dict):
    ok = await update_speaker_names(entry_id, names)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": True}


@app.delete("/archive/{entry_id}")
async def archive_delete(entry_id: str):
    ok = await delete_archive_entry(entry_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_response(segments: list, fmt: str):
    if fmt == "json":
        return JSONResponse(to_json(segments))
    if fmt == "txt":
        return PlainTextResponse(to_txt(segments))
    if fmt == "srt":
        return PlainTextResponse(to_srt(segments))
    if fmt == "md":
        return PlainTextResponse(to_markdown(segments))
    return JSONResponse({"error": f"unknown format: {fmt}"}, status_code=400)


# Static files last so they don't shadow the API routes
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
