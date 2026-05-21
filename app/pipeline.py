import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ffmpeg

from .transcribe import transcribe
from .diarize import diarize
from .merge import merge
from .db import AUDIO_DIR, save_transcription

_executor = ThreadPoolExecutor(max_workers=4)


async def run_pipeline(
    job_id: str,
    audio_path: str,
    jobs: dict,
    min_speakers: int | None,
    max_speakers: int | None,
    filename: str = "audio",
) -> None:
    job = jobs[job_id]
    loop = asyncio.get_running_loop()

    async def emit(event: dict) -> None:
        await job["queue"].put(event)

    wav_path = audio_path + ".wav"

    try:
        job["status"] = "processing"

        # Normalize to 16 kHz mono WAV
        await emit({"stage": "normalizing", "pct": 5, "message": "Normalizing audio..."})
        await loop.run_in_executor(
            _executor,
            lambda: (
                ffmpeg.input(audio_path)
                .output(wav_path, ar=16000, ac=1)
                .run(quiet=True, overwrite_output=True)
            ),
        )

        await emit({"stage": "processing", "pct": 10, "message": "Starting transcription and diarization..."})

        # Whisper writes its 0..1 progress here from a worker thread; the poll
        # loop below reads it and forwards real progress to the SSE stream.
        job["whisper_progress"] = 0.0
        def _on_whisper_progress(frac: float) -> None:
            job["whisper_progress"] = frac

        transcribe_future = loop.run_in_executor(
            _executor, transcribe, wav_path, _on_whisper_progress,
        )
        diarize_future = loop.run_in_executor(_executor, diarize, wav_path, min_speakers, max_speakers)

        # Poll often enough that the bar feels live without flooding the SSE stream.
        pending = {transcribe_future, diarize_future}
        last_pct = 10
        while pending:
            done, pending = await asyncio.wait(pending, timeout=1.5)
            if not pending:
                break
            whisper_done = transcribe_future not in pending
            # 10..80% reserved for transcription; jump to 80 once it's done and
            # only diarization remains.
            pct = 80 if whisper_done else 10 + int(job["whisper_progress"] * 70)
            message = "Identifying speakers..." if whisper_done else "Transcribing..."
            if pct != last_pct:
                await emit({"stage": "processing", "pct": pct, "message": message})
                last_pct = pct

        whisper_result = await transcribe_future
        diarization_turns = await diarize_future

        await emit({"stage": "merging", "pct": 90, "message": "Merging transcription and speakers..."})
        segments = await loop.run_in_executor(_executor, merge, whisper_result, diarization_turns)

        job["result"] = segments
        job["status"] = "done"

        # Persist the original audio alongside the transcript so click-to-seek
        # works on archived entries.
        audio_ext = Path(filename).suffix or Path(audio_path).suffix or ".audio"
        persisted_audio = AUDIO_DIR / f"{job_id}{audio_ext}"
        try:
            shutil.move(audio_path, persisted_audio)
        except OSError:
            audio_ext = None  # move failed — record entry without audio

        await save_transcription(job_id, filename, segments, audio_ext)
        await emit({"stage": "done", "pct": 100, "message": "Done!"})

    except Exception as exc:
        job["status"] = "error"
        await emit({"stage": "error", "message": str(exc)})

    finally:
        # WAV is always temporary; audio_path may have been moved to AUDIO_DIR.
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        await job["queue"].put(None)  # sentinel — tells SSE stream to close
