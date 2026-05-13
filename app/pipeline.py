import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import ffmpeg

from .transcribe import transcribe
from .diarize import diarize
from .merge import merge
from .db import save_transcription

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

        transcribe_future = loop.run_in_executor(_executor, transcribe, wav_path)
        diarize_future = loop.run_in_executor(_executor, diarize, wav_path, min_speakers, max_speakers)

        # Emit progress every 8 s while both ML tasks run
        pct = 10
        pending = {transcribe_future, diarize_future}
        while pending:
            done, pending = await asyncio.wait(pending, timeout=8)
            if pending:
                pct = min(pct + 12, 80)
                await emit({"stage": "processing", "pct": pct, "message": "Processing..."})

        whisper_result = await transcribe_future
        diarization_turns = await diarize_future

        await emit({"stage": "merging", "pct": 90, "message": "Merging transcription and speakers..."})
        segments = await loop.run_in_executor(_executor, merge, whisper_result, diarization_turns)

        job["result"] = segments
        job["status"] = "done"
        await save_transcription(job_id, filename, segments)
        await emit({"stage": "done", "pct": 100, "message": "Done!"})

    except Exception as exc:
        job["status"] = "error"
        await emit({"stage": "error", "message": str(exc)})

    finally:
        for path in (audio_path, wav_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        await job["queue"].put(None)  # sentinel — tells SSE stream to close
