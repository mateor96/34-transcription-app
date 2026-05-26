"""VAD-gated mlx-whisper transcription.

Instead of handing Whisper the whole file (where it hallucinates on silence
and, on short noisy clips, even mis-detects the language — e.g. emitting
Chinese on German noise), we transcribe only the speech regions VAD found.

Approach (validated empirically; mlx-whisper's own `clip_timestamps` hangs, so
we slice the audio manually):
  1. Merge VAD regions into windows <= ~28s (Whisper's 30s receptive field),
     padded slightly so boundary words keep lead-in context.
  2. Detect the language once on the longest window, then pin it for every
     window — per-window auto-detect flip-flops and mis-fires on short clips.
  3. Transcribe each window with anti-hallucination decode params, offset the
     timestamps back to absolute time, and drop low-confidence/repetitive
     segments (residual noise that slipped through VAD).
"""
from __future__ import annotations

from typing import Callable, Optional

import mlx_whisper
from mlx_whisper.audio import SAMPLE_RATE, load_audio

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Window construction
_MAX_WINDOW_S = 28.0   # stay under Whisper's 30s receptive field
_PAD_S = 0.2           # lead-in/out context so boundary words aren't clipped
_MERGE_GAP_S = 0.5     # bridge regions separated by less than this

# Per-segment confidence gate — drops residual hallucination on noise that VAD
# let through. Conservative so real (quiet) speech is kept.
_MAX_NO_SPEECH_PROB = 0.85
_MAX_COMPRESSION_RATIO = 2.4   # high ratio == repetitive gibberish

# Decode params. condition_on_previous_text=False is the single most important
# anti-hallucination flag: it stops a hallucinated phrase from being fed back as
# the prompt and self-reinforcing into a loop.
_DECODE = dict(
    word_timestamps=True,
    condition_on_previous_text=False,
    hallucination_silence_threshold=2.0,
)


def _build_windows(regions: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Merge/pad VAD regions into <=_MAX_WINDOW_S transcription windows."""
    windows: list[tuple[float, float]] = []
    for start, end in regions:
        start = max(0.0, start - _PAD_S)
        end = min(duration, end + _PAD_S)
        if (
            windows
            and start - windows[-1][1] <= _MERGE_GAP_S
            and end - windows[-1][0] <= _MAX_WINDOW_S
        ):
            windows[-1] = (windows[-1][0], end)   # extend current window
        else:
            windows.append((start, end))
    return windows


def _keep_segment(seg: dict) -> bool:
    if seg.get("no_speech_prob", 0.0) > _MAX_NO_SPEECH_PROB:
        return False
    if seg.get("compression_ratio", 0.0) > _MAX_COMPRESSION_RATIO:
        return False
    return bool(seg.get("text", "").strip())


def transcribe(
    audio_path: str,
    speech_regions: Optional[list[tuple[float, float]]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> dict:
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    if speech_regions:
        windows = _build_windows(speech_regions, duration)
    else:
        windows = [(0.0, duration)]   # no VAD → whole file (fallback)

    if not windows:
        return {"segments": [], "language": language}

    common = dict(_DECODE)
    if initial_prompt:
        common["initial_prompt"] = initial_prompt

    def _run(window: tuple[float, float], lang: Optional[str]) -> dict:
        s, e = window
        chunk = audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)]
        return mlx_whisper.transcribe(
            chunk, path_or_hf_repo=model, language=lang, **common
        )

    # Pin language once, from the longest window (most likely real speech).
    if language is None:
        longest = max(range(len(windows)), key=lambda i: windows[i][1] - windows[i][0])
        first = _run(windows[longest], None)
        language = first.get("language")
        cache = {longest: first}
    else:
        cache = {}

    all_segments: list[dict] = []
    for i, window in enumerate(windows):
        result = cache.get(i) or _run(window, language)
        offset = window[0]
        for seg in result.get("segments", []):
            if not _keep_segment(seg):
                continue
            seg["start"] += offset
            seg["end"] += offset
            for w in seg.get("words", []):
                w["start"] += offset
                w["end"] += offset
            all_segments.append(seg)
        if progress_cb:
            progress_cb((i + 1) / len(windows))

    all_segments.sort(key=lambda s: s["start"])
    return {"segments": all_segments, "language": language}
