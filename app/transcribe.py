"""mlx-whisper transcription with a progress callback.

mlx_whisper doesn't expose a progress hook, but it drives a `tqdm` progress
bar internally. We swap the `tqdm` reference inside `mlx_whisper.transcribe`
for a stand-in that forwards `update()` calls to our callback as a 0..1
fraction.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import mlx_whisper
import mlx_whisper.transcribe as _mw_transcribe

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Patching mlx_whisper.transcribe.tqdm is process-global. Serialize so a
# concurrent transcription can't see the wrong callback wired in.
_patch_lock = threading.Lock()


class _ProgressTqdm:
    """Minimal stand-in for tqdm.tqdm — forwards updates as a 0..1 fraction."""

    def __init__(self, total=None, callback: Optional[Callable[[float], None]] = None, **_kw):
        self.total = total or 1
        self.n = 0
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.callback:
            self.callback(1.0)

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.callback and self.total:
            self.callback(min(self.n / self.total, 1.0))

    def close(self) -> None:
        pass


def transcribe(
    audio_path: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    class _Wrapped(_ProgressTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, callback=progress_cb, **kwargs)

    class _FakeTqdmModule:
        tqdm = _Wrapped

    with _patch_lock:
        original = _mw_transcribe.tqdm
        _mw_transcribe.tqdm = _FakeTqdmModule
        try:
            return mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model,
                word_timestamps=True,
            )
        finally:
            _mw_transcribe.tqdm = original
