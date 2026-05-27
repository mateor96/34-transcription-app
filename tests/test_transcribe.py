"""Tests for app.transcribe pure helpers: VAD window construction and the
per-segment confidence filter. The model call itself is exercised end-to-end,
not here."""
from __future__ import annotations

from app.transcribe import _build_windows, _keep_segment


def test_build_windows_merges_close_regions():
    # gap 0.3s (< 0.5 merge threshold) -> single padded window
    windows = _build_windows([(1.0, 2.0), (2.3, 3.0)], duration=10.0)
    assert len(windows) == 1
    assert windows[0] == (0.8, 3.2)        # 0.2s padding each side


def test_build_windows_keeps_distant_regions_separate():
    windows = _build_windows([(1.0, 2.0), (5.0, 6.0)], duration=10.0)  # gap 3s
    assert len(windows) == 2


def test_build_windows_splits_when_exceeding_max_window():
    # contiguous regions whose merged span would exceed ~28s must split
    windows = _build_windows([(0.0, 15.0), (15.3, 30.0)], duration=30.0)
    assert len(windows) == 2


def test_build_windows_clamps_to_duration():
    windows = _build_windows([(0.0, 5.0)], duration=5.0)
    assert windows[0][0] == 0.0            # padding can't go below 0
    assert windows[0][1] == 5.0            # ...or beyond duration


def test_keep_segment_drops_silence():
    assert not _keep_segment({"text": "hallo", "no_speech_prob": 0.9})


def test_keep_segment_drops_repetitive():
    assert not _keep_segment({"text": "x" * 80, "compression_ratio": 3.0})


def test_keep_segment_drops_empty_text():
    assert not _keep_segment({"text": "   ", "no_speech_prob": 0.0})


def test_keep_segment_keeps_normal_speech():
    assert _keep_segment({"text": "hallo welt", "no_speech_prob": 0.1, "compression_ratio": 1.4})
