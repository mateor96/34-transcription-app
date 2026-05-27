"""Tests for app.diarize._drop_phantom_speakers — the phantom-speaker filter
that fixes auto speaker over-counting without forcing a speaker count."""
from __future__ import annotations

from app.diarize import _drop_phantom_speakers


def test_folds_small_share_speaker_into_nearest():
    # The real failure: one dominant speaker, one real second speaker, and a
    # tiny phantom cluster (a few seconds). The phantom must be reassigned.
    turns = [
        {"start": 0.0,    "end": 900.0,  "speaker": "A"},        # 900s dominant
        {"start": 900.0,  "end": 905.0,  "speaker": "PHANTOM"},  # 5s (<5%)
        {"start": 905.0,  "end": 1120.0, "speaker": "B"},        # 215s (~19%)
    ]
    out = _drop_phantom_speakers(turns)
    speakers = {t["speaker"] for t in out}
    assert speakers == {"A", "B"}
    assert len(out) == len(turns)            # turns kept, only relabeled
    assert "PHANTOM" not in speakers


def test_noop_when_all_speakers_substantial():
    turns = [
        {"start": 0.0,   "end": 100.0, "speaker": "A"},
        {"start": 100.0, "end": 200.0, "speaker": "B"},
    ]
    assert _drop_phantom_speakers(turns) == turns


def test_noop_for_single_speaker():
    turns = [
        {"start": 0.0,  "end": 10.0, "speaker": "A"},
        {"start": 10.0, "end": 12.0, "speaker": "A"},
    ]
    assert _drop_phantom_speakers(turns) == turns


def test_two_real_speakers_kept_one_phantom_dropped():
    # Mirrors the observed sweep: 928 + 220 real, 30 + 5 phantom -> 2 speakers.
    turns = [
        {"start": 0.0,    "end": 928.0,  "speaker": "S1"},
        {"start": 928.0,  "end": 1148.0, "speaker": "S3"},
        {"start": 1148.0, "end": 1178.0, "speaker": "S0"},   # 30s, ~2.5%
        {"start": 1178.0, "end": 1183.0, "speaker": "S2"},   # 5s
    ]
    out = _drop_phantom_speakers(turns)
    assert {t["speaker"] for t in out} == {"S1", "S3"}
