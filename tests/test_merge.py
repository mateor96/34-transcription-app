"""Tests for app.merge — assigning speakers to whisper words and grouping into segments."""
from __future__ import annotations

from app.merge import merge


def _whisper(words):
    return {"segments": [{"words": words}]}


def test_merge_empty_input_returns_empty_list():
    assert merge({"segments": []}, []) == []


def test_merge_no_words_returns_empty_list():
    assert merge(_whisper([]), [{"start": 0, "end": 10, "speaker": "SPEAKER_00"}]) == []


def test_merge_assigns_speaker_via_word_midpoint():
    words = [
        {"word": "Hello",  "start": 0.0, "end": 1.0},  # midpoint 0.5 → in SPEAKER_00 turn
        {"word": " world", "start": 5.0, "end": 6.0},  # midpoint 5.5 → in SPEAKER_01 turn
    ]
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 7.0, "speaker": "SPEAKER_01"},
    ]
    out = merge(_whisper(words), turns)
    assert len(out) == 2
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[0]["text"]    == "Hello"
    assert out[1]["speaker"] == "SPEAKER_01"
    assert out[1]["text"]    == "world"


def test_merge_uses_max_overlap_not_midpoint():
    """A word whose midpoint lands in S0's (short) turn but which overlaps S1
    more must be assigned to S1 — the case max-overlap fixes over midpoint."""
    words = [{"word": "x", "start": 2.0, "end": 3.0}]  # midpoint 2.5
    turns = [
        {"start": 2.4, "end": 2.6, "speaker": "S0"},   # contains midpoint, overlap 0.2
        {"start": 2.6, "end": 3.0, "speaker": "S1"},   # overlap 0.4
    ]
    out = merge(_whisper(words), turns)
    assert out[0]["speaker"] == "S1"


def test_merge_groups_consecutive_same_speaker():
    words = [
        {"word": "Hi",     "start": 0.0, "end": 0.5},
        {"word": " there", "start": 0.6, "end": 1.0},
        {"word": " you",   "start": 1.1, "end": 1.5},
    ]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    out = merge(_whisper(words), turns)
    assert len(out) == 1
    assert out[0]["text"]  == "Hi there you"
    assert out[0]["start"] == 0.0
    assert out[0]["end"]   == 1.5


def test_merge_segment_word_count():
    words = [{"word": "x", "start": 0, "end": 1}, {"word": "y", "start": 1, "end": 2}]
    turns = [{"start": 0, "end": 3, "speaker": "S0"}]
    out = merge(_whisper(words), turns)
    assert len(out[0]["words"]) == 2


def test_merge_word_in_gap_falls_back_to_nearest_turn():
    """Word midpoint at 2.5 falls between turns ending at 2.0 and starting at 3.0."""
    words = [{"word": "lone", "start": 2.0, "end": 3.0}]  # midpoint 2.5
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    out = merge(_whisper(words), turns)
    assert len(out) == 1
    # Either nearest will do — what matters is that we don't crash and we assign *some* speaker
    assert out[0]["speaker"] in {"SPEAKER_00", "SPEAKER_01"}


def test_merge_no_turns_falls_back_to_speaker_00():
    words = [{"word": "alone", "start": 1.0, "end": 2.0}]
    out = merge(_whisper(words), [])
    assert out[0]["speaker"] == "SPEAKER_00"


def test_merge_speaker_switches_create_new_segments():
    words = [
        {"word": "A", "start": 0.0, "end": 0.5},
        {"word": "B", "start": 1.0, "end": 1.5},
        {"word": "C", "start": 2.0, "end": 2.5},
        {"word": "D", "start": 3.0, "end": 3.5},
    ]
    turns = [
        {"start": 0.0, "end": 0.7, "speaker": "S0"},
        {"start": 0.8, "end": 1.7, "speaker": "S1"},
        {"start": 1.8, "end": 2.7, "speaker": "S0"},
        {"start": 2.8, "end": 3.7, "speaker": "S1"},
    ]
    out = merge(_whisper(words), turns)
    assert [s["speaker"] for s in out] == ["S0", "S1", "S0", "S1"]
