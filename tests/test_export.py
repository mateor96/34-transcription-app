"""Tests for app.export — transcript format conversions."""
from __future__ import annotations

from app.export import to_json, to_markdown, to_srt, to_txt
from app.export import _srt_time, _ts


SEGS = [
    {"speaker": "SPEAKER_00", "start": 0.0,    "end": 2.5,  "text": "Hello",
     "words": [{"word": "Hello", "start": 0.0, "end": 2.5}]},
    {"speaker": "SPEAKER_01", "start": 3.0,    "end": 5.5,  "text": "Hi back"},
    {"speaker": "SPEAKER_00", "start": 65.123, "end": 70.0, "text": "Bye"},
]


# ── to_txt ───────────────────────────────────────────────────────────────────

def test_to_txt_joins_speaker_text_blocks():
    out = to_txt(SEGS)
    assert "SPEAKER_00: Hello" in out
    assert "SPEAKER_01: Hi back" in out
    assert out.count("\n\n") == 2  # 3 segments → 2 separators


def test_to_txt_empty():
    assert to_txt([]) == ""


# ── to_json ──────────────────────────────────────────────────────────────────

def test_to_json_includes_sorted_speakers():
    out = to_json(SEGS)
    assert out["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_to_json_includes_all_segments():
    out = to_json(SEGS)
    assert len(out["segments"]) == 3
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][0]["text"]    == "Hello"


def test_to_json_words_default_empty_list():
    out = to_json(SEGS)
    assert out["segments"][1]["words"] == []  # no "words" key on input


def test_to_json_empty():
    assert to_json([]) == {"speakers": [], "segments": []}


# ── to_srt ───────────────────────────────────────────────────────────────────

def test_to_srt_numbers_segments_from_1():
    out = to_srt(SEGS)
    lines = out.split("\n")
    assert lines[0] == "1"
    assert lines[4] == "2"
    assert lines[8] == "3"


def test_to_srt_uses_hh_mm_ss_comma_ms_format():
    out = to_srt(SEGS)
    assert "00:00:00,000 --> 00:00:02,500" in out
    assert "00:01:05,123 --> 00:01:10,000" in out


def test_srt_time_zero():
    assert _srt_time(0) == "00:00:00,000"


def test_srt_time_with_milliseconds():
    assert _srt_time(125.456) == "00:02:05,456"


def test_srt_time_over_hour():
    assert _srt_time(3661.5) == "01:01:01,500"


# ── to_markdown ──────────────────────────────────────────────────────────────

def test_to_markdown_uses_bold_speaker_and_timestamp():
    out = to_markdown(SEGS)
    assert "**SPEAKER_00** `00:00`: Hello" in out
    assert "**SPEAKER_00** `01:05`: Bye" in out


def test_ts_format():
    assert _ts(0)     == "00:00"
    assert _ts(65)    == "01:05"
    assert _ts(3600)  == "60:00"
