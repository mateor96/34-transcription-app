"""Tests for app.eval — the reference-free transcript quality scorer."""
from __future__ import annotations

from app.eval import flags, score_transcript


def _seg(speaker, start, end, text, words=None):
    return {"speaker": speaker, "start": start, "end": end, "text": text,
            "words": words if words is not None else [{"word": text, "start": start, "end": end}]}


def test_clean_transcript_is_healthy():
    segs = [
        _seg("S0", 0.0, 2.0, "Hi, wie geht es dir?"),
        _seg("S1", 2.0, 4.0, "Ganz gut, und selbst?"),
        _seg("S0", 4.0, 6.0, "Auch gut, danke."),
    ]
    score = score_transcript(segs)
    assert score["n_speakers"] == 2
    assert score["n_thin_speakers"] == 0
    assert score["max_repeat_run"] == 1
    assert flags(score) == []


def test_repetition_loop_is_flagged():
    segs = [_seg("S0", float(i), float(i) + 1, "Thank you.") for i in range(8)]
    score = score_transcript(segs)
    assert score["max_repeat_run"] == 8
    assert score["filler_segment_ratio"] == 1.0   # every segment is pure filler
    assert "repeat-loop" in flags(score)
    assert "filler" in flags(score)


def test_legitimate_filler_in_sentence_is_not_flagged():
    # "vielen dank" used inside real sentences must NOT trip the filler flag.
    segs = [
        _seg("S0", 0.0, 3.0, "Ja, vielen Dank für das ausführliche Gespräch heute."),
        _seg("S1", 3.0, 6.0, "Sehr gerne, vielen Dank auch dir und bis bald."),
        _seg("S0", 6.0, 9.0, "Wir hören dann nächste Woche voneinander."),
    ]
    score = score_transcript(segs)
    assert score["filler_segment_ratio"] == 0.0
    assert "filler" not in flags(score)


def test_phantom_speaker_is_flagged():
    # S0 dominates; S_phantom has a tiny share of total talk time
    segs = [_seg("S0", 0.0, 100.0, "lange rede " * 5,
                 words=[{"word": "w", "start": 0.0, "end": 100.0}])]
    segs.append(_seg("PHANTOM", 100.0, 101.0, "ja",
                     words=[{"word": "ja", "start": 100.0, "end": 101.0}]))
    score = score_transcript(segs)
    assert score["n_speakers"] == 2
    assert score["n_thin_speakers"] == 1            # ~1% share
    assert "phantom-speaker" in flags(score)


def test_talk_time_uses_word_durations_not_segment_span():
    # A segment spanning 0..100 but with only 1s of actual words must not be
    # counted as 100s of talk time (the inflation bug we saw in ad-hoc checks).
    seg = _seg("S0", 0.0, 100.0, "kurz", words=[{"word": "kurz", "start": 0.0, "end": 1.0}])
    other = _seg("S1", 100.0, 110.0, "zehn sekunden lang",
                 words=[{"word": "x", "start": 100.0, "end": 110.0}])
    score = score_transcript([seg, other])
    # S0 has 1s, S1 has 10s -> S0 is the minority by real talk time
    assert score["speaker_shares"]["S1"] > score["speaker_shares"]["S0"]


def test_vad_metrics_detect_words_over_silence():
    # word at t=50 but the only speech region is 0..2 -> word is over silence
    segs = [_seg("S0", 0.0, 1.0, "hallo", words=[{"word": "hallo", "start": 0.0, "end": 1.0}]),
            _seg("S0", 50.0, 51.0, "geist", words=[{"word": "geist", "start": 50.0, "end": 51.0}])]
    score = score_transcript(segs, speech_regions=[(0.0, 2.0)])
    assert score["words_outside_vad"] > 0.4
    assert "words-over-silence" in flags(score)


def test_empty_transcript_does_not_crash():
    score = score_transcript([])
    assert score["n_segments"] == 0
    assert score["n_speakers"] == 0
    assert flags(score) == []
