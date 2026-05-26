def merge(whisper_result: dict, diarization_turns: list[dict]) -> list[dict]:
    """
    Assigns a speaker to each word by maximum temporal overlap with the
    diarization turns, then groups consecutive same-speaker words into segments.
    """
    words = [
        {"word": w["word"], "start": w["start"], "end": w["end"], "speaker": None}
        for seg in whisper_result.get("segments", [])
        for w in seg.get("words", [])
    ]

    if not words:
        return []

    for word in words:
        word["speaker"] = _find_speaker(word["start"], word["end"], diarization_turns)

    return _group_by_speaker(words)


def _find_speaker(start: float, end: float, turns: list[dict]) -> str:
    """Speaker whose turn overlaps the word the most.

    Maximum-overlap beats midpoint-in-turn for words that straddle a speaker
    change, where the midpoint can land on the wrong side of the boundary.
    """
    if not turns:
        return "SPEAKER_00"

    best_speaker = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
    if best_speaker is not None:
        return best_speaker

    # No overlap (word sits in a gap) → nearest turn by edge distance.
    midpoint = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(midpoint - t["start"]), abs(midpoint - t["end"])))
    return nearest["speaker"]


def _group_by_speaker(words: list[dict]) -> list[dict]:
    segments = []
    current_speaker = words[0]["speaker"]
    current_words = [words[0]]

    for word in words[1:]:
        if word["speaker"] == current_speaker:
            current_words.append(word)
        else:
            segments.append(_make_segment(current_speaker, current_words))
            current_speaker = word["speaker"]
            current_words = [word]

    segments.append(_make_segment(current_speaker, current_words))
    return segments


def _make_segment(speaker: str, words: list[dict]) -> dict:
    return {
        "speaker": speaker,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": "".join(w["word"] for w in words).strip(),
        "words": words,
    }
