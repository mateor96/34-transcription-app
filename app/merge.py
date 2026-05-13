def merge(whisper_result: dict, diarization_turns: list[dict]) -> list[dict]:
    """
    Assigns a speaker to each word by finding which diarization turn contains
    the word's midpoint, then groups consecutive same-speaker words into segments.
    """
    words = [
        {"word": w["word"], "start": w["start"], "end": w["end"], "speaker": None}
        for seg in whisper_result.get("segments", [])
        for w in seg.get("words", [])
    ]

    if not words:
        return []

    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        word["speaker"] = _find_speaker(midpoint, diarization_turns)

    return _group_by_speaker(words)


def _find_speaker(midpoint: float, turns: list[dict]) -> str:
    for turn in turns:
        if turn["start"] <= midpoint <= turn["end"]:
            return turn["speaker"]
    # Fall back to nearest turn if midpoint lands in a gap
    if not turns:
        return "SPEAKER_00"
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
