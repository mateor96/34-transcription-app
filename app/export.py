def to_txt(segments: list[dict]) -> str:
    return "\n\n".join(f"{s['speaker']}: {s['text']}" for s in segments)


def to_json(segments: list[dict]) -> dict:
    return {
        "speakers": sorted({s["speaker"] for s in segments}),
        "segments": [
            {
                "speaker": s["speaker"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "words": s.get("words", []),
            }
            for s in segments
        ],
    }


def to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [
            str(i),
            f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}",
            f"{seg['speaker']}: {seg['text']}",
            "",
        ]
    return "\n".join(lines)


def to_markdown(segments: list[dict]) -> str:
    return "\n\n".join(
        f"**{s['speaker']}** `{_ts(s['start'])}`: {s['text']}"
        for s in segments
    )


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
