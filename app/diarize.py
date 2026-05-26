from collections import defaultdict

import torch
from pyannote.audio import Pipeline

_pipeline: Pipeline | None = None

# A "speaker" is treated as a clustering artifact (amplified noise, a few
# backchannel "ja"s, or a real speaker that VBx split off a second cluster)
# when their total talk time is below BOTH an absolute floor AND a share of the
# whole conversation. The relative test is the important one: a 30s sliver of a
# 20-minute call is noise even though 30s isn't "small" in absolute terms.
# Artifacts are folded into the nearest real speaker — we never force a count.
_MIN_SPEAKER_ABS_S = 5.0       # always keep anyone past this on short clips
_MIN_SPEAKER_FRACTION = 0.05   # ...and on longer ones, require >=5% of talk time


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _pipeline.to(device)
    return _pipeline


def _drop_phantom_speakers(turns: list[dict]) -> list[dict]:
    """Reassign turns of sub-threshold speakers to the nearest surviving speaker."""
    totals: dict[str, float] = defaultdict(float)
    for t in turns:
        totals[t["speaker"]] += t["end"] - t["start"]

    grand_total = sum(totals.values())
    floor = max(_MIN_SPEAKER_ABS_S, _MIN_SPEAKER_FRACTION * grand_total)
    survivors = {spk for spk, total in totals.items() if total >= floor}
    # Don't strip everything (degenerate / very short recordings) or no-ops.
    if not survivors or len(survivors) == len(totals):
        return turns

    survivor_turns = [t for t in turns if t["speaker"] in survivors]
    out: list[dict] = []
    for t in turns:
        if t["speaker"] in survivors:
            out.append(t)
            continue
        mid = (t["start"] + t["end"]) / 2
        nearest = min(
            survivor_turns,
            key=lambda s: min(abs(mid - s["start"]), abs(mid - s["end"])),
        )
        out.append({**t, "speaker": nearest["speaker"]})
    return out


def diarize(audio_path: str, min_speakers: int | None = None, max_speakers: int | None = None) -> list[dict]:
    pipeline = _get_pipeline()
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(audio_path, **kwargs)
    annotation = output.exclusive_speaker_diarization

    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    # Only auto-prune phantoms when the caller didn't pin the count explicitly.
    if min_speakers is None and max_speakers is None:
        turns = _drop_phantom_speakers(turns)
    return turns
