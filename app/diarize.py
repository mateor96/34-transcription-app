import torch
from pyannote.audio import Pipeline

_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _pipeline.to(device)
    return _pipeline


def diarize(audio_path: str, min_speakers: int | None = None, max_speakers: int | None = None) -> list[dict]:
    pipeline = _get_pipeline()
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(audio_path, **kwargs)
    annotation = output.exclusive_speaker_diarization

    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
