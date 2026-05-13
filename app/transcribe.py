import mlx_whisper

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(audio_path: str, model: str = DEFAULT_MODEL) -> dict:
    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        word_timestamps=True,
    )
