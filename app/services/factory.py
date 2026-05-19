from .anthropic_svc import AnthropicService
from .gemini import GeminiService
from .lmstudio import LMStudioService
from .ollama import OllamaService
from .openai_svc import OpenAIService

SUMMARIZATION_PROMPT = """You are a meeting assistant. Below is a meeting transcript with speaker labels and timestamps.

Write a concise summary with bullet points covering:
- Key topics discussed
- Decisions made (if any)
- Action items — include who is responsible (if any)
- Participants

Use 4–8 bullet points total. Only summarize what is actually in the transcript. Use plain bullet points starting with -.

TRANSCRIPT:
{transcript}

SUMMARY:"""


def format_transcript(segments: list, speaker_names: dict) -> str:
    lines = []
    for seg in segments:
        sid = seg.get("speaker", "SPEAKER_00")
        name = speaker_names.get(sid) or sid.replace("SPEAKER_", "Speaker ")
        start = seg.get("start", 0)
        mm, ss = int(start) // 60, int(start) % 60
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{mm:02d}:{ss:02d}] {name}: {text}")
    return "\n".join(lines)


def get_summarizer(cfg: dict):
    provider = cfg.get("provider") or "lmstudio"
    base_url = cfg.get("base_url") or ""
    model    = cfg.get("model") or ""
    api_key  = cfg.get("api_key") or ""

    if provider == "lmstudio":
        return LMStudioService(base_url=base_url or "http://localhost:1234", model=model)
    if provider == "ollama":
        return OllamaService(base_url=base_url or "http://localhost:11434", model=model)
    if provider == "gemini":
        return GeminiService(api_key=api_key, model=model)
    if provider == "anthropic":
        return AnthropicService(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIService(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider: {provider!r}")
