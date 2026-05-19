import logging
from typing import AsyncGenerator

from .exceptions import ProviderAuthError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


class GeminiService:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ProviderAuthError("Gemini API key is required.", provider="gemini")
        try:
            import google.genai as genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise ProviderUnavailableError(
                "google-genai not installed. Run: uv add google-genai",
                provider="gemini",
            )
        self._model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"gemini/{self._model}"

    async def stream_summarize(self, text: str) -> AsyncGenerator[str, None]:
        from .factory import SUMMARIZATION_PROMPT
        prompt = SUMMARIZATION_PROMPT.format(transcript=text)
        try:
            async for chunk in self._client.aio.models.generate_content_stream(
                model=self._model, contents=prompt
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            msg = str(e)
            if "API_KEY" in msg.upper() or "401" in msg or "403" in msg:
                raise ProviderAuthError(f"Invalid Gemini API key: {e}", provider=self.provider_name())
            raise ProviderUnavailableError(f"Gemini error: {e}", provider=self.provider_name())

    async def check_health(self) -> bool:
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model, contents="say ok"
            )
            return bool(resp.text)
        except Exception:
            return False
