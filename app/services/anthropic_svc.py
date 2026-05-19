import logging
from typing import AsyncGenerator

from .exceptions import ProviderAuthError, ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicService:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ProviderAuthError("Anthropic API key is required.", provider="anthropic")
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ProviderUnavailableError(
                "anthropic not installed. Run: uv add anthropic",
                provider="anthropic",
            )
        self._model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"anthropic/{self._model}"

    async def stream_summarize(self, text: str) -> AsyncGenerator[str, None]:
        import anthropic
        from .factory import SUMMARIZATION_PROMPT
        prompt = SUMMARIZATION_PROMPT.format(transcript=text)
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for token in stream.text_stream:
                    yield token
        except anthropic.AuthenticationError as e:
            raise ProviderAuthError(f"Invalid Anthropic API key: {e}", provider=self.provider_name())
        except anthropic.NotFoundError as e:
            raise ProviderModelError(f"Model not found: {e}", provider=self.provider_name())
        except anthropic.APIConnectionError as e:
            raise ProviderUnavailableError(f"Cannot reach Anthropic API: {e}", provider=self.provider_name())

    async def check_health(self) -> bool:
        import anthropic
        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=5,
                messages=[{"role": "user", "content": "say ok"}],
            )
            return bool(msg.content)
        except (anthropic.AuthenticationError, anthropic.NotFoundError):
            return False
        except Exception:
            return False
