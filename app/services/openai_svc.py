import logging
from typing import AsyncGenerator

from .exceptions import ProviderAuthError, ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"


class OpenAIService:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ProviderAuthError("OpenAI API key is required.", provider="openai")
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ProviderUnavailableError(
                "openai not installed. Run: uv add openai",
                provider="openai",
            )
        self._model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"openai/{self._model}"

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
        import openai
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=1024,
                temperature=0.3,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"Invalid OpenAI API key: {e}", provider=self.provider_name())
        except openai.NotFoundError as e:
            raise ProviderModelError(f"Model not found: {e}", provider=self.provider_name())
        except openai.APIConnectionError as e:
            raise ProviderUnavailableError(f"Cannot reach OpenAI API: {e}", provider=self.provider_name())

    async def check_health(self) -> bool:
        import openai
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "say ok"}],
                max_tokens=5,
            )
            return bool(resp.choices)
        except (openai.AuthenticationError, openai.NotFoundError):
            return False
        except Exception:
            return False
