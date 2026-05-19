import json
import logging
from typing import AsyncGenerator

import httpx

from .exceptions import ProviderAuthError, ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234"
DEFAULT_MODEL = ""  # empty = use whatever LM Studio has loaded


class LMStudioService:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"lmstudio/{self.model}" if self.model else "lmstudio"

    async def stream_summarize(self, text: str) -> AsyncGenerator[str, None]:
        from .factory import SUMMARIZATION_PROMPT
        prompt = SUMMARIZATION_PROMPT.format(transcript=text)
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        if self.model:
            payload["model"] = self.model

        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code == 404:
                        raise ProviderModelError(
                            f"Model '{self.model}' not found in LM Studio.",
                            provider=self.provider_name(),
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content") or ""
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.ConnectError:
            raise ProviderUnavailableError(
                f"Cannot connect to LM Studio at {self.base_url}. Is it running?",
                provider=self.provider_name(),
            )
        except httpx.ReadTimeout:
            raise ProviderUnavailableError(
                "LM Studio timed out. The model may still be loading.",
                provider=self.provider_name(),
            )

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False
