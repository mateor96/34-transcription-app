import json
import logging
from typing import AsyncGenerator

import httpx

from .exceptions import ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class OllamaService:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"ollama/{self.model}"

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.3},
        }

        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        body = (response.text or "").strip()
                        if response.status_code == 404:
                            raise ProviderModelError(
                                f"Model '{self.model}' not found. Run: ollama pull {self.model}",
                                provider=self.provider_name(),
                            )
                        raise ProviderModelError(
                            f"Ollama rejected the request: {body[:300] or 'no detail'}",
                            provider=self.provider_name(),
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise ProviderUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}. Run: ollama serve",
                provider=self.provider_name(),
            )
        except httpx.ReadTimeout:
            raise ProviderUnavailableError(
                "Ollama timed out. The model may still be loading — try again.",
                provider=self.provider_name(),
            )

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
