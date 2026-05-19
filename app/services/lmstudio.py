import json
import logging
from typing import AsyncGenerator

import httpx

from .exceptions import ProviderAuthError, ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234"
DEFAULT_MODEL = ""  # empty = use whatever LM Studio has loaded


def _extract_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return err.get("message") or str(err)
        if isinstance(err, str):
            return err
    except json.JSONDecodeError:
        pass
    return (body or "no detail").strip()[:300]


class LMStudioService:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"lmstudio/{self.model}" if self.model else "lmstudio"

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
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
                    if response.status_code >= 400:
                        await response.aread()
                        body = response.text
                        msg = _extract_error_message(body)
                        if response.status_code == 404:
                            raise ProviderModelError(
                                f"Model not found in LM Studio: {msg}",
                                provider=self.provider_name(),
                            )
                        raise ProviderModelError(
                            f"LM Studio rejected the request: {msg}",
                            provider=self.provider_name(),
                        )
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
