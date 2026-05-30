"""Google Gemini provider.

Uses the OpenAI-compatible endpoint Google exposes at
https://generativelanguage.googleapis.com/v1beta/openai/ — saves us a
second SDK dependency and means the call shape is identical to OpenAI.
The trade-off: a small subset of Gemini features (e.g. native grounded
search, full multimodal) aren't exposed through the compat endpoint.
For the chat use-case PMC needs today, the compat surface is enough.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
    Response,
)


GOOGLE_DEFAULT_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-pro",
    "gemini-2.0-flash",
]

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GoogleProvider:
    name = "google"

    def _client(self, api_key: str):
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as e:
            raise ProviderError("openai SDK not installed", kind="model") from e
        return AsyncOpenAI(api_key=api_key, base_url=_BASE_URL)

    async def chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Response:
        client = self._client(config.api_key)
        google_msgs: list[dict] = []
        if system:
            google_msgs.append({"role": "system", "content": system})
        google_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            r = await client.chat.completions.create(
                model=config.model,
                messages=google_msgs,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise self._classify(e) from e
        choice = r.choices[0]
        return Response(
            text=choice.message.content or "",
            model=r.model,
            usage={
                "input_tokens": getattr(r.usage, "prompt_tokens", 0),
                "output_tokens": getattr(r.usage, "completion_tokens", 0),
            },
            finish_reason=choice.finish_reason,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        client = self._client(config.api_key)
        google_msgs: list[dict] = []
        if system:
            google_msgs.append({"role": "system", "content": system})
        google_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            stream = await client.chat.completions.create(
                model=config.model,
                messages=google_msgs,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise self._classify(e) from e

    async def list_models(self, *, api_key: str) -> list[str]:
        return list(GOOGLE_DEFAULT_MODELS)

    async def validate_key(self, *, api_key: str) -> bool:
        client = self._client(api_key)
        try:
            await client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=4,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _classify(e: Exception) -> ProviderError:
        s = str(e).lower()
        if "401" in s or "403" in s or "api key" in s and "invalid" in s:
            return ProviderError(str(e), kind="auth")
        if "429" in s or "quota" in s or "rate" in s:
            return ProviderError(str(e), kind="rate_limit")
        if "model" in s and ("not found" in s or "not supported" in s):
            return ProviderError(str(e), kind="model")
        return ProviderError(str(e), kind="unknown")
