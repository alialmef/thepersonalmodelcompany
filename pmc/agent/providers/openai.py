"""OpenAI provider — GPT-4 / GPT-5 family."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
    Response,
)


OPENAI_DEFAULT_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]


class OpenAIProvider:
    name = "openai"

    def _client(self, api_key: str):
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as e:
            raise ProviderError("openai SDK not installed", kind="model") from e
        return AsyncOpenAI(api_key=api_key)

    async def chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Response:
        client = self._client(config.api_key)
        openai_msgs: list[dict] = []
        if system:
            openai_msgs.append({"role": "system", "content": system})
        openai_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            r = await client.chat.completions.create(
                model=config.model,
                messages=openai_msgs,
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
        openai_msgs: list[dict] = []
        if system:
            openai_msgs.append({"role": "system", "content": system})
        openai_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            stream = await client.chat.completions.create(
                model=config.model,
                messages=openai_msgs,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise self._classify(e) from e

    async def list_models(self, *, api_key: str) -> list[str]:
        return list(OPENAI_DEFAULT_MODELS)

    async def validate_key(self, *, api_key: str) -> bool:
        client = self._client(api_key)
        try:
            await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=4,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _classify(e: Exception) -> ProviderError:
        s = str(e).lower()
        if "401" in s or "incorrect api key" in s or "invalid_api_key" in s:
            return ProviderError(str(e), kind="auth")
        if "429" in s or "rate" in s:
            return ProviderError(str(e), kind="rate_limit")
        if "model_not_found" in s or "does not exist" in s:
            return ProviderError(str(e), kind="model")
        return ProviderError(str(e), kind="unknown")
