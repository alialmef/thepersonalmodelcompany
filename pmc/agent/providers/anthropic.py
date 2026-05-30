"""Anthropic provider — Claude Sonnet / Opus / Haiku."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
    Response,
)


# Curated default model list surfaced in the Settings picker. Users can
# type a custom id if Anthropic ships a new one before we refresh this.
ANTHROPIC_DEFAULT_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


class AnthropicProvider:
    name = "anthropic"

    def _client(self, api_key: str):
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as e:
            raise ProviderError(
                "anthropic SDK not installed", kind="model"
            ) from e
        return anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Response:
        client = self._client(config.api_key)
        anthropic_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        kwargs: dict = {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": anthropic_msgs,
        }
        if system:
            kwargs["system"] = system
        try:
            r = await client.messages.create(**kwargs)
        except Exception as e:
            raise self._classify(e) from e
        text = "".join(
            block.text for block in r.content if getattr(block, "type", "") == "text"
        )
        return Response(
            text=text,
            model=r.model,
            usage={
                "input_tokens": getattr(r.usage, "input_tokens", 0),
                "output_tokens": getattr(r.usage, "output_tokens", 0),
            },
            finish_reason=getattr(r, "stop_reason", None),
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
        anthropic_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        kwargs: dict = {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": anthropic_msgs,
        }
        if system:
            kwargs["system"] = system
        try:
            async with client.messages.stream(**kwargs) as stream:
                async for chunk in stream.text_stream:
                    yield chunk
        except Exception as e:
            raise self._classify(e) from e

    async def list_models(self, *, api_key: str) -> list[str]:
        # Anthropic exposes a /v1/models listing, but the curated list
        # is more useful to users than the full catalog (which includes
        # deprecated dated revisions). Surface the curated set + let
        # the picker accept free-text overrides.
        return list(ANTHROPIC_DEFAULT_MODELS)

    async def validate_key(self, *, api_key: str) -> bool:
        client = self._client(api_key)
        try:
            await client.messages.create(
                model=ANTHROPIC_DEFAULT_MODELS[-1],  # cheapest (Haiku)
                max_tokens=4,
                messages=[{"role": "user", "content": "ok"}],
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _classify(e: Exception) -> ProviderError:
        s = str(e).lower()
        if "401" in s or "auth" in s or "unauthorized" in s:
            return ProviderError(str(e), kind="auth")
        if "429" in s or "rate" in s:
            return ProviderError(str(e), kind="rate_limit")
        if "model" in s and ("not found" in s or "invalid" in s):
            return ProviderError(str(e), kind="model")
        return ProviderError(str(e), kind="unknown")
