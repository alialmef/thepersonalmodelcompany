"""OpenRouter provider — single endpoint to every open-source frontier
model (and many proprietary ones) routed through one API.

This is the "everything else" lane: Kimi K2, DeepSeek V3/R1, Qwen 3,
Llama 4, Mistral Large, GLM-4.5, Yi, etc. Users who want to run on
open weights (or compare across providers cheaply) point here.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
    Response,
)


# A curated subset of OpenRouter's catalog that's both serverless and
# strong enough to be a real personal agent. Users can paste any
# OpenRouter model id; this list just seeds the picker.
OPENROUTER_DEFAULT_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-r2",
    "deepseek/deepseek-v3.5",
    "qwen/qwen3-235b-a22b-instruct",
    "qwen/qwen3-coder-480b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "x-ai/grok-4",
    "mistralai/mistral-large-2",
]

_BASE_URL = "https://openrouter.ai/api/v1/"


class OpenRouterProvider:
    name = "openrouter"

    def _client(self, api_key: str):
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as e:
            raise ProviderError("openai SDK not installed", kind="model") from e
        # OpenRouter recommends sending a referer + app title for
        # leaderboard / abuse signals. Doesn't affect functionality.
        return AsyncOpenAI(
            api_key=api_key,
            base_url=_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://thepersonalmodelcompany.com",
                "X-Title": "Personal Model Company",
            },
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Response:
        client = self._client(config.api_key)
        or_msgs: list[dict] = []
        if system:
            or_msgs.append({"role": "system", "content": system})
        or_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            r = await client.chat.completions.create(
                model=config.model,
                messages=or_msgs,
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
        or_msgs: list[dict] = []
        if system:
            or_msgs.append({"role": "system", "content": system})
        or_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            stream = await client.chat.completions.create(
                model=config.model,
                messages=or_msgs,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise self._classify(e) from e

    async def list_models(self, *, api_key: str) -> list[str]:
        return list(OPENROUTER_DEFAULT_MODELS)

    async def validate_key(self, *, api_key: str) -> bool:
        client = self._client(api_key)
        try:
            # Smallest free model on OpenRouter; doesn't burn credits.
            await client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=4,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _classify(e: Exception) -> ProviderError:
        s = str(e).lower()
        if "401" in s or "unauthorized" in s or "invalid_api_key" in s:
            return ProviderError(str(e), kind="auth")
        if "429" in s or "rate" in s or "quota" in s:
            return ProviderError(str(e), kind="rate_limit")
        if "model" in s and ("not found" in s or "invalid" in s or "unavailable" in s):
            return ProviderError(str(e), kind="model")
        return ProviderError(str(e), kind="unknown")
