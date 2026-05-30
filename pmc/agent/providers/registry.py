"""Provider lookup. Imported lazily so missing SDKs don't break startup."""

from __future__ import annotations

from typing import Optional

from pmc.agent.providers.base import Provider


# Display metadata used by the Settings UI's provider picker. The
# `default_models` list seeds the model dropdown when the user picks
# the provider; users can type custom model ids too.
KNOWN_PROVIDERS: list[dict] = [
    {
        "id": "anthropic",
        "label": "Claude (Anthropic)",
        "default_models": [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        "key_prefix_hint": "sk-ant-",
        "console_url": "https://console.anthropic.com/settings/keys",
    },
    {
        "id": "openai",
        "label": "GPT (OpenAI)",
        "default_models": [
            "gpt-5",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "key_prefix_hint": "sk-",
        "console_url": "https://platform.openai.com/api-keys",
    },
    {
        "id": "google",
        "label": "Gemini (Google)",
        "default_models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-pro",
            "gemini-2.0-flash",
        ],
        "key_prefix_hint": "AIza",
        "console_url": "https://aistudio.google.com/apikey",
    },
    {
        "id": "openrouter",
        "label": "Open source via OpenRouter",
        "default_models": [
            "moonshotai/kimi-k2.6",
            "deepseek/deepseek-r2",
            "deepseek/deepseek-v3.5",
            "qwen/qwen3-235b-a22b-instruct",
            "qwen/qwen3-coder-480b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "x-ai/grok-4",
            "mistralai/mistral-large-2",
        ],
        "key_prefix_hint": "sk-or-",
        "console_url": "https://openrouter.ai/keys",
    },
]


def get_provider(name: str) -> Optional[Provider]:
    """Return a provider instance for the given id, or None if unknown."""
    if name == "anthropic":
        from pmc.agent.providers.anthropic import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from pmc.agent.providers.openai import OpenAIProvider
        return OpenAIProvider()
    if name == "google":
        from pmc.agent.providers.google import GoogleProvider
        return GoogleProvider()
    if name == "openrouter":
        from pmc.agent.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider()
    return None


def is_known_provider(name: str) -> bool:
    return name in {"anthropic", "openai", "google", "openrouter"}
