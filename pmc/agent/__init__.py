"""PMC agent — bring-your-own frontier-model layer.

The user picks a provider (Anthropic, OpenAI, Google, or OpenRouter for
everything else) and supplies their own API key. PMC stores the key
encrypted-at-rest and proxies chat requests to the chosen provider.

No fine-tuning. No model hosting. The agent's intelligence comes from
the frontier model the user picked; the *value* comes from the
structured personal graph the agent reasons over.

Module shape:
  crypto.py            — at-rest encryption for user API keys (Fernet)
  providers/base.py    — abstract Provider, Message, Response, Tool types
  providers/anthropic.py
  providers/openai.py
  providers/google.py
  providers/openrouter.py
  providers/registry.py — name → Provider lookup
  router.py            — FastAPI routes mounted at /v1/agent/*
"""

from pmc.agent.providers.base import (
    Message,
    Provider,
    ProviderConfig,
    ProviderError,
    Response,
)
from pmc.agent.providers.registry import (
    KNOWN_PROVIDERS,
    get_provider,
)

__all__ = [
    "KNOWN_PROVIDERS",
    "Message",
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "Response",
    "get_provider",
]
