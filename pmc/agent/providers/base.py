"""Provider interface — what every BYO-model adapter must implement.

Keep this surface as small as possible. Every concrete adapter just
needs to translate our (messages, system, model) tuple into its native
SDK call and translate the response back into a Response.

We deliberately don't model tool calls in Phase 1.2 — that lands when
the sub-agent swarm in Phase 4 needs structured outputs. For now the
agent is plain chat completions over the user's graph context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional, Protocol


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class ProviderConfig:
    """What's stored per account about which model PMC should talk to.
    The api_key is the *decrypted* key — it lives in memory only for
    the duration of one request. At rest it's encrypted by pmc.agent.crypto.
    """

    provider: str       # "anthropic" | "openai" | "google" | "openrouter"
    model: str          # provider-specific model id
    api_key: str        # plaintext, only in memory


@dataclass
class Response:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None


class ProviderError(RuntimeError):
    """Raised when a provider call fails. Carries a hint about whether
    it's a config issue (bad key, unknown model) the caller should
    surface to the user vs a transient backend failure."""

    def __init__(self, message: str, *, kind: str = "unknown") -> None:
        super().__init__(message)
        self.kind = kind  # "auth" | "rate_limit" | "model" | "network" | "unknown"


class Provider(Protocol):
    """The minimal surface every BYO-model adapter implements."""

    name: str  # "anthropic" | "openai" | "google" | "openrouter"

    async def chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Response:
        """Single-shot chat. Returns a Response."""
        ...

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        config: ProviderConfig,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Streaming chat. Yields text chunks. Implementations may
        choose to fall back to chat() + yielding a single chunk if the
        provider's SDK doesn't expose streaming cleanly."""
        ...

    async def list_models(self, *, api_key: str) -> list[str]:
        """Return the user-selectable model ids the provider exposes for
        this key. Used by the Settings UI to populate the picker.
        Implementations may return a curated subset rather than the
        full catalog."""
        ...

    async def validate_key(self, *, api_key: str) -> bool:
        """Cheap probe to confirm the key actually works. Used by the
        Settings UI's 'Test connection' button."""
        ...
