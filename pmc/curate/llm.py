"""LLM client protocol for curation steps that need a judge model.

We keep this minimal — just a `complete()` method — so the rest of the curate
layer can be tested with a deterministic mock and we can swap providers later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Sync LLM client used by curation steps that need a judge model."""

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


class MockLLMClient:
    """Deterministic stub for tests and offline runs.

    If a key in `responses` is a substring of the prompt, returns its value.
    Otherwise returns `default`. Every call is recorded in `calls`.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "",
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append((system, prompt))
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return self.default
