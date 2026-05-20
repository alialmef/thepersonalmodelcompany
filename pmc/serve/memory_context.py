"""Per-user memory + identity context for the chat endpoint.

Wraps the three things every chat request needs from the memory layer:

- The user's `IdentityProfile` (system-prompt facts)
- The user's `MemoryStore` (vector store of past writing)
- A `Retriever` over that store

Loaded lazily on first request per user and cached for the lifetime of the
process. Designed so the serve layer can run with or without memory wired up:
construct PMCServer with no context provider and chat() works the same way
it did before this module existed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pmc.memory.embeddings import EmbeddingsClient
from pmc.memory.identity import IdentityProfile, build_identity_prompt
from pmc.memory.retriever import RetrievalResult, Retriever
from pmc.memory.store import MemoryStore
from pmc.storage.paths import StoragePaths


@dataclass
class MemoryContext:
    """Everything chat() needs for one user's recall + identity."""

    user_id: str
    identity: IdentityProfile
    store: MemoryStore
    retriever: Retriever

    def system_prompt(self) -> str:
        return build_identity_prompt(self.identity)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if self.store.count() == 0:
            return []
        return self.retriever.search(query=query, k=k)

    def format_context_block(self, results: list[RetrievalResult]) -> str:
        return self.retriever.format_context_block(results)


class MemoryContextProvider:
    """Lazy per-user MemoryContext loader with in-process caching.

    Pass to PMCServer. On first chat() per user, the provider opens the user's
    SQLite store, loads their identity.json, and constructs a Retriever. The
    context is cached so subsequent requests reuse the open DB handle.

    If a user has no identity.json on disk, we fall back to a minimal profile
    derived from user_id alone — the chat still works, just with a generic
    "your personal AI model" framing instead of style-tailored facts.
    """

    def __init__(self, paths: StoragePaths, embeddings: EmbeddingsClient) -> None:
        self.paths = paths
        self.embeddings = embeddings
        self._cache: dict[str, MemoryContext] = {}

    def get(self, user_id: str) -> MemoryContext:
        cached = self._cache.get(user_id)
        if cached is not None:
            return cached

        identity = self._load_identity(user_id)
        store_path = self.paths.memory_store_file(user_id)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(store_path)
        retriever = Retriever(store=store, embeddings=self.embeddings)

        ctx = MemoryContext(
            user_id=user_id,
            identity=identity,
            store=store,
            retriever=retriever,
        )
        self._cache[user_id] = ctx
        return ctx

    def invalidate(self, user_id: str) -> None:
        """Drop the cached context. Use after a deletion or retrain."""
        ctx = self._cache.pop(user_id, None)
        if ctx is not None:
            ctx.store.close()

    def _load_identity(self, user_id: str) -> IdentityProfile:
        identity_path = self.paths.identity_file(user_id)
        if not identity_path.exists():
            # Minimal profile — chat works, just without style facts.
            return IdentityProfile(user_id=user_id, display_name=user_id)
        try:
            data = json.loads(identity_path.read_text())
        except json.JSONDecodeError:
            return IdentityProfile(user_id=user_id, display_name=user_id)
        return IdentityProfile(
            user_id=data.get("user_id", user_id),
            display_name=data.get("display_name", user_id),
            style_summary=data.get("style_summary"),
            style_facts=tuple(data.get("style_facts", [])),
            tone=data.get("tone"),
        )


def save_identity(paths: StoragePaths, profile: IdentityProfile) -> Path:
    """Persist an IdentityProfile to disk under the user's storage root."""
    out = paths.identity_file(profile.user_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "user_id": profile.user_id,
                "display_name": profile.display_name,
                "style_summary": profile.style_summary,
                "style_facts": list(profile.style_facts),
                "tone": profile.tone,
            },
            indent=2,
        )
    )
    return out


def enrich_messages(
    messages: list[dict[str, str]],
    context: MemoryContext,
    *,
    retrieval_k: int = 5,
) -> list[dict[str, str]]:
    """Prepend a system message with identity prompt + retrieved memory.

    - If a system message is already present, the identity prompt is prepended
      to it (so callers can still pin extra behavior on top).
    - The most recent user message drives retrieval; if there are no user
      messages, only the identity prompt is included.
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    query = user_messages[-1]["content"] if user_messages else ""

    results = context.retrieve(query, k=retrieval_k) if query else []
    context_block = context.format_context_block(results)

    system_text = context.system_prompt()
    if context_block:
        system_text = f"{system_text}\n\n{context_block}"

    existing_system = next(
        (i for i, m in enumerate(messages) if m.get("role") == "system"), None
    )
    if existing_system is not None:
        # Merge our prompt above the caller's system content.
        merged = dict(messages[existing_system])
        merged["content"] = f"{system_text}\n\n{merged.get('content', '')}".strip()
        return [
            *messages[:existing_system],
            merged,
            *messages[existing_system + 1 :],
        ]

    return [{"role": "system", "content": system_text}, *messages]
