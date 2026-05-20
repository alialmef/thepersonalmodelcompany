"""Sync helpers — write user data into the MemoryStore.

The store is a passive container; this module is what knows how to
turn a `Completion` into a memory `MemoryItem` and embed it. Called from:

- The curate pipeline (batch sync after each curation run)
- The orchestrator's incremental ingest path (one item at a time, live)

We embed the user's writing — the assistant-role messages on each completion.
That's the substance the model needs to recall later. The conversational
context (user-role messages, i.e. what others wrote to the user) is included
in the metadata so the retriever can show "you wrote this in reply to:" if
useful in the UI.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable

from pmc.memory.embeddings import EmbeddingsClient
from pmc.memory.store import MemoryItem, MemoryStore
from pmc.schema.conversation import Completion, Role


def _stable_id(completion: Completion, text: str) -> str:
    """Deterministic ID so re-syncing the same completion overwrites rather
    than duplicates. Hash the completion id + the candidate text."""
    base = f"{str(completion.id)}:{text}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]


def _candidate_text(completion: Completion) -> str | None:
    """Pull the first usable candidate's text. None if nothing to embed."""
    if not completion.candidates:
        return None
    cand = completion.candidates[0]
    pieces = [m.content for m in cand.messages if m.role == Role.ASSISTANT and m.content.strip()]
    if not pieces:
        return None
    return "\n".join(pieces).strip()


def _conversation_context(completion: Completion, max_chars: int = 400) -> str:
    """Short summary of what was said to the user, for metadata only."""
    incoming = [
        m.content for m in completion.conversation.messages
        if m.role == Role.USER and m.content.strip()
    ]
    if not incoming:
        return ""
    joined = " | ".join(incoming)
    return joined[:max_chars]


def completion_to_memory_item(completion: Completion) -> MemoryItem | None:
    """Convert a Completion into a MemoryItem. None if no usable candidate text."""
    text = _candidate_text(completion)
    if text is None:
        return None

    # Source type from the first message origin, if available.
    origin = None
    if completion.candidates and completion.candidates[0].messages:
        origin = getattr(completion.candidates[0].messages[0], "origin", None)
    if origin is None and completion.conversation.messages:
        origin = getattr(completion.conversation.messages[0], "origin", None)

    source = "unknown"
    source_id = None
    if origin is not None:
        source = getattr(origin, "kind", source) or source
        source_id = getattr(origin, "id", None)

    return MemoryItem(
        id=_stable_id(completion, text),
        text=text,
        source=str(source),
        source_id=source_id,
        created_at=time.time(),
        metadata={
            "completion_id": str(completion.id),
            "in_reply_to": _conversation_context(completion),
        },
    )


def sync_completions_to_memory(
    completions: Iterable[Completion],
    store: MemoryStore,
    embeddings: EmbeddingsClient,
    *,
    batch_size: int = 100,
) -> int:
    """Embed a batch of completions and write them into the memory store.

    Returns the number of items written. Items with no usable candidate text
    are skipped silently — they had nothing to embed.

    Embedding is the only network call; we batch up to `batch_size` per request
    to stay efficient. Storage writes are also batched into a single transaction.
    """
    items: list[MemoryItem] = []
    for c in completions:
        item = completion_to_memory_item(c)
        if item is not None:
            items.append(item)
    if not items:
        return 0

    # Embed in batches.
    written = 0
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        vectors = embeddings.embed([it.text for it in chunk])
        written += store.add_many(zip(chunk, vectors))
    return written
