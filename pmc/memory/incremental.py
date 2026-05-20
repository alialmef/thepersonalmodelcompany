"""Incremental ingestion — embed new items into the memory store as they appear.

This is the "your model learns from you every day" plumbing. The Mac app's
ingest layer (Rust modules watching iMessage / Notes / Mail) drops new
Completions into the per-user store; this module embeds each new item and
writes it into the MemoryStore so it's immediately recallable.

Unlike a one-shot training run, this is a stream — single items or small
batches, called frequently. Embedding cost is ~$0.0001 per item with
text-embedding-3-small, so per-day cost stays effectively zero.

Crucially: incremental ingest does NOT trigger retraining. The LoRA weights
are refreshed on a separate cadence (see `pmc/orchestrator/refresh.py`).
This module only updates the recall layer — facts are immediately available
to the model via retrieval; voice mimicry only changes on full retrains.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from pmc.memory.embeddings import EmbeddingsClient
from pmc.memory.store import MemoryStore
from pmc.memory.sync import completion_to_memory_item
from pmc.schema.conversation import Completion


def ingest_one(
    completion: Completion,
    store: MemoryStore,
    embeddings: EmbeddingsClient,
) -> bool:
    """Embed and store a single completion. Returns True if written."""
    item = completion_to_memory_item(completion)
    if item is None:
        return False
    [vec] = embeddings.embed([item.text])
    store.add(item, vec)
    return True


def ingest_many(
    completions: Iterable[Completion],
    store: MemoryStore,
    embeddings: EmbeddingsClient,
    *,
    batch_size: int = 100,
) -> int:
    """Embed and store a batch. Returns count of items written.

    Items with no embeddable text are silently skipped. The store add is a
    single transaction per batch.
    """
    items = [
        item
        for item in (completion_to_memory_item(c) for c in completions)
        if item is not None
    ]
    if not items:
        return 0

    written = 0
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        vectors = embeddings.embed([it.text for it in chunk])
        written += store.add_many(zip(chunk, vectors))
    return written


def diff_and_ingest(
    completions: Iterable[Completion],
    store: MemoryStore,
    embeddings: EmbeddingsClient,
) -> tuple[int, int]:
    """Embed + store only completions not already in the memory store.

    Returns (new_items_written, items_already_present). Useful for the
    daemon's "scan for new data" pass — safe to call repeatedly without
    re-embedding the same items.
    """
    new_items: list[Completion] = []
    existing = 0
    for c in completions:
        item = completion_to_memory_item(c)
        if item is None:
            continue
        if store.get(item.id) is not None:
            existing += 1
        else:
            new_items.append(c)
    written = ingest_many(new_items, store, embeddings) if new_items else 0
    return written, existing


def watermark(store: MemoryStore) -> float:
    """Return the timestamp of the most recently added memory item.

    Used by a watcher daemon to skip re-scanning items it has already seen.
    Returns 0.0 if the store is empty.
    """
    latest = 0.0
    for item, _ in store.iter_all():
        if item.created_at > latest:
            latest = item.created_at
    return latest or time.time()
