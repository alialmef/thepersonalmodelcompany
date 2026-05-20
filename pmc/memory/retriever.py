"""Semantic retriever — top-k cosine over a user's MemoryStore.

Loads all vectors into memory on first query and caches them. For ~10K items
× 1536 dims that's ~60MB — fine for one user, expensive if we ever fan out
to many users in one process. If/when that becomes a problem, swap the inner
math for FAISS or pgvector without changing this module's API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pmc.memory.embeddings import EmbeddingsClient
from pmc.memory.store import MemoryItem, MemoryStore


@dataclass(frozen=True)
class RetrievalResult:
    """One result from a semantic search."""

    item: MemoryItem
    score: float  # cosine similarity in [-1, 1]; higher = more similar


class Retriever:
    """Semantic search over a MemoryStore.

    Construct with a store + embeddings client. Call `search(query, k)` to get
    the top-k most similar items. The first call loads the matrix into memory;
    subsequent calls reuse the cache. Call `refresh()` after batch inserts.
    """

    def __init__(self, store: MemoryStore, embeddings: EmbeddingsClient) -> None:
        self.store = store
        self.embeddings = embeddings
        self._cache: list[tuple[MemoryItem, list[float]]] | None = None

    def refresh(self) -> None:
        """Drop the matrix cache. Next search re-reads from SQLite."""
        self._cache = None

    def _load(self) -> list[tuple[MemoryItem, list[float]]]:
        if self._cache is None:
            self._cache = list(self.store.iter_all())
        return self._cache

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        sources: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return top-k most similar items.

        - `query`: natural-language query string.
        - `k`: max results.
        - `min_score`: filter out results below this cosine similarity.
        - `sources`: if provided, only search within these source types
          (e.g. `["imessage", "notes"]`).
        """
        items = self._load()
        if sources is not None:
            items = [(it, vec) for it, vec in items if it.source in sources]
        if not items:
            return []

        # Embed the query (single API call).
        query_vec = self.embeddings.embed([query])[0]
        q_norm = math.sqrt(sum(x * x for x in query_vec)) or 1.0

        scored: list[RetrievalResult] = []
        for item, vec in items:
            # Vectors from OpenAI are already L2-normalized; for safety we
            # normalize defensively, which costs nothing for normalized inputs.
            v_norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = sum(q * v for q, v in zip(query_vec, vec))
            score = dot / (q_norm * v_norm)
            if score >= min_score:
                scored.append(RetrievalResult(item=item, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def format_context_block(
        self,
        results: list[RetrievalResult],
        max_chars: int = 2000,
    ) -> str:
        """Format retrieved snippets as a system-prompt context block.

        Truncates the combined text at `max_chars` so we don't blow past
        context windows on small bases. Items are listed newest-first.
        """
        if not results:
            return ""
        # Sort by recency within the top-k so the model sees fresh memories last.
        results_sorted = sorted(results, key=lambda r: r.item.created_at)
        lines: list[str] = ["Relevant snippets from your past writing:"]
        used = 0
        for r in results_sorted:
            snippet = f"- [{r.item.source}] {r.item.text.strip()}"
            if used + len(snippet) > max_chars:
                break
            lines.append(snippet)
            used += len(snippet) + 1  # +1 for newline
        return "\n".join(lines)
