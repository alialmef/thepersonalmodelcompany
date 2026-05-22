"""State-of-the-art recall layer for PMC.

The existing `pmc.memory` module is a flat vector store keyed by training
completions — useful as a primitive but not enough to support the
"meets you like a friend who knows you" experience.

This module adds the architecture we actually need:

    Layer 1 — RAW SUBSTRATE      (already present in <storage>/raw/*.jsonl)
    Layer 2 — EPISODIC MEMORY    (this module — Episodes with summaries + embeddings)
    Layer 3 — SEMANTIC GRAPH     (this module — bi-temporal Facts)
    Layer 4 — WORKING MEMORY     (this module — daily snapshot of what's live)
    Layer 5 — NARRATIVE MEMORY   (this module — monthly era / arc snapshot)

Key choices, on the record:

  * **Consolidation runs on a frontier model** (Claude Sonnet 4.6, configurable).
    Small models can't reliably produce structured state-change detection,
    entity tagging, and emotional-tone labels at the quality the user
    experiences as "knowing." We use prompt caching so the cost stays
    bounded for daily incremental passes.

  * **Embeddings are local** (sentence-transformers BGE-base). Embeddings
    don't need a frontier model and we'd rather keep that compute on
    the user's Mac.

  * **Storage is SQLite + FTS5 + a vector blob column**. One file per
    user at `<storage>/users/<uid>/recall.db`. Inspectable, portable,
    backup-able. Co-locates structured rows with full-text and vector
    indexes so multi-signal retrieval is one query each, not three.

  * **Bi-temporal facts**. Every Fact carries `valid_from`, `valid_until`,
    and `invalidated_by`. State changes never delete prior facts — they
    supersede. We can answer "what was true on date X" honestly.

  * **Multi-signal retrieval**. The agent calls `recall.retrieve(query,
    scope, k)`. Fuses vector similarity + BM25 + entity-link boost +
    working-memory boost into one ranked list of MemoryFragments.
"""

from pmc.memory.recall.schema import (
    Episode,
    EpisodeKind,
    Fact,
    MemoryFragment,
    WorkingMemorySnapshot,
)
from pmc.memory.recall.store import RecallStore
from pmc.memory.recall.consolidate import Consolidator
from pmc.memory.recall.retrieve import retrieve, RetrievalScope

__all__ = [
    "Episode",
    "EpisodeKind",
    "Fact",
    "MemoryFragment",
    "WorkingMemorySnapshot",
    "RecallStore",
    "Consolidator",
    "retrieve",
    "RetrievalScope",
]
