"""Memory layer — the recall half of the personal model.

A PMC adapter captures *how* a user writes (style, voice, rhythm). The memory
layer captures *what* the user has said — facts, events, relationships,
preferences — and surfaces them at inference time via semantic retrieval.

This is what lets your model say "you mentioned Portland last week" instead
of hallucinating an answer. LoRA can't reliably memorize facts; vectors can.

Three components:

- `EmbeddingsClient` — protocol + OpenAI/mock impls for turning text into vectors
- `MemoryStore` — per-user, SQLite-backed vector store with cosine search
- `Retriever` — semantic search wrapper that returns top-k snippets for a query
- `build_identity_prompt` — composes the per-user system prompt that frames
  the model as "yours, talking to you"
"""

from pmc.memory.embeddings import (
    EmbeddingsClient,
    MockEmbeddings,
    OpenAIEmbeddings,
)
from pmc.memory.identity import (
    IdentityProfile,
    build_first_contact_message,
    build_identity_prompt,
)
from pmc.memory.retriever import RetrievalResult, Retriever
from pmc.memory.store import MemoryItem, MemoryStore
from pmc.memory.sync import (
    completion_to_memory_item,
    sync_completions_to_memory,
)

__all__ = [
    "EmbeddingsClient",
    "IdentityProfile",
    "MemoryItem",
    "MemoryStore",
    "MockEmbeddings",
    "OpenAIEmbeddings",
    "RetrievalResult",
    "Retriever",
    "build_first_contact_message",
    "build_identity_prompt",
    "completion_to_memory_item",
    "sync_completions_to_memory",
]
