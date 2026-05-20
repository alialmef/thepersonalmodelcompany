"""Embedding clients.

Tiny protocol so the memory layer can be tested deterministically with
`MockEmbeddings` (hash-derived vectors) and run in production with
`OpenAIEmbeddings` (text-embedding-3-small, 1536 dims, ~$0.02/1M tokens).
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Protocol, runtime_checkable


# Default embedding dimension for text-embedding-3-small. Stored alongside
# every vector so we can detect dimension drift if we ever swap models.
DEFAULT_DIM = 1536


@runtime_checkable
class EmbeddingsClient(Protocol):
    """Turn text into fixed-length vectors. Batched for efficiency."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...


class MockEmbeddings:
    """Deterministic embeddings derived from a text hash.

    Same text → same vector. Different texts → different (but similar-looking)
    vectors. Useful for tests where we want predictable retrieval ordering
    without spending OpenAI credits.

    Not for production. Vectors do not encode semantic similarity, only
    text equality.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # SHA-256 fanned into `dim` floats in [-1, 1], then L2-normalized.
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        vec: list[float] = []
        i = 0
        while len(vec) < self.dim:
            chunk = seed[i % len(seed)]
            vec.append((chunk / 127.5) - 1.0)
            i += 1
        # Normalize so cosine == dot product.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class OpenAIEmbeddings:
    """text-embedding-3-small by default. Batches at 100 inputs per call.

    Caller is responsible for providing an API key via OPENAI_API_KEY env var
    or by passing one in. The OpenAI SDK is imported lazily so this module
    can be imported in environments that don't have it installed.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = DEFAULT_DIM,
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.batch_size = batch_size
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "openai package required for OpenAIEmbeddings. "
                    "Install with: pip install 'pmc[eval]'"
                ) from e
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set and no api_key passed to OpenAIEmbeddings"
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # OpenAI rejects empty strings; replace with a single space so
            # the index alignment with the caller's list is preserved.
            safe = [t if t.strip() else " " for t in batch]
            resp = client.embeddings.create(model=self.model, input=safe)
            for item in resp.data:
                out.append(list(item.embedding))
        return out
