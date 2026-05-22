"""Local embedding pipeline.

We use sentence-transformers with the BGE-base model — strong English
embeddings, runs in ~50ms/passage on Apple Silicon CPU, fully local. No
network call, no API cost, no user-data egress for this step.

We lazy-load the model so importing the recall package is cheap.
"""

from __future__ import annotations

import threading
from typing import Iterable, Optional

import numpy as np

_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class LocalEmbedder:
    """Thread-safe singleton wrapper around sentence-transformers.

    Loads on first use. Subsequent encodes are fast. Embeddings are
    L2-normalized so downstream cosine == dot product.
    """

    _lock = threading.Lock()
    _model = None
    _model_name: Optional[str] = None

    @classmethod
    def model(cls, name: str = _DEFAULT_MODEL):
        if cls._model is None or cls._model_name != name:
            with cls._lock:
                if cls._model is None or cls._model_name != name:
                    from sentence_transformers import SentenceTransformer
                    cls._model = SentenceTransformer(name)
                    cls._model_name = name
        return cls._model

    @classmethod
    def name(cls) -> str:
        return cls._model_name or _DEFAULT_MODEL

    @classmethod
    def embed(cls, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        m = cls.model()
        vectors = m.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    @classmethod
    def embed_one(cls, text: str) -> np.ndarray:
        return cls.embed([text])[0]
