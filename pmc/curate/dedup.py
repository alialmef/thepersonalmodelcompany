"""Exact and near-duplicate detection.

Personal data is full of repetition: forwarded threads, copy-pasted templates,
auto-replies. We dedupe with two passes:

1. Exact: stable content hash, O(1) lookup.
2. Near-duplicate: character n-gram shingles + Jaccard similarity. Linear scan
   for V0 — fine up to ~100K examples. Swap in MinHash/LSH if it gets slow.
"""

from __future__ import annotations

import hashlib
import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> set[str]:
    """Character n-grams over the normalized text."""
    norm = _normalize(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class Deduplicator:
    """Tracks seen content and reports duplicates.

    threshold: Jaccard similarity above which two items are considered duplicates.
    n: shingle size in characters. 5 works well for short-to-medium text.
    min_length: don't bother shingling content shorter than this — exact-hash only.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        n: int = 5,
        min_length: int = 30,
    ) -> None:
        self.threshold = threshold
        self.n = n
        self.min_length = min_length
        self._hashes: set[str] = set()
        self._shingles: list[set[str]] = []

    def check(self, text: str) -> tuple[bool, float]:
        """Return (is_duplicate, max_similarity)."""
        h = content_hash(text)
        if h in self._hashes:
            return True, 1.0

        if len(_normalize(text)) < self.min_length:
            self._hashes.add(h)
            return False, 0.0

        sh = shingles(text, self.n)
        max_sim = 0.0
        for existing in self._shingles:
            sim = jaccard(sh, existing)
            if sim > max_sim:
                max_sim = sim
                if sim >= self.threshold:
                    return True, sim

        self._hashes.add(h)
        self._shingles.append(sh)
        return False, max_sim
