"""Per-user vector store, SQLite-backed.

One database file per user under `storage_root/<user_id>/memory/store.db`.
Embeddings are stored as raw float32 BLOBs; metadata lives in a JSON column.

Why SQLite + cosine-in-Python rather than pgvector/FAISS:

- Zero ops surface. The store is one file, copy-able, deletable, auditable.
- Per-user isolation is implicit (one DB file per user).
- For V0 scale (~10K items per user) cosine over a numpy matrix is
  sub-millisecond. We can swap to FAISS or pgvector later without changing
  the public API of MemoryStore — only the search implementation.
- "Delete a source" / "forget this conversation" / "wipe everything" become
  trivial SQL — no vector DB to coordinate with.

The store does NOT own embedding generation — callers pass vectors in.
That keeps the embedding model swappable and lets us re-embed cheaply.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    source      TEXT,            -- e.g. "imessage", "notes", "mail"
    source_id   TEXT,            -- original identifier within the source
    created_at  REAL NOT NULL,   -- unix timestamp
    metadata    TEXT NOT NULL,   -- JSON blob (free-form per-source data)
    dim         INTEGER NOT NULL,
    embedding   BLOB NOT NULL    -- float32 packed
);

CREATE INDEX IF NOT EXISTS idx_source ON memory_items(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at);
"""


@dataclass(frozen=True)
class MemoryItem:
    """One unit of recall — a snippet of the user's writing with metadata."""

    id: str
    text: str
    source: str
    source_id: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


def _pack_vector(vec: list[float]) -> bytes:
    """float32 packed bytes — small, fast to deserialize."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


class MemoryStore:
    """SQLite-backed per-user vector store.

    Construct with a path to a SQLite file. The DB is auto-created if missing.
    Thread-safe for reads; writes are serialized through SQLite's own locking.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # -------- mutation --------

    def add(self, item: MemoryItem, embedding: list[float]) -> None:
        """Add or replace one item. Use add_many for batches."""
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_items "
            "(id, text, source, source_id, created_at, metadata, dim, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.id,
                item.text,
                item.source,
                item.source_id,
                item.created_at,
                json.dumps(item.metadata),
                len(embedding),
                _pack_vector(embedding),
            ),
        )
        self._conn.commit()

    def add_many(self, items: Iterable[tuple[MemoryItem, list[float]]]) -> int:
        """Bulk add. Returns number of rows written. Single transaction."""
        rows = [
            (
                item.id,
                item.text,
                item.source,
                item.source_id,
                item.created_at,
                json.dumps(item.metadata),
                len(embedding),
                _pack_vector(embedding),
            )
            for item, embedding in items
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO memory_items "
            "(id, text, source, source_id, created_at, metadata, dim, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def delete(self, item_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def delete_source(self, source: str) -> int:
        """Wipe everything from one source. Returns rows deleted."""
        cur = self._conn.execute("DELETE FROM memory_items WHERE source = ?", (source,))
        self._conn.commit()
        return cur.rowcount

    def clear(self) -> int:
        """Wipe the whole store. Returns rows deleted."""
        cur = self._conn.execute("DELETE FROM memory_items")
        self._conn.commit()
        return cur.rowcount

    # -------- read --------

    def count(self, source: str | None = None) -> int:
        if source is None:
            cur = self._conn.execute("SELECT COUNT(*) FROM memory_items")
        else:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE source = ?", (source,)
            )
        return int(cur.fetchone()[0])

    def iter_all(self) -> Iterable[tuple[MemoryItem, list[float]]]:
        """Stream every item + its embedding. Used by Retriever to build matrix."""
        cur = self._conn.execute(
            "SELECT id, text, source, source_id, created_at, metadata, dim, embedding "
            "FROM memory_items"
        )
        for row in cur:
            item = MemoryItem(
                id=row[0],
                text=row[1],
                source=row[2],
                source_id=row[3],
                created_at=row[4],
                metadata=json.loads(row[5]),
            )
            yield item, _unpack_vector(row[7], row[6])

    def get(self, item_id: str) -> tuple[MemoryItem, list[float]] | None:
        cur = self._conn.execute(
            "SELECT id, text, source, source_id, created_at, metadata, dim, embedding "
            "FROM memory_items WHERE id = ?",
            (item_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        item = MemoryItem(
            id=row[0],
            text=row[1],
            source=row[2],
            source_id=row[3],
            created_at=row[4],
            metadata=json.loads(row[5]),
        )
        return item, _unpack_vector(row[7], row[6])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
