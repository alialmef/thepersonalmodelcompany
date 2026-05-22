"""SQLite-backed recall store.

One file per user at `<storage>/users/<uid>/recall.db`. Co-locates:
  * the structural episode rows
  * an FTS5 index for keyword search
  * a vector blob column (cosine via numpy at retrieval time)
  * the bi-temporal fact table
  * working / narrative snapshots

Why SQLite over a separate vector DB:
  * one file = one artifact the user owns
  * fast enough for the per-user scale we'll see in years (< 1M episodes)
  * trivially inspectable, backupable, exportable
  * we sidestep the operational complexity of a vector DB for a use case
    that doesn't actually need it

When per-user scale crosses ~1M episodes (in years) we'll migrate to
LanceDB or pgvector. Until then this is the right tool.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from pmc.memory.recall.schema import (
    Episode,
    EpisodeKind,
    Fact,
    NarrativeSnapshot,
    WorkingMemorySnapshot,
)


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    time_start INTEGER NOT NULL,    -- unix ms
    time_end INTEGER,
    place_id TEXT,
    participant_ids TEXT NOT NULL,  -- JSON array
    raw_source TEXT NOT NULL,
    raw_pointers TEXT NOT NULL,     -- JSON array
    summary TEXT,
    summary_model TEXT,
    topics TEXT NOT NULL DEFAULT '[]',
    emotional_tone TEXT,
    importance REAL NOT NULL DEFAULT 0.5,
    ingestion_time INTEGER NOT NULL,
    consolidation_time INTEGER
);
CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(time_start DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_kind ON episodes(kind);
CREATE INDEX IF NOT EXISTS idx_episodes_pending ON episodes(consolidation_time)
    WHERE consolidation_time IS NULL;

CREATE TABLE IF NOT EXISTS episode_embeddings (
    episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    vec BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL,
    computed_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
    summary,
    topics,
    raw_text,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS episode_entities (
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    role TEXT,
    PRIMARY KEY (episode_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_ep_ents_entity ON episode_entities(entity_id);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    object_kind TEXT NOT NULL DEFAULT 'literal',
    confidence REAL NOT NULL DEFAULT 0.7,
    valid_from INTEGER,
    valid_until INTEGER,
    invalidated_by TEXT,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    ingestion_time INTEGER NOT NULL,
    summary_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(subject_id) WHERE valid_until IS NULL;

CREATE TABLE IF NOT EXISTS entity_importance (
    entity_id TEXT PRIMARY KEY,
    score REAL NOT NULL,
    last_accessed INTEGER,
    computed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS working_memory_snapshot (
    snapshot_date INTEGER PRIMARY KEY,
    payload TEXT NOT NULL,
    produced_by TEXT NOT NULL,
    produced_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS narrative_snapshot (
    snapshot_month TEXT PRIMARY KEY,  -- 'YYYY-MM'
    payload TEXT NOT NULL,
    produced_by TEXT NOT NULL,
    produced_at INTEGER NOT NULL
);
"""


def _ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _dt(ms: Optional[int]) -> Optional[datetime]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _pack_vector(vec: list[float] | np.ndarray) -> bytes:
    if isinstance(vec, np.ndarray):
        vec = vec.astype(np.float32).tolist()
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vector(blob: bytes, dim: int) -> np.ndarray:
    return np.array(struct.unpack(f"<{dim}f", blob), dtype=np.float32)


class RecallStore:
    """Per-user, SQLite-backed recall store.

    A `RecallStore` instance is cheap. Open one per user when handling
    a request; close it after. Don't share connections across threads
    (SQLite's connection objects aren't thread-safe by default).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RecallStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ episodes

    def upsert_episode(self, episode: Episode, raw_text_for_fts: str = "") -> None:
        self.conn.execute(
            """
            INSERT INTO episodes (
                id, kind, time_start, time_end, place_id, participant_ids,
                raw_source, raw_pointers, summary, summary_model, topics,
                emotional_tone, importance, ingestion_time, consolidation_time
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind,
                time_start=excluded.time_start,
                time_end=excluded.time_end,
                place_id=excluded.place_id,
                participant_ids=excluded.participant_ids,
                raw_source=excluded.raw_source,
                raw_pointers=excluded.raw_pointers,
                summary=COALESCE(excluded.summary, episodes.summary),
                summary_model=COALESCE(excluded.summary_model, episodes.summary_model),
                topics=excluded.topics,
                emotional_tone=COALESCE(excluded.emotional_tone, episodes.emotional_tone),
                importance=excluded.importance,
                consolidation_time=COALESCE(excluded.consolidation_time, episodes.consolidation_time)
            """,
            (
                episode.id,
                episode.kind.value,
                _ms(episode.time_start),
                _ms(episode.time_end),
                episode.place_id,
                json.dumps(episode.participant_ids),
                episode.raw_source,
                json.dumps(episode.raw_pointers),
                episode.summary,
                episode.summary_model,
                json.dumps(episode.topics),
                episode.emotional_tone,
                episode.importance,
                _ms(episode.ingestion_time),
                _ms(episode.consolidation_time),
            ),
        )
        # FTS row (replace by deleting + reinserting).
        self.conn.execute("DELETE FROM episode_fts WHERE rowid = (SELECT rowid FROM episodes WHERE id = ?)", (episode.id,))
        if episode.summary or raw_text_for_fts:
            self.conn.execute(
                """
                INSERT INTO episode_fts (rowid, summary, topics, raw_text)
                SELECT rowid, ?, ?, ? FROM episodes WHERE id = ?
                """,
                (
                    episode.summary or "",
                    " ".join(episode.topics),
                    raw_text_for_fts,
                    episode.id,
                ),
            )

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        row = self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return _row_to_episode(row) if row else None

    def pending_consolidation(self, limit: int = 200, include_preview: bool = True) -> list[Episode]:
        """Episodes that need consolidation by a frontier model.

        Includes both freshly-migrated episodes (consolidation_time IS
        NULL) and ones consolidated only with a preview / heuristic
        model — those still benefit from a real LLM pass to populate
        facts and state changes.
        """
        if include_preview:
            sql = """
                SELECT * FROM episodes
                WHERE consolidation_time IS NULL
                   OR summary_model LIKE 'preview/%'
                ORDER BY time_start DESC LIMIT ?
            """
        else:
            sql = "SELECT * FROM episodes WHERE consolidation_time IS NULL ORDER BY time_start DESC LIMIT ?"
        rows = self.conn.execute(sql, (limit,)).fetchall()
        return [_row_to_episode(r) for r in rows]

    def recent_episodes(self, limit: int = 50) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY time_start DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def episodes_for_entity(self, entity_id: str, limit: int = 100) -> list[Episode]:
        rows = self.conn.execute(
            """
            SELECT e.* FROM episodes e
            JOIN episode_entities ee ON ee.episode_id = e.id
            WHERE ee.entity_id = ?
            ORDER BY e.time_start DESC
            LIMIT ?
            """,
            (entity_id, limit),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    # ------------------------------------------------------------------ embeddings

    def set_embedding(self, episode_id: str, vec: list[float] | np.ndarray, model: str) -> None:
        arr = np.asarray(vec, dtype=np.float32)
        # Normalize so cosine = dot product downstream.
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        self.conn.execute(
            """
            INSERT INTO episode_embeddings (episode_id, vec, dim, model, computed_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(episode_id) DO UPDATE SET
                vec=excluded.vec, dim=excluded.dim, model=excluded.model, computed_at=excluded.computed_at
            """,
            (episode_id, _pack_vector(arr.tolist()), int(arr.shape[0]), model, int(time.time() * 1000)),
        )

    def all_embeddings(self) -> Iterable[tuple[str, np.ndarray]]:
        """Stream (episode_id, vector). Used by retrieval; not for huge stores."""
        rows = self.conn.execute("SELECT episode_id, vec, dim FROM episode_embeddings").fetchall()
        for r in rows:
            yield r["episode_id"], _unpack_vector(r["vec"], r["dim"])

    # ------------------------------------------------------------------ entities

    def link_entities(self, episode_id: str, entities: list[tuple[str, str]]) -> None:
        """`entities` is a list of (entity_id, role)."""
        self.conn.execute("DELETE FROM episode_entities WHERE episode_id = ?", (episode_id,))
        if not entities:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO episode_entities (episode_id, entity_id, role) VALUES (?,?,?)",
            [(episode_id, eid, role) for eid, role in entities],
        )

    # ------------------------------------------------------------------ facts (bi-temporal)

    def upsert_fact(self, fact: Fact) -> None:
        self.conn.execute(
            """
            INSERT INTO facts (
                id, subject_id, predicate, object_value, object_kind, confidence,
                valid_from, valid_until, invalidated_by, source_episode_ids,
                ingestion_time, summary_model
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                valid_until=excluded.valid_until,
                invalidated_by=excluded.invalidated_by,
                confidence=excluded.confidence,
                source_episode_ids=excluded.source_episode_ids
            """,
            (
                fact.id,
                fact.subject_id,
                fact.predicate,
                fact.object_value,
                fact.object_kind,
                fact.confidence,
                _ms(fact.valid_from),
                _ms(fact.valid_until),
                fact.invalidated_by,
                json.dumps(fact.source_episode_ids),
                _ms(fact.ingestion_time),
                fact.summary_model,
            ),
        )

    def active_facts_for(self, subject_id: str) -> list[Fact]:
        """Facts that are currently true (valid_until IS NULL)."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE subject_id = ? AND valid_until IS NULL",
            (subject_id,),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def all_active_facts(self) -> list[Fact]:
        rows = self.conn.execute("SELECT * FROM facts WHERE valid_until IS NULL").fetchall()
        return [_row_to_fact(r) for r in rows]

    def invalidate_fact(self, fact_id: str, invalidated_by: str, at: datetime) -> None:
        self.conn.execute(
            "UPDATE facts SET valid_until = ?, invalidated_by = ? WHERE id = ?",
            (_ms(at), invalidated_by, fact_id),
        )

    # ------------------------------------------------------------------ working / narrative

    def set_working_memory(self, snap: WorkingMemorySnapshot) -> None:
        self.conn.execute(
            """
            INSERT INTO working_memory_snapshot (snapshot_date, payload, produced_by, produced_at)
            VALUES (?,?,?,?)
            ON CONFLICT(snapshot_date) DO UPDATE SET
                payload=excluded.payload, produced_by=excluded.produced_by, produced_at=excluded.produced_at
            """,
            (
                _ms(snap.snapshot_date),
                snap.model_dump_json(),
                snap.produced_by,
                _ms(snap.produced_at),
            ),
        )

    def latest_working_memory(self) -> Optional[WorkingMemorySnapshot]:
        row = self.conn.execute(
            "SELECT payload FROM working_memory_snapshot ORDER BY snapshot_date DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return WorkingMemorySnapshot.model_validate_json(row["payload"])

    def set_narrative(self, snap: NarrativeSnapshot) -> None:
        self.conn.execute(
            """
            INSERT INTO narrative_snapshot (snapshot_month, payload, produced_by, produced_at)
            VALUES (?,?,?,?)
            ON CONFLICT(snapshot_month) DO UPDATE SET
                payload=excluded.payload, produced_by=excluded.produced_by, produced_at=excluded.produced_at
            """,
            (
                snap.snapshot_month,
                snap.model_dump_json(),
                snap.produced_by,
                _ms(snap.produced_at),
            ),
        )

    def latest_narrative(self) -> Optional[NarrativeSnapshot]:
        row = self.conn.execute(
            "SELECT payload FROM narrative_snapshot ORDER BY snapshot_month DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return NarrativeSnapshot.model_validate_json(row["payload"])

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        cur = self.conn.execute
        return {
            "episodes": cur("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "embedded": cur("SELECT COUNT(*) FROM episode_embeddings").fetchone()[0],
            "facts_active": cur("SELECT COUNT(*) FROM facts WHERE valid_until IS NULL").fetchone()[0],
            "facts_superseded": cur("SELECT COUNT(*) FROM facts WHERE valid_until IS NOT NULL").fetchone()[0],
            "entity_links": cur("SELECT COUNT(*) FROM episode_entities").fetchone()[0],
            "consolidated_by_preview": cur("SELECT COUNT(*) FROM episodes WHERE summary_model LIKE 'preview/%'").fetchone()[0],
            "consolidated_by_claude": cur("SELECT COUNT(*) FROM episodes WHERE summary_model IS NOT NULL AND summary_model NOT LIKE 'preview/%'").fetchone()[0],
            "pending_consolidation": cur(
                "SELECT COUNT(*) FROM episodes WHERE consolidation_time IS NULL OR summary_model LIKE 'preview/%'"
            ).fetchone()[0],
        }


def _row_to_episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        kind=EpisodeKind(row["kind"]),
        time_start=_dt(row["time_start"]),
        time_end=_dt(row["time_end"]),
        place_id=row["place_id"],
        participant_ids=json.loads(row["participant_ids"]) if row["participant_ids"] else [],
        raw_source=row["raw_source"],
        raw_pointers=json.loads(row["raw_pointers"]) if row["raw_pointers"] else [],
        summary=row["summary"],
        summary_model=row["summary_model"],
        topics=json.loads(row["topics"]) if row["topics"] else [],
        emotional_tone=row["emotional_tone"],
        importance=row["importance"],
        ingestion_time=_dt(row["ingestion_time"]),
        consolidation_time=_dt(row["consolidation_time"]),
    )


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        subject_id=row["subject_id"],
        predicate=row["predicate"],
        object_value=row["object_value"],
        object_kind=row["object_kind"],
        confidence=row["confidence"],
        valid_from=_dt(row["valid_from"]),
        valid_until=_dt(row["valid_until"]),
        invalidated_by=row["invalidated_by"],
        source_episode_ids=json.loads(row["source_episode_ids"]) if row["source_episode_ids"] else [],
        ingestion_time=_dt(row["ingestion_time"]),
        summary_model=row["summary_model"],
    )
