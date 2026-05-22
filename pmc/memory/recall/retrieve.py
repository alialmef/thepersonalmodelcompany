"""Multi-signal retrieval — the function the agent calls.

Combines four signals, each normalized to [0, 1] then fused with
configurable weights:

    final = w_vec * vector_sim
          + w_bm25 * bm25_score
          + w_entity * entity_link_boost
          + w_recency * recency_decay
          + w_working * working_memory_boost

We follow Mem0's finding that no single signal carries the whole load —
a query like "the deal with Sarah" needs entity-link (Sarah) AND
keyword (deal) AND vector (semantic similarity); none of those alone
returns the right episode.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from pmc.memory.recall.embed import LocalEmbedder
from pmc.memory.recall.schema import MemoryFragment
from pmc.memory.recall.store import RecallStore


@dataclass
class RetrievalScope:
    """Filters that narrow the search before ranking.

    All filters are optional and AND-combined. `entity_ids` boosts
    rather than restricts — the agent often wants "episodes touching
    Sarah" weighted higher but not exclusively.
    """

    entity_ids: list[str] | None = None              # boost (not filter)
    must_entity_ids: list[str] | None = None         # hard filter
    time_after: Optional[datetime] = None
    time_before: Optional[datetime] = None
    kinds: list[str] | None = None
    min_importance: float = 0.0


# Default weights — tuned by intuition; will refine against held-out
# queries once we have a benchmark suite.
W_VEC      = 0.45
W_BM25     = 0.25
W_ENTITY   = 0.15
W_RECENCY  = 0.10
W_WORKING  = 0.05


def retrieve(
    store: RecallStore,
    query: str,
    scope: Optional[RetrievalScope] = None,
    k: int = 10,
) -> list[MemoryFragment]:
    """Top-k MemoryFragments matching `query` within `scope`.

    Stateless apart from the store. Safe to call from many threads as
    long as each has its own RecallStore connection.
    """
    scope = scope or RetrievalScope()

    # ---- candidate set (pre-filter by hard scope) -----------------------
    where, params = _build_scope_where(scope)
    rows = store.conn.execute(
        f"""
        SELECT id, kind, summary, topics, time_start, time_end, participant_ids,
               importance, raw_source, raw_pointers
        FROM episodes
        WHERE summary IS NOT NULL
        {where}
        ORDER BY time_start DESC
        LIMIT 2000
        """,
        params,
    ).fetchall()
    if not rows:
        return []

    candidate_ids = [r["id"] for r in rows]

    # ---- signal 1: vector similarity ------------------------------------
    q_vec = LocalEmbedder.embed_one(query)
    vec_scores: dict[str, float] = {}
    for epid, ev in store.all_embeddings():
        if epid in candidate_ids and ev.shape == q_vec.shape:
            vec_scores[epid] = float(np.dot(q_vec, ev))

    # ---- signal 2: BM25 / FTS5 ------------------------------------------
    fts_query = _to_fts_query(query)
    bm25_scores: dict[str, float] = {}
    if fts_query:
        # bm25() returns lower-is-better; flip + normalize.
        fts_rows = store.conn.execute(
            """
            SELECT e.id AS id, bm25(episode_fts) AS rank
            FROM episode_fts
            JOIN episodes e ON e.rowid = episode_fts.rowid
            WHERE episode_fts MATCH ?
            LIMIT 500
            """,
            (fts_query,),
        ).fetchall()
        if fts_rows:
            ranks = [r["rank"] for r in fts_rows]
            r_min, r_max = min(ranks), max(ranks)
            spread = (r_max - r_min) or 1.0
            for r in fts_rows:
                bm25_scores[r["id"]] = 1.0 - ((r["rank"] - r_min) / spread)

    # ---- signal 3: entity link boost ------------------------------------
    entity_scores: dict[str, float] = {}
    if scope.entity_ids:
        marks = ",".join("?" for _ in scope.entity_ids)
        ent_rows = store.conn.execute(
            f"""
            SELECT episode_id, COUNT(*) AS hits
            FROM episode_entities
            WHERE entity_id IN ({marks})
            GROUP BY episode_id
            """,
            tuple(scope.entity_ids),
        ).fetchall()
        for r in ent_rows:
            entity_scores[r["episode_id"]] = min(1.0, r["hits"] / 3.0)

    # ---- signal 4: recency decay (half-life 14 days) --------------------
    # Clamp age to non-negative: future-dated episodes (calendar items
    # that haven't happened yet) get *full* recency boost, not infinite.
    # Without the clamp `exp(-negative)` explodes to astronomical values
    # and a single far-future episode drowns out everything else.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    recency_scores: dict[str, float] = {}
    for r in rows:
        age_days = max(0.0, (now_ms - r["time_start"]) / (1000 * 60 * 60 * 24))
        recency_scores[r["id"]] = math.exp(-age_days / 14.0)

    # ---- signal 5: working memory boost --------------------------------
    wm_boost_ids = _working_memory_episode_ids(store)
    wm_scores = {epid: 1.0 for epid in wm_boost_ids if epid in candidate_ids}

    # ---- fuse + rank ----------------------------------------------------
    fragments: list[MemoryFragment] = []
    for r in rows:
        epid = r["id"]
        v = vec_scores.get(epid, 0.0)
        b = bm25_scores.get(epid, 0.0)
        e = entity_scores.get(epid, 0.0)
        rcy = recency_scores.get(epid, 0.0)
        wm = wm_scores.get(epid, 0.0)
        score = (
            W_VEC     * v
            + W_BM25    * b
            + W_ENTITY  * e
            + W_RECENCY * rcy
            + W_WORKING * wm
        ) * (0.5 + 0.5 * r["importance"])
        # Skip candidates with zero positive signal — saves printing
        # 2000 noise rows in the result.
        if score <= 0:
            continue
        import json
        fragments.append(MemoryFragment(
            episode_id=epid,
            summary=r["summary"] or "",
            score=float(score),
            time_start=datetime.fromtimestamp(r["time_start"]/1000.0, tz=timezone.utc),
            time_end=datetime.fromtimestamp(r["time_end"]/1000.0, tz=timezone.utc) if r["time_end"] else None,
            participants=json.loads(r["participant_ids"]) if r["participant_ids"] else [],
            topics=json.loads(r["topics"]) if r["topics"] else [],
            source=r["raw_source"],
            raw_pointers=json.loads(r["raw_pointers"]) if r["raw_pointers"] else [],
            vector_score=v,
            bm25_score=b,
            entity_score=e,
            recency_boost=rcy,
            working_memory_boost=wm,
        ))

    fragments.sort(key=lambda f: f.score, reverse=True)
    return fragments[:k]


# ----------------------------------------------------------------------


def _build_scope_where(scope: RetrievalScope) -> tuple[str, tuple]:
    clauses: list[str] = []
    params: list = []
    if scope.time_after:
        clauses.append("time_start >= ?")
        params.append(int(scope.time_after.timestamp() * 1000))
    if scope.time_before:
        clauses.append("time_start <= ?")
        params.append(int(scope.time_before.timestamp() * 1000))
    if scope.kinds:
        marks = ",".join("?" for _ in scope.kinds)
        clauses.append(f"kind IN ({marks})")
        params.extend(scope.kinds)
    if scope.min_importance > 0:
        clauses.append("importance >= ?")
        params.append(float(scope.min_importance))
    if scope.must_entity_ids:
        # Restrict to episodes touching ALL must-entities.
        for eid in scope.must_entity_ids:
            clauses.append(
                "id IN (SELECT episode_id FROM episode_entities WHERE entity_id = ?)"
            )
            params.append(eid)
    return ("AND " + " AND ".join(clauses)) if clauses else "", tuple(params)


def _to_fts_query(query: str) -> str:
    # FTS5 wants MATCH terms separated by spaces; strip operator chars.
    tokens = re.findall(r"[A-Za-z0-9']+", query)
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens[:8])


def _working_memory_episode_ids(store: RecallStore) -> list[str]:
    snap = store.latest_working_memory()
    if not snap:
        return []
    ids: list[str] = []
    for item in snap.recent_episodes:
        if isinstance(item, dict) and "episode_id" in item:
            ids.append(item["episode_id"])
    return ids
