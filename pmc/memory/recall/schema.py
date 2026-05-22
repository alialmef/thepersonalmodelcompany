"""Typed schemas for the recall layer.

We use pydantic for the public surface (consistent with the rest of pmc)
and plain SQL for the on-disk representation. The two are kept in sync
by the `RecallStore` (see `store.py`).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EpisodeKind(str, Enum):
    """High-level shape of one episode.

    Episodes are the unit of recall. The kind drives both how the
    consolidation model is prompted and how the retrieval API filters.
    """

    # iMessage / SMS / chat thread cluster (same partner, ~24h window)
    conversation = "conversation"
    # Apple Notes / TextEdit document authored or substantially edited
    note_authored = "note_authored"
    # Calendar event that actually occurred (past or imminent)
    calendar_event = "calendar_event"
    # Photo cluster — same place, same day, > N photos
    photo_cluster = "photo_cluster"
    # FaceTime / phone call long enough to be meaningful (> 60s)
    call = "call"
    # Mail correspondence cluster
    mail_exchange = "mail_exchange"
    # Code commit burst (git log over a focused timeframe)
    code_burst = "code_burst"
    # Browser research session (multiple visits to one domain in a day)
    web_session = "web_session"
    # Synthesized: agent observed something during conversation
    agent_observed = "agent_observed"


class Episode(BaseModel):
    """A time-anchored slice of life — the unit of recall.

    Every Episode carries enough provenance to answer "where did the
    agent learn this from?" and enough structure to be retrieved by
    time, place, person, or semantic similarity.
    """

    id: str
    kind: EpisodeKind
    time_start: datetime
    time_end: Optional[datetime] = None
    place_id: Optional[str] = None
    participant_ids: list[str] = Field(default_factory=list)
    raw_source: str  # 'imessage', 'photos', 'notes', etc.
    raw_pointers: list[dict] = Field(default_factory=list)
    # Set during consolidation
    summary: Optional[str] = None
    summary_model: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    emotional_tone: Optional[str] = None
    importance: float = 0.5  # 0-1
    ingestion_time: datetime
    consolidation_time: Optional[datetime] = None


class Fact(BaseModel):
    """A bi-temporal fact about an entity.

    Layer 3 of the memory architecture. Each Fact has both an event time
    (`valid_from` / `valid_until`) and an ingestion time. State changes
    *supersede* rather than overwrite — when Sarah moves from Brooklyn
    to LA, we keep the Brooklyn fact with `valid_until = move_date` and
    create a new LA fact.
    """

    id: str
    subject_id: str           # Person / Place / Project entity id
    predicate: str            # 'lives_in', 'works_at', 'pregnant_with', 'feels_about', etc.
    object_value: str         # entity_id or literal
    object_kind: str = "literal"  # 'entity' | 'literal'
    confidence: float = 0.7
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    invalidated_by: Optional[str] = None
    source_episode_ids: list[str] = Field(default_factory=list)
    ingestion_time: datetime
    summary_model: Optional[str] = None  # which model produced this fact


class MemoryFragment(BaseModel):
    """One ranked chunk of memory returned by `retrieve()`.

    Carries enough structure that the agent can decide whether to
    surface it verbatim, paraphrase it, or just use it as latent
    context. Always includes provenance back to source episodes so
    the agent can ground claims.
    """

    episode_id: str
    summary: str
    score: float                  # fused multi-signal score, 0-1
    time_start: datetime
    time_end: Optional[datetime] = None
    participants: list[str] = Field(default_factory=list)  # display names
    topics: list[str] = Field(default_factory=list)
    source: str
    raw_pointers: list[dict] = Field(default_factory=list)
    # Components of the fused score, for debugging / tuning
    vector_score: float = 0.0
    bm25_score: float = 0.0
    entity_score: float = 0.0
    recency_boost: float = 0.0
    working_memory_boost: float = 0.0


class WorkingMemorySnapshot(BaseModel):
    """Daily snapshot of what's live in the user's life right now.

    Refreshed by the consolidation worker each night. The agent reads
    this on every turn as base context, before any per-query retrieval.
    """

    snapshot_date: datetime
    top_open_loops: list[dict] = Field(default_factory=list)
    hot_people: list[dict] = Field(default_factory=list)
    rising_themes: list[dict] = Field(default_factory=list)
    upcoming_events: list[dict] = Field(default_factory=list)
    recent_episodes: list[dict] = Field(default_factory=list)
    anticipation: list[str] = Field(default_factory=list)  # 3-5 proactive items
    produced_by: str  # which model assembled this
    produced_at: datetime


class NarrativeSnapshot(BaseModel):
    """Monthly snapshot of the user's life arc.

    Eras, identity, trajectories. Refreshed monthly because narrative
    shifts slowly and full recomputation is expensive.
    """

    snapshot_month: str  # 'YYYY-MM'
    eras: list[dict] = Field(default_factory=list)  # [{label, start, end, themes, key_people, summary}]
    current_era: Optional[dict] = None
    identity_arcs: list[dict] = Field(default_factory=list)  # long-running threads
    trajectory_notes: list[str] = Field(default_factory=list)
    produced_by: str
    produced_at: datetime
