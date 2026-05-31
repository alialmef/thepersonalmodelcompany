"""Per-user redactions + paused-sources store.

The redact + manage surface (`/knowledge-update` in the Mac app) lets a
user:
  * pause / resume specific data sources (no new ingest, existing rows
    stay until explicitly forgotten)
  * mark a Person / Topic / Date-range as private (the synthesis layer
    must exclude these when assembling agent context)
  * delete individual ingested items (raw + derived)
  * nuclear erase (delegated to DeletionManager)

This file owns the *non-content* state: which sources are paused, and
which redaction rules are in force. The actual delete-the-data work is
done elsewhere — for source delete via `UserStore.delete_source`, for
nuclear via `DeletionManager`, for per-item via a new
`UserStore.delete_raw_item` (not yet implemented; V1 surfaces the
private + pause primitives and lands per-item forget in a follow-up).

Storage: `{storage_root}/users/{user_id}/redactions.json` — a single
small JSON document per user. Atomic-enough for V0 single-process
serving; rewrite as a Postgres table when we need cross-process
concurrency.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PausedSource(BaseModel):
    source_id: str
    paused_at: datetime


class Redaction(BaseModel):
    """A user-declared exclusion. Three kinds:

      * `person`     — value is a normalized identifier (email, phone,
                       contact name). Anything attributed to this
                       identifier gets stripped from agent context.
      * `topic`      — value is a free-text label. Items whose text or
                       metadata match (substring, case-insensitive) get
                       stripped.
      * `date_range` — value is an ISO 8601 interval `"YYYY-MM-DD/YYYY-MM-DD"`.
                       Items whose timestamp falls inside the range get
                       stripped.
    """

    id: str
    kind: str        # "person" | "topic" | "date_range"
    value: str
    added_at: datetime
    note: Optional[str] = None  # optional human-readable why


class RedactionsState(BaseModel):
    paused_sources: list[PausedSource] = Field(default_factory=list)
    redactions: list[Redaction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _short_id() -> str:
    return secrets.token_urlsafe(9)


class RedactionsStore:
    """Per-user pause + redaction state. One JSON file per user."""

    def __init__(self, storage_root: Path | str) -> None:
        self.storage_root = Path(storage_root)

    # ---- internal ------------------------------------------------------

    def _path(self, user_id: str) -> Path:
        # Match the convention used by UserStore + ArtifactStore so a user's
        # entire footprint sits under one directory.
        p = self.storage_root / "users" / user_id / "redactions.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load(self, user_id: str) -> RedactionsState:
        p = self._path(user_id)
        if not p.is_file():
            return RedactionsState()
        try:
            raw = json.loads(p.read_text())
            return RedactionsState.model_validate(raw)
        except Exception:
            # Corrupt file — favor not losing the data. Move it aside
            # and start fresh; an operator can inspect later.
            try:
                p.rename(p.with_suffix(".corrupt"))
            except Exception:
                pass
            return RedactionsState()

    def _save(self, user_id: str, state: RedactionsState) -> None:
        p = self._path(user_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2))
        os.replace(tmp, p)

    # ---- reads ---------------------------------------------------------

    def state(self, user_id: str) -> RedactionsState:
        return self._load(user_id)

    def is_paused(self, user_id: str, source_id: str) -> bool:
        return any(s.source_id == source_id for s in self._load(user_id).paused_sources)

    def paused_source_ids(self, user_id: str) -> list[str]:
        return [s.source_id for s in self._load(user_id).paused_sources]

    def is_redacted_person(self, user_id: str, person_identifier: str) -> bool:
        pid = person_identifier.strip().lower()
        if not pid:
            return False
        for r in self._load(user_id).redactions:
            if r.kind == "person" and r.value.strip().lower() == pid:
                return True
        return False

    def is_redacted_topic(self, user_id: str, text: str) -> bool:
        """Returns True if `text` matches any redacted-topic substring
        (case-insensitive). Used by the synthesis layer when assembling
        agent context."""
        lowered = text.lower()
        for r in self._load(user_id).redactions:
            if r.kind == "topic" and r.value.lower() in lowered:
                return True
        return False

    # ---- pause / resume ------------------------------------------------

    def pause_source(self, user_id: str, source_id: str) -> RedactionsState:
        state = self._load(user_id)
        # Idempotent — pause is a set membership operation
        if not any(s.source_id == source_id for s in state.paused_sources):
            state.paused_sources.append(
                PausedSource(source_id=source_id, paused_at=_utcnow())
            )
            self._save(user_id, state)
        return state

    def resume_source(self, user_id: str, source_id: str) -> RedactionsState:
        state = self._load(user_id)
        before = len(state.paused_sources)
        state.paused_sources = [
            s for s in state.paused_sources if s.source_id != source_id
        ]
        if len(state.paused_sources) != before:
            self._save(user_id, state)
        return state

    # ---- redactions CRUD ----------------------------------------------

    def add_redaction(
        self,
        user_id: str,
        *,
        kind: str,
        value: str,
        note: Optional[str] = None,
    ) -> Redaction:
        if kind not in ("person", "topic", "date_range"):
            raise ValueError(f"unknown redaction kind: {kind!r}")
        if not value.strip():
            raise ValueError("redaction value must be non-empty")
        state = self._load(user_id)
        r = Redaction(
            id=_short_id(),
            kind=kind,
            value=value.strip(),
            added_at=_utcnow(),
            note=note,
        )
        state.redactions.append(r)
        self._save(user_id, state)
        return r

    def remove_redaction(self, user_id: str, redaction_id: str) -> bool:
        state = self._load(user_id)
        before = len(state.redactions)
        state.redactions = [r for r in state.redactions if r.id != redaction_id]
        changed = len(state.redactions) != before
        if changed:
            self._save(user_id, state)
        return changed

    def clear(self, user_id: str) -> None:
        """Wipe everything (called from the nuclear erase path)."""
        p = self._path(user_id)
        if p.is_file():
            p.unlink()
