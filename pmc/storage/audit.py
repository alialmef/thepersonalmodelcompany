"""Cross-stage audit log — what happened to this user, when.

`pmc/train/bundle.py` carries a per-bundle audit log (events tied to one
training run). This module is the *user-level* audit: every event across
every run, written append-only as JSONL so it's safe to tail / read while
new events are being added.

Events are typed loosely (stage + event name + data dict) so any pipeline
stage can write to the log without coupling to a fixed schema. The point is
provenance — answering "what data went into this model?" and "what happened
to this user's account on 2026-01-15?" — not strict observability.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pmc.storage.paths import StoragePaths

KNOWN_STAGES = {
    "ingest",
    "curate",
    "train",
    "eval",
    "gate",
    "deploy",
    "delete",
    "export",
    "user",
    "action",
    "memory",
}


class AuditEvent(BaseModel):
    """One event in the per-user audit log."""

    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    stage: str
    event: str
    run_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class AuditLog:
    """Append-only event log, one file per user."""

    def __init__(self, root: Path | str) -> None:
        self.paths = StoragePaths(root)

    def log(
        self,
        user_id: str,
        stage: str,
        event: str,
        *,
        run_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> AuditEvent:
        record = AuditEvent(
            user_id=user_id,
            stage=stage,
            event=event,
            run_id=run_id,
            data=data or {},
        )
        path = self.paths.audit_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json() + "\n"
        # Append atomically — open in append mode flushes per-line on most systems.
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return record

    def events(
        self,
        user_id: str,
        *,
        stage: str | None = None,
        event: str | None = None,
        run_id: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Read events, optionally filtered."""
        path = self.paths.audit_file(user_id)
        if not path.is_file():
            return []
        out: list[AuditEvent] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = AuditEvent.model_validate_json(line)
                except Exception:
                    continue
                if stage is not None and record.stage != stage:
                    continue
                if event is not None and record.event != event:
                    continue
                if run_id is not None and record.run_id != run_id:
                    continue
                if since is not None and record.timestamp < since:
                    continue
                out.append(record)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def events_for_run(self, user_id: str, run_id: str) -> list[AuditEvent]:
        return self.events(user_id, run_id=run_id)

    def latest(self, user_id: str, n: int = 20) -> list[AuditEvent]:
        """Newest `n` events first."""
        all_events = self.events(user_id)
        return list(reversed(all_events[-n:]))

    def clear(self, user_id: str) -> bool:
        """Truncate the audit log. Use only on hard delete."""
        path = self.paths.audit_file(user_id)
        if not path.is_file():
            return False
        path.unlink()
        return True


__all__ = ["AuditEvent", "AuditLog", "KNOWN_STAGES"]
