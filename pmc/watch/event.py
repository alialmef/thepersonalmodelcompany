"""The Event + Decision types that flow through the gate.

An Event is "something happened on this Mac." It has a source (which
OS hook detected it), a kind (what shape of change), a path or
identifier, and a timestamp. That's it — the classifier's job is to
look at this thin record and decide what to do.

Keep the event payload small. The full data (file contents, DB rows)
lives on disk; we don't carry it through the pipe. If the router
needs more it re-reads from the source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Source(str, Enum):
    """Where the event came from. Stays in sync with the source modules
    under `pmc/watch/sources/`."""
    FS = "fs"                   # filesystem event via watchdog/FSEvents
    SQLITE = "sqlite"           # SQLite WAL change watcher
    DISTNOTIF = "distnotif"     # NSDistributedNotificationCenter
    SHELL = "shell"             # zsh preexec hook (opt-in)
    POLL = "poll"               # periodic poll (fallback for things w/o hooks)


class Kind(str, Enum):
    """Shape of the change. Independent of source."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    DB_WRITE = "db_write"
    APP_EVENT = "app_event"


class Decision(str, Enum):
    """What the classifier decided to do with the event."""
    PROMOTE = "promote"   # run the relevant extractor now; write to graph
    DEFER = "defer"       # queue for the next consolidator / batch LLM pass
    DROP = "drop"         # not interesting; don't even queue


@dataclass(frozen=True)
class Event:
    source: Source
    kind: Kind
    path: str                       # absolute path, or DB URI, or notif name
    ts: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def short(self) -> str:
        """One-line representation for logs."""
        return f"{self.source.value}/{self.kind.value}  {self.path}"


@dataclass
class Classified:
    """An Event after the classifier has tagged it."""
    event: Event
    decision: Decision
    reason: str = ""        # short human-readable rationale
    extractor: Optional[str] = None  # for PROMOTE: which extractor to run
    confidence: float = 1.0          # rules = 1.0; LLM = its self-rated conf


__all__ = ["Source", "Kind", "Decision", "Event", "Classified"]
