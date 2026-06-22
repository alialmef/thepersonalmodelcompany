"""Temporal status — was this entity touched recently?

The design principle: big capacity, small attention. The agent holds
everything; it speaks like a person — only what's currently in motion
shapes the voice. Status tags here are the lens that lets the model
distinguish "you build PMC" (active) from "you record voice memos"
(dormant) without us discarding the dormant data.

Each entity gets ONE of:
   active  — touched in the last ACTIVE_WINDOW_DAYS
   stable  — touched in last STABLE_WINDOW_DAYS, but not active
   new     — first appearance in last 7 days
   dormant — older than STABLE_WINDOW_DAYS

The chat context block uses these to split a two-band layout. The
opener prompt is told to only draw "current behavior" guesses from
active entries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ACTIVE_WINDOW_DAYS = 30
STABLE_WINDOW_DAYS = 90
NEW_WINDOW_DAYS = 7


def status_for(entity: dict[str, Any], *, kind: str) -> str:
    """Best-effort temporal classification of one entity dict (the
    shape produced by `summarize_for_agent` and the synthesis loaders).
    Returns one of: 'active', 'stable', 'new', 'dormant', 'unknown'.
    """
    now = datetime.now(timezone.utc)
    last = _last_touch(entity, kind)
    first = _first_seen(entity, kind)

    if last is None and first is None:
        return "unknown"

    if last is None:
        last = first

    days_since = (now - last).days

    if days_since <= NEW_WINDOW_DAYS and first is not None:
        if (now - first).days <= NEW_WINDOW_DAYS:
            return "new"

    if days_since <= ACTIVE_WINDOW_DAYS:
        return "active"
    if days_since <= STABLE_WINDOW_DAYS:
        return "stable"
    return "dormant"


def _last_touch(e: dict[str, Any], kind: str) -> datetime | None:
    """The most-recent-activity timestamp for this entity, depending
    on kind. Different kinds use different field names; we try the
    most common ones."""
    candidates = (
        "last_seen", "last_touched", "last_commit", "last_modified",
        "modified", "last_activity", "last_visit", "ts",
    )
    for k in candidates:
        v = e.get(k)
        dt = _parse_iso(v)
        if dt is not None:
            return dt
    return None


def _first_seen(e: dict[str, Any], kind: str) -> datetime | None:
    candidates = ("first_seen", "created_at", "opened_at", "added_at")
    for k in candidates:
        v = e.get(k)
        dt = _parse_iso(v)
        if dt is not None:
            return dt
    return None


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


# Quick predicate used by the context builder.
def is_active(entity: dict[str, Any], *, kind: str) -> bool:
    s = status_for(entity, kind=kind)
    return s in ("active", "new")


__all__ = [
    "ACTIVE_WINDOW_DAYS", "STABLE_WINDOW_DAYS", "NEW_WINDOW_DAYS",
    "status_for", "is_active",
]
