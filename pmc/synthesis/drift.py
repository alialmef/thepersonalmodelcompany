"""Drift detection over the graph.

Patterns name the steady state. Drift names what's different about
the present versus the recent past.

Each drift is a structured observation: a category, a verbal
summary, and a quantitative delta. Examples (real-shape):
  - "Last flight 9 days ago — Delta OLB → JFK (you were in Italy)."
  - "No commits in 4 days; usual cadence is 2-3/day on PMC."
  - "Last voice memo 5 days ago — normally weekly."

Pure compute, like patterns. No agent calls.

Output: <storage_root>/users/<uid>/graph/synth/drift.jsonl
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pmc.storage.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class Drift:
    id: str
    category: str           # "travel" | "events" | "creation" | "code" | "comms" | "places"
    headline: str           # second-person, "Last flight X days ago — Delta JFK→ATL"
    detail: str
    metric: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _synth_dir(storage_root: Path | str, user_id: str) -> Path:
    p = Path(storage_root) / "users" / user_id / "graph" / "synth"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _drift_path(storage_root: Path | str, user_id: str) -> Path:
    return _synth_dir(storage_root, user_id) / "drift.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _days_ago(dt: datetime, now: datetime) -> int:
    return max(0, (now - dt).days)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _travel_drift(graph_store: GraphStore, user_id: str, now: datetime) -> Drift | None:
    """Most recent Wallet boarding pass."""
    latest: tuple[datetime, str] | None = None
    for p in graph_store.iter_entities(user_id, "project"):
        sources = p.get("sources") or []
        if not any("wallet:boardingPass" in s for s in sources):
            continue
        dt = _parse_iso(p.get("last_activity"))
        if not dt:
            continue
        if latest is None or dt > latest[0]:
            latest = (dt, p.get("name") or "")
    if latest is None:
        return None
    dt, name = latest
    days = _days_ago(dt, now)
    headline = f"Last flight {days} days ago — {name}."
    detail = f"On record from Apple Wallet boarding pass dated {dt.strftime('%Y-%m-%d')}."
    return Drift(
        id=f"drift_travel_{days}d",
        category="travel",
        headline=headline,
        detail=detail,
        metric={"days_ago": days, "name": name, "date": dt.isoformat()},
    )


def _events_drift(graph_store: GraphStore, user_id: str, now: datetime) -> Drift | None:
    """Most recent ticketed event."""
    latest: tuple[datetime, str] | None = None
    for p in graph_store.iter_entities(user_id, "project"):
        sources = p.get("sources") or []
        if not any("wallet:eventTicket" in s for s in sources):
            continue
        dt = _parse_iso(p.get("last_activity"))
        if not dt:
            continue
        if latest is None or dt > latest[0]:
            latest = (dt, p.get("name") or "")
    if latest is None:
        return None
    dt, name = latest
    days = _days_ago(dt, now)
    headline = f"Last event {days} days ago — {name}."
    return Drift(
        id=f"drift_events_{days}d",
        category="events",
        headline=headline,
        detail=f"Wallet event ticket dated {dt.strftime('%Y-%m-%d')}.",
        metric={"days_ago": days, "name": name, "date": dt.isoformat()},
    )


def _creation_drift(
    graph_store: GraphStore, storage_root: Path | str, user_id: str, now: datetime,
) -> Drift | None:
    """Most recent voice memo (file mtime)."""
    latest_dt: datetime | None = None
    latest_name = ""
    for f in graph_store.iter_entities(user_id, "file"):
        if f.get("kind") != "voice_memo":
            continue
        dt = _parse_iso(f.get("modified"))
        if not dt:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_name = f.get("name") or ""
    if latest_dt is None:
        return None
    days = _days_ago(latest_dt, now)
    headline = f"Last voice memo {days} days ago."
    return Drift(
        id=f"drift_creation_{days}d",
        category="creation",
        headline=headline,
        detail=f"File: {latest_name}",
        metric={"days_ago": days, "filename": latest_name, "date": latest_dt.isoformat()},
    )


def _code_drift(graph_store: GraphStore, user_id: str, now: datetime) -> Drift | None:
    """Most recent commit across all tracked CodeRepos."""
    latest: tuple[datetime, str, int] | None = None  # dt, name, count30d
    for r in graph_store.iter_entities(user_id, "repo"):
        dt = _parse_iso(r.get("last_commit"))
        if not dt:
            continue
        if latest is None or dt > latest[0]:
            latest = (dt, r.get("name") or "?", int(r.get("commit_count_30d") or 0))
    if latest is None:
        return None
    dt, name, count = latest
    days = _days_ago(dt, now)
    headline = f"Last commit {days} days ago — {name}."
    detail = (
        f"{count} commits across all repos in the last 30 days."
        if count else f"On {name} — repo with sparse recent activity."
    )
    return Drift(
        id=f"drift_code_{days}d",
        category="code",
        headline=headline,
        detail=detail,
        metric={"days_ago": days, "repo": name, "commit_count_30d": count},
    )


def _comms_drift(graph_store: GraphStore, user_id: str, now: datetime) -> Drift | None:
    """Most recent live open loop — proxy for last meaningful inbound."""
    latest: tuple[datetime, str] | None = None
    for l in graph_store.iter_entities(user_id, "open_loop"):
        if "voice_memo_transcript" in (l.get("sources") or []):
            continue
        dt = _parse_iso(l.get("last_touched")) or _parse_iso(l.get("opened_at"))
        if not dt:
            continue
        if latest is None or dt > latest[0]:
            ex = (l.get("excerpt") or "").strip()
            latest = (dt, ex)
    if latest is None:
        return None
    dt, ex = latest
    days = _days_ago(dt, now)
    snippet = ex[:60] + ("…" if len(ex) > 60 else "")
    headline = f"Last unanswered message {days} days ago."
    return Drift(
        id=f"drift_comms_{days}d",
        category="comms",
        headline=headline,
        detail=f"“{snippet}”",
        metric={"days_ago": days, "excerpt": ex[:200], "date": dt.isoformat()},
    )


def _location_drift(graph_store: GraphStore, user_id: str, now: datetime) -> Drift | None:
    """Most recent Wallet venue place visit."""
    latest: tuple[datetime, str] | None = None
    for p in graph_store.iter_entities(user_id, "place"):
        if "wallet" not in (p.get("sources") or []):
            continue
        dt = _parse_iso(p.get("last_seen"))
        if not dt:
            continue
        if latest is None or dt > latest[0]:
            latest = (dt, p.get("label") or "?")
    if latest is None:
        return None
    dt, label = latest
    days = _days_ago(dt, now)
    headline = f"Last venue {days} days ago — {label}."
    return Drift(
        id=f"drift_places_{days}d",
        category="places",
        headline=headline,
        detail=f"From Wallet pass location data.",
        metric={"days_ago": days, "label": label, "date": dt.isoformat()},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_drift(
    *,
    graph_store: GraphStore,
    storage_root: Path | str,
    user_id: str,
) -> list[Drift]:
    """Run all drift detectors, persist, return."""
    now = datetime.now(timezone.utc)
    drifts: list[Drift] = []
    for detector in [
        lambda gs, uid, n: _travel_drift(gs, uid, n),
        lambda gs, uid, n: _events_drift(gs, uid, n),
        lambda gs, uid, n: _creation_drift(gs, storage_root, uid, n),
        lambda gs, uid, n: _code_drift(gs, uid, n),
        lambda gs, uid, n: _comms_drift(gs, uid, n),
        lambda gs, uid, n: _location_drift(gs, uid, n),
    ]:
        try:
            d = detector(graph_store, user_id, now)
        except Exception:
            continue
        if d is not None:
            drifts.append(d)
    drifts.sort(key=lambda d: d.metric.get("days_ago", 999))
    _write_drift(storage_root, user_id, drifts)
    return drifts


def load_drift(storage_root: Path | str, user_id: str) -> list[Drift]:
    p = _drift_path(storage_root, user_id)
    if not p.is_file():
        return []
    out: list[Drift] = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(Drift(**d))
        except Exception:
            continue
    return out


def _write_drift(storage_root: Path | str, user_id: str, drifts: list[Drift]) -> None:
    import os
    p = _drift_path(storage_root, user_id)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        for d in drifts:
            f.write(json.dumps(asdict(d), default=str) + "\n")
    os.replace(tmp, p)
