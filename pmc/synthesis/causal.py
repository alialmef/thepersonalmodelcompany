"""Causal correlation detection over the graph.

Patterns name the steady state. Drift names what's most recent.
Causal observes correlations between categories — "when X happens,
Y tends to also." First-pass implementation. Pure compute, no agent.

These observations are *correlational*, not causal in the
philosophical sense. We surface them as "when X, Y" so the agent
can grow more nuanced as it accumulates more data.

Output: <storage_root>/users/<uid>/graph/synth/causal.jsonl
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pmc.storage.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class CausalObservation:
    id: str
    when: str               # human-readable trigger description
    then: str               # what the correlation suggests
    confidence: str         # "weak" | "moderate" | "strong"
    metric: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _synth_dir(storage_root: Path | str, user_id: str) -> Path:
    p = Path(storage_root) / "users" / user_id / "graph" / "synth"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _causal_path(storage_root: Path | str, user_id: str) -> Path:
    return _synth_dir(storage_root, user_id) / "causal.jsonl"


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


def _to_week(dt: datetime) -> str:
    """ISO week key — 'YYYY-Wnn'."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _travel_vs_voice_memos(graph_store: GraphStore, user_id: str) -> CausalObservation | None:
    """Compare weeks containing a Wallet boarding pass vs weeks without
    one. Do voice memos cluster around travel?"""
    flight_weeks: set[str] = set()
    voice_weeks: defaultdict[str, int] = defaultdict(int)

    for p in graph_store.iter_entities(user_id, "project"):
        sources = p.get("sources") or []
        if not any("wallet:boardingPass" in s for s in sources):
            continue
        dt = _parse_iso(p.get("last_activity"))
        if dt:
            flight_weeks.add(_to_week(dt))

    for f in graph_store.iter_entities(user_id, "file"):
        if f.get("kind") != "voice_memo":
            continue
        dt = _parse_iso(f.get("modified"))
        if dt:
            voice_weeks[_to_week(dt)] += 1

    if not flight_weeks or not voice_weeks:
        return None

    flight_set = flight_weeks
    voice_in_flight_weeks = sum(c for w, c in voice_weeks.items() if w in flight_set)
    voice_in_other_weeks = sum(c for w, c in voice_weeks.items() if w not in flight_set)

    flight_week_count = max(1, len(flight_set))
    other_week_count = max(1, len(set(voice_weeks.keys()) - flight_set))
    rate_in_flight = voice_in_flight_weeks / flight_week_count
    rate_in_other = voice_in_other_weeks / other_week_count

    if rate_in_other == 0 and rate_in_flight == 0:
        return None

    # Ratio test
    ratio = rate_in_flight / max(0.001, rate_in_other)
    if ratio < 1.5 and ratio > 1 / 1.5:
        return None  # not a meaningful skew either way

    if ratio > 1.5:
        when = f"You're traveling (Wallet boarding pass that week)"
        then = (
            f"You record ~{rate_in_flight:.1f} voice memos per week vs "
            f"~{rate_in_other:.1f} when you're not traveling — "
            f"~{ratio:.1f}× more often."
        )
        confidence = "strong" if ratio > 2.5 else "moderate"
    else:
        inv = 1 / ratio
        when = f"You're not traveling that week"
        then = (
            f"You record ~{rate_in_other:.1f} voice memos per week vs "
            f"~{rate_in_flight:.1f} during travel — ~{inv:.1f}× more often "
            f"when home."
        )
        confidence = "strong" if inv > 2.5 else "moderate"

    return CausalObservation(
        id="causal_travel_voice_memos",
        when=when,
        then=then,
        confidence=confidence,
        metric={
            "flight_weeks": len(flight_set),
            "voice_in_flight_weeks": voice_in_flight_weeks,
            "voice_in_other_weeks": voice_in_other_weeks,
            "rate_in_flight": round(rate_in_flight, 3),
            "rate_in_other": round(rate_in_other, 3),
            "ratio": round(ratio, 3),
        },
    )


def _travel_vs_commits(graph_store: GraphStore, user_id: str) -> CausalObservation | None:
    """Does code activity slow during travel weeks? We only have
    commit_count_30d per repo, not per-commit timestamps, so this is a
    rough check: do flight weeks land in months where commit activity
    is lower?

    For V1 this is a stub. Will become real when we extract per-commit
    timestamps from git history in a follow-up.
    """
    return None


def _events_vs_voice_memos(graph_store: GraphStore, user_id: str) -> CausalObservation | None:
    """Same shape as travel-vs-voice. Are voice memos more common in
    weeks where you went to an event?"""
    event_weeks: set[str] = set()
    voice_weeks: defaultdict[str, int] = defaultdict(int)

    for p in graph_store.iter_entities(user_id, "project"):
        sources = p.get("sources") or []
        if not any("wallet:eventTicket" in s for s in sources):
            continue
        dt = _parse_iso(p.get("last_activity"))
        if dt:
            event_weeks.add(_to_week(dt))

    for f in graph_store.iter_entities(user_id, "file"):
        if f.get("kind") != "voice_memo":
            continue
        dt = _parse_iso(f.get("modified"))
        if dt:
            voice_weeks[_to_week(dt)] += 1

    if not event_weeks or not voice_weeks:
        return None

    voice_in_event = sum(c for w, c in voice_weeks.items() if w in event_weeks)
    voice_in_other = sum(c for w, c in voice_weeks.items() if w not in event_weeks)

    event_count = max(1, len(event_weeks))
    other_count = max(1, len(set(voice_weeks.keys()) - event_weeks))
    rate_in_event = voice_in_event / event_count
    rate_in_other = voice_in_other / other_count

    if rate_in_other == 0 and rate_in_event == 0:
        return None
    ratio = rate_in_event / max(0.001, rate_in_other)
    if 0.67 < ratio < 1.5:
        return None

    if ratio > 1.5:
        when = "You went to an event that week (ticketed)"
        then = (
            f"Voice memo rate: ~{rate_in_event:.1f}/week during event-weeks "
            f"vs ~{rate_in_other:.1f}/week otherwise — ~{ratio:.1f}× more often."
        )
    else:
        inv = 1 / ratio
        when = "You didn't go to a ticketed event that week"
        then = (
            f"Voice memo rate: ~{rate_in_other:.1f}/week vs ~{rate_in_event:.1f} "
            f"during event-weeks — ~{inv:.1f}× more often without events."
        )

    return CausalObservation(
        id="causal_events_voice_memos",
        when=when,
        then=then,
        confidence="weak" if 0.67 < ratio < 2 and 0.5 < ratio < 2 else "moderate",
        metric={
            "event_weeks": len(event_weeks),
            "voice_in_event_weeks": voice_in_event,
            "voice_in_other_weeks": voice_in_other,
            "ratio": round(ratio, 3),
        },
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_DETECTORS = [
    _travel_vs_voice_memos,
    _events_vs_voice_memos,
    _travel_vs_commits,  # stub for now
]


def build_causal(
    *,
    graph_store: GraphStore,
    storage_root: Path | str,
    user_id: str,
) -> list[CausalObservation]:
    """Run all detectors, persist, return."""
    out: list[CausalObservation] = []
    for det in _DETECTORS:
        try:
            r = det(graph_store, user_id)
        except Exception:
            continue
        if r is not None:
            out.append(r)
    _write_causal(storage_root, user_id, out)
    return out


def load_causal(storage_root: Path | str, user_id: str) -> list[CausalObservation]:
    p = _causal_path(storage_root, user_id)
    if not p.is_file():
        return []
    out: list[CausalObservation] = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(CausalObservation(**d))
        except Exception:
            continue
    return out


def _write_causal(storage_root: Path | str, user_id: str, items: list[CausalObservation]) -> None:
    import os
    p = _causal_path(storage_root, user_id)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        for d in items:
            f.write(json.dumps(asdict(d), default=str) + "\n")
    os.replace(tmp, p)
