"""Salience scoring — who matters to the user, in what proportion.

A salience score in [0, 1] per entity. Used to decide who gets the
expensive per-entity characterization pass, who ends up in the
`whoami` bootstrap packet, and what the chat context block surfaces.

For people, salience combines:
  - channel_counts: total messages exchanged (log-scaled)
  - recency: last_seen — recent gets a boost
  - engagement: number of open_loops linked to this person
  - temperature: the extractor's own warm/etc tag
  - inferred_role: occasional/regular/close

These are deliberately simple weights for v1. The consolidator pass
recomputes them each run — no need for incremental updates.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pmc.storage.graph_store import GraphStore, _is_quality_person


WEIGHTS = {
    "channel_counts": 0.4,
    "recency":        0.3,
    "engagement":     0.2,
    "temperature":    0.05,
    "role":           0.05,
}


def score_people(
    store: GraphStore,
    user_id: str,
) -> list[tuple[dict[str, Any], float]]:
    """Score every quality-passing person. Returns
    `[(person_record, salience_in_0_to_1)]` sorted descending."""

    # Pre-index open_loops by related person id so we can count engagement
    # in one pass.
    loop_count: dict[str, int] = {}
    for loop in store.iter_entities(user_id, "open_loop"):
        for pid in loop.get("related_person_ids") or []:
            loop_count[pid] = loop_count.get(pid, 0) + 1

    now = datetime.now(timezone.utc)
    scored: list[tuple[dict[str, Any], float]] = []
    for p in store.iter_entities(user_id, "person"):
        if not _is_quality_person(p):
            continue
        salience = _person_salience(p, loop_count.get(p.get("id", ""), 0), now)
        scored.append((p, salience))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------


def _person_salience(p: dict[str, Any], loops: int, now: datetime) -> float:
    cc = _channel_score(p.get("channel_counts") or {})
    rec = _recency_score(p.get("last_seen"), now)
    eng = _engagement_score(loops)
    temp = _temperature_score(p.get("temperature"))
    role = _role_score(p.get("inferred_role"))
    total = (
        cc   * WEIGHTS["channel_counts"]
        + rec * WEIGHTS["recency"]
        + eng * WEIGHTS["engagement"]
        + temp * WEIGHTS["temperature"]
        + role * WEIGHTS["role"]
    )
    return min(1.0, max(0.0, total))


def _channel_score(counts: dict[str, Any]) -> float:
    """log10-scaled message count, normalized so 1000+ messages → ~1.0."""
    total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    if total <= 0:
        return 0.0
    return min(1.0, math.log10(1 + total) / 3.0)  # log10(1000) = 3


def _recency_score(last_seen: Any, now: datetime) -> float:
    """1.0 if today; 0.5 at 30 days; ~0.0 beyond a year."""
    if not isinstance(last_seen, str) or not last_seen:
        return 0.0
    try:
        dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return 0.0
    days = max(0, (now - dt).days)
    # Exponential decay: half-life ~30 days
    return math.exp(-days / 30.0)


def _engagement_score(loops: int) -> float:
    """log-scaled count of open_loops linking to this person."""
    if loops <= 0:
        return 0.0
    return min(1.0, math.log10(1 + loops) / 1.5)  # 30+ loops = 1.0


def _temperature_score(t: Any) -> float:
    if t == "hot":   return 1.0
    if t == "warm":  return 0.6
    if t == "cool":  return 0.3
    return 0.0


def _role_score(r: Any) -> float:
    if r == "close":      return 1.0
    if r == "regular":    return 0.7
    if r == "occasional": return 0.3
    return 0.0


__all__ = ["score_people", "WEIGHTS"]
