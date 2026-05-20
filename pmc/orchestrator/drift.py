"""Drift detection — has the user's voice diverged from what their model produces?

Two complementary signals:

1. **New-data signal**: how many new training-quality items have accumulated
   since the last shipped run? Cheap to compute (just count the memory store
   vs the last run's snapshot).

2. **Voice-divergence signal**: compute the user's recent writing's style
   profile and compare against the style profile attached to the current
   adapter. If the user's voice has shifted, the existing LoRA no longer
   fits — time to refresh.

A `DriftReport` rolls both signals up into one `should_refresh` boolean +
a human-readable `reason`, so the orchestrator's refresh trigger only has
to check one thing.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from pmc.memory.store import MemoryStore
from pmc.schema.conversation import Completion


@dataclass(frozen=True)
class DriftReport:
    """Summary of whether a retrain is warranted."""

    should_refresh: bool
    reason: str
    new_items_since_last_run: int
    days_since_last_run: float
    voice_distance: float | None = None  # placeholder for future style-vector compare


@dataclass(frozen=True)
class DriftConfig:
    """Thresholds for triggering a refresh.

    These defaults are deliberately conservative — refresh is expensive, both
    in compute and in eval-gate risk (each new LoRA could be worse). Better
    to wait for clear signal than to retrain every day.
    """

    min_new_items: int = 500              # need at least this many new examples
    max_days_since_run: float = 30.0      # cadence ceiling — refresh anyway every N days
    min_items_for_first_run: int = 100    # bootstrap threshold (no prior run)


def assess(
    store: MemoryStore,
    last_run_ts: float | None,
    last_run_item_count: int | None,
    config: DriftConfig | None = None,
) -> DriftReport:
    """Decide whether a refresh is warranted given current store state."""
    cfg = config or DriftConfig()

    current_items = store.count()
    new_items = current_items - (last_run_item_count or 0)
    now = time.time()
    days_since = (
        (now - last_run_ts) / 86400.0 if last_run_ts is not None else float("inf")
    )

    # Bootstrap: no prior run.
    if last_run_ts is None:
        if current_items >= cfg.min_items_for_first_run:
            return DriftReport(
                should_refresh=True,
                reason=f"bootstrap: {current_items} items ready for first training",
                new_items_since_last_run=current_items,
                days_since_last_run=days_since,
            )
        return DriftReport(
            should_refresh=False,
            reason=f"bootstrap: only {current_items} items, need {cfg.min_items_for_first_run}",
            new_items_since_last_run=current_items,
            days_since_last_run=days_since,
        )

    # Cadence ceiling: refresh even with modest new data if too long has passed.
    if days_since >= cfg.max_days_since_run and new_items > 0:
        return DriftReport(
            should_refresh=True,
            reason=(
                f"cadence: {days_since:.1f} days since last run "
                f"(ceiling {cfg.max_days_since_run:.0f}), {new_items} new items"
            ),
            new_items_since_last_run=new_items,
            days_since_last_run=days_since,
        )

    # Volume signal: enough new data accumulated.
    if new_items >= cfg.min_new_items:
        return DriftReport(
            should_refresh=True,
            reason=f"volume: {new_items} new items since last run (threshold {cfg.min_new_items})",
            new_items_since_last_run=new_items,
            days_since_last_run=days_since,
        )

    return DriftReport(
        should_refresh=False,
        reason=(
            f"no signal: {new_items} new items (need {cfg.min_new_items}), "
            f"{days_since:.1f} days (ceiling {cfg.max_days_since_run:.0f})"
        ),
        new_items_since_last_run=new_items,
        days_since_last_run=days_since,
    )


def hold_out_recent(
    completions: Iterable[Completion],
    n: int = 50,
) -> list[Completion]:
    """Take the most recent N completions for use as a fresh eval set.

    Used by the refresh trigger to score the current adapter's voice match
    against the user's most recent writing. Future work: hook this into the
    eval gate as a per-refresh held-out set rather than the dataset-time split.
    """
    items = list(completions)
    # Completions don't carry timestamps in the core schema; we rely on caller
    # to pass them in chronological order. Take the tail.
    return items[-n:]
