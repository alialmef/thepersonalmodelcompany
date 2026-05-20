"""Refresh trigger — eval-gated retrain orchestration.

The continuous-learning loop has three triggers (any one fires a refresh):

- `manual`: user clicked "retrain my model now"
- `drift`: `pmc.orchestrator.drift.assess()` returned should_refresh=True
- `cadence`: scheduled run hit its cadence ceiling

Whatever the trigger, the loop is the same:

    1. Snapshot the current state (run_id of the active adapter + memory count)
    2. Launch a training run (Modal or local)
    3. When it produces a new adapter, run the eval gate against the prior
    4. Promote if scalar > previous; else log as 'rejected' and keep current
    5. Append a RunRecord either way

The ledger is the immutable history. The `active.json` pointer (already in
`storage/`) is what the serve layer uses to pick an adapter — promotion just
flips that pointer atomically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pmc.orchestrator.drift import DriftConfig, DriftReport, assess
from pmc.orchestrator.runs_ledger import (
    RunRecord,
    append_run,
    latest_shipped,
    new_run_id,
)


@dataclass(frozen=True)
class RefreshDecision:
    """Output of `should_refresh()` — the why + the next run_id if firing."""

    fire: bool
    trigger: str           # "manual" | "drift" | "cadence" | "none"
    reason: str
    drift: DriftReport | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of a completed refresh — what to write to the ledger."""

    record: RunRecord
    promoted: bool


def should_refresh(
    ledger_path: Path,
    new_item_count: int,
    drift_config: DriftConfig | None = None,
    manual: bool = False,
) -> RefreshDecision:
    """Decide whether to fire a refresh. Does not actually train."""
    if manual:
        return RefreshDecision(
            fire=True,
            trigger="manual",
            reason="user-initiated refresh",
            run_id=new_run_id(),
        )

    previous = latest_shipped(ledger_path)
    last_ts = previous.ts if previous else None
    last_count = previous.train_examples if previous else None

    # Faux MemoryStore wrapper: assess() only calls .count(). Pass a tiny shim
    # so we don't have to open the SQLite handle here.
    class _Counter:
        def count(self_inner) -> int:
            return new_item_count

    report = assess(
        store=_Counter(),  # type: ignore[arg-type]
        last_run_ts=last_ts,
        last_run_item_count=last_count,
        config=drift_config,
    )
    if not report.should_refresh:
        return RefreshDecision(
            fire=False,
            trigger="none",
            reason=report.reason,
            drift=report,
        )

    trigger = "cadence" if "cadence" in report.reason else "drift"
    return RefreshDecision(
        fire=True,
        trigger=trigger,
        reason=report.reason,
        drift=report,
        run_id=new_run_id(),
    )


def evaluate_and_promote(
    new_run_id_: str,
    new_scalar: float,
    ledger_path: Path,
    *,
    train_examples: int,
    base_model: str,
    adapter_size_mb: float,
    promote_fn: Callable[[str], None],
    notes: str = "",
    extras: dict[str, float | int | str] | None = None,
) -> RefreshResult:
    """Compare a fresh adapter against the current best and promote if better.

    - `new_scalar`: the canonical eval metric for the new adapter (higher = better)
    - `promote_fn(run_id)`: flips the active.json pointer to the given run_id
                            (caller-supplied so this module doesn't depend on storage)

    Always writes a RunRecord, whether the new adapter wins or loses. Losing
    runs are recorded as "rejected" so the ledger shows the user — and us —
    that we tried and the gate held.
    """
    previous = latest_shipped(ledger_path)
    prior_scalar = previous.scalar if (previous and previous.scalar is not None) else float("-inf")
    promote = new_scalar > prior_scalar

    if promote:
        promote_fn(new_run_id_)

    record = RunRecord(
        run_id=new_run_id_,
        ts=time.time(),
        status="shipped" if promote else "rejected",
        scalar=new_scalar,
        base_model=base_model,
        train_examples=train_examples,
        adapter_size_mb=adapter_size_mb,
        promoted_from=previous.run_id if (promote and previous) else None,
        notes=notes
        or (
            f"promoted (gain {new_scalar - prior_scalar:+.4f})"
            if promote
            else f"rejected (would have been {new_scalar - prior_scalar:+.4f})"
        ),
        extras=extras or {},
    )
    append_run(ledger_path, record)
    return RefreshResult(record=record, promoted=promote)
