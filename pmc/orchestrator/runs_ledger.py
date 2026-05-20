"""Per-user append-only run ledger.

Inspired by Karpathy's `results.tsv` pattern in autoresearch — one row per
training attempt, persisted forever, the source of truth for the model's
history. Lets us answer:

- Is the user's model getting better over time?
- Which run is currently shipped?
- When was the last refresh attempted, and did it pass the eval gate?

The ledger is intentionally flat JSONL (one record per line) so it's trivial
to read by humans, by Python, by jq, by anything. Lives at
`storage_root/users/{user_id}/runs.jsonl`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    """One row of the run ledger."""

    run_id: str                          # unique per training attempt
    ts: float                            # unix timestamp
    status: str                          # "shipped" | "rejected" | "crashed" | "drift_only"
    scalar: float | None = None          # the single canonical eval metric
    base_model: str | None = None
    train_examples: int | None = None
    adapter_size_mb: float | None = None
    promoted_from: str | None = None     # previous run_id replaced by this one
    notes: str = ""
    extras: dict[str, float | int | str] = field(default_factory=dict)


def append_run(ledger_path: Path, record: RunRecord) -> None:
    """Append a RunRecord to the user's runs.jsonl. Parent dirs created."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(record))
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(payload + "\n")


def read_runs(ledger_path: Path) -> list[RunRecord]:
    """Load all RunRecords. Returns [] if the ledger doesn't exist."""
    if not ledger_path.exists():
        return []
    out: list[RunRecord] = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        out.append(
            RunRecord(
                run_id=data["run_id"],
                ts=data["ts"],
                status=data["status"],
                scalar=data.get("scalar"),
                base_model=data.get("base_model"),
                train_examples=data.get("train_examples"),
                adapter_size_mb=data.get("adapter_size_mb"),
                promoted_from=data.get("promoted_from"),
                notes=data.get("notes", ""),
                extras=data.get("extras", {}),
            )
        )
    return out


def latest_shipped(ledger_path: Path) -> RunRecord | None:
    """Return the most recent record with status='shipped'."""
    runs = read_runs(ledger_path)
    for record in reversed(runs):
        if record.status == "shipped":
            return record
    return None


def best_scalar(ledger_path: Path) -> float | None:
    """Return the highest scalar across all shipped runs (running-best)."""
    runs = read_runs(ledger_path)
    scalars = [r.scalar for r in runs if r.status == "shipped" and r.scalar is not None]
    return max(scalars) if scalars else None


def new_run_id() -> str:
    """ISO-8601 timestamp run id — sortable, human-readable, unique enough."""
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())
