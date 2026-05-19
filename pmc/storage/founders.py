"""Founder tracker — first N users get their first training free.

Per the founder-pricing decision: the first 100 users to actually trigger a
training run get one free Llama 3.1 8B (Try tier) training. Tracked silently
on the backend — no urgency tactics on the site, no countdown UI.

Single source of truth: `{storage_root}/founders.json`. The file holds the
list of granted user_ids and the total slots used. Atomic-enough for V0
single-process serving; if you go multi-process behind a load balancer, swap
this for a row in Postgres with a UNIQUE constraint on user_id.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_TOTAL_SLOTS = 100


class FounderState(BaseModel):
    total_slots: int = DEFAULT_TOTAL_SLOTS
    used: int = 0
    founders: list[str] = Field(default_factory=list)
    granted_at: dict[str, str] = Field(default_factory=dict)  # user_id → ISO timestamp


class FounderGrant(BaseModel):
    """Result of a founder-status check/grant."""

    user_id: str
    is_founder: bool
    slots_remaining: int
    granted_now: bool = False
    granted_at: datetime | None = None


class FounderTracker:
    """First-N-users tracker for the founder free-tier offer."""

    def __init__(
        self,
        root: Path | str,
        *,
        total_slots: int = DEFAULT_TOTAL_SLOTS,
    ) -> None:
        self.root = Path(root)
        self.total_slots = total_slots
        self._file = self.root / "founders.json"

    # -- queries ----------------------------------------------------------

    def is_founder(self, user_id: str) -> bool:
        return user_id in self._load().founders

    def slots_remaining(self) -> int:
        state = self._load()
        return max(0, state.total_slots - state.used)

    def used(self) -> int:
        return self._load().used

    def state(self) -> FounderState:
        return self._load()

    # -- grant ------------------------------------------------------------

    def grant_if_available(self, user_id: str) -> FounderGrant:
        """Idempotently grant founder status to user_id if slots remain.

        If the user is already a founder, returns the existing grant.
        If slots are exhausted, returns a non-founder grant.
        """
        state = self._load()
        # Already a founder
        if user_id in state.founders:
            return FounderGrant(
                user_id=user_id,
                is_founder=True,
                slots_remaining=max(0, state.total_slots - state.used),
                granted_now=False,
                granted_at=datetime.fromisoformat(state.granted_at[user_id])
                if user_id in state.granted_at else None,
            )
        # No slots left
        if state.used >= state.total_slots:
            return FounderGrant(
                user_id=user_id,
                is_founder=False,
                slots_remaining=0,
                granted_now=False,
            )
        # Grant
        now = datetime.now()
        state.founders.append(user_id)
        state.granted_at[user_id] = now.isoformat()
        state.used += 1
        self._save(state)
        return FounderGrant(
            user_id=user_id,
            is_founder=True,
            slots_remaining=max(0, state.total_slots - state.used),
            granted_now=True,
            granted_at=now,
        )

    # -- internal ---------------------------------------------------------

    def _load(self) -> FounderState:
        """Load persisted state. The constructor's `total_slots` is authoritative —
        raising or lowering it later applies immediately. The persisted total in
        the file is overwritten on next save."""
        if not self._file.is_file():
            return FounderState(total_slots=self.total_slots)
        try:
            state = FounderState.model_validate_json(self._file.read_text())
        except Exception:
            return FounderState(total_slots=self.total_slots)
        # Always honor the live constructor arg, not the persisted total
        state.total_slots = self.total_slots
        return state

    def _save(self, state: FounderState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Always keep the canonical total in sync with whatever was configured
        state.total_slots = self.total_slots
        self._file.write_text(state.model_dump_json(indent=2))


__all__ = [
    "DEFAULT_TOTAL_SLOTS",
    "FounderGrant",
    "FounderState",
    "FounderTracker",
]
