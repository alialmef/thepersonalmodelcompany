"""Durable action execution receipt storage."""

from __future__ import annotations

from pathlib import Path

from pmc.actions.adapters.base import ActionExecutionReceipt
from pmc.storage.paths import StoragePaths


class ActionStore:
    """Append-only receipts for simulated, staged, executed, and undone actions."""

    def __init__(self, root: Path | str) -> None:
        self.paths = StoragePaths(root)

    def append_receipt(self, user_id: str, receipt: ActionExecutionReceipt) -> None:
        path = self.paths.action_receipts_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(receipt.model_dump_json() + "\n")

    def list_receipts(
        self,
        user_id: str,
        *,
        proposal_id: str | None = None,
        limit: int | None = None,
    ) -> list[ActionExecutionReceipt]:
        receipts: list[ActionExecutionReceipt] = []
        path = self.paths.action_receipts_file(user_id)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        receipts.append(ActionExecutionReceipt.model_validate_json(line))
        if proposal_id is not None:
            receipts = [r for r in receipts if r.proposal_id == proposal_id]
        if limit is not None and limit >= 0:
            receipts = receipts[-limit:]
        return receipts

    def get_receipt(self, user_id: str, receipt_id: str) -> ActionExecutionReceipt | None:
        for receipt in self.list_receipts(user_id):
            if receipt.id == receipt_id:
                return receipt
        return None
