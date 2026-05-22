"""Risk and trust policy for proposed actions."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from pmc.actions.schema import ActionRisk
from pmc.schema.verification import TrustReport


class ActionReviewGate(BaseModel):
    """Review-time execution gate returned with every proposed action."""

    execution_allowed: bool
    requires_confirmation: bool
    readiness_required: str


_READINESS_BY_RISK = {
    ActionRisk.LOW: "unproven",
    ActionRisk.MEDIUM: "sandbox",
    ActionRisk.HIGH: "supervised",
}


def action_review_gate(report: TrustReport, risk: ActionRisk) -> ActionReviewGate:
    """Decide whether the current trust state can execute this risk tier."""

    readiness = report.readiness
    execution_allowed = (
        risk == ActionRisk.LOW
        or (risk == ActionRisk.MEDIUM and readiness in {"sandbox", "supervised"})
        or (risk == ActionRisk.HIGH and readiness == "supervised")
    )
    return ActionReviewGate(
        execution_allowed=execution_allowed,
        requires_confirmation=risk != ActionRisk.LOW,
        readiness_required=_READINESS_BY_RISK[risk],
    )


def infer_action_risk(surface: str, operation: str, payload: dict[str, Any]) -> ActionRisk:
    """Conservative lexical risk classifier for untrusted proposal payloads.

    This intentionally starts simple and deterministic. Adapter-specific policy
    can replace or refine it once execution surfaces are registered.
    """

    payload_text = json.dumps(payload, sort_keys=True, default=str)
    text = f"{surface} {operation} {payload_text}".lower()
    high_terms = (
        "send",
        "delete",
        "remove",
        "publish",
        "post",
        "pay",
        "purchase",
        "transfer",
        "share",
        "shell",
        "execute",
        "run_command",
        "commit",
        "push",
        "merge",
    )
    medium_terms = (
        "create",
        "update",
        "edit",
        "move",
        "rename",
        "schedule",
        "reschedule",
        "cancel",
        "write",
        "save",
    )
    if any(term in text for term in high_terms):
        return ActionRisk.HIGH
    if any(term in text for term in medium_terms):
        return ActionRisk.MEDIUM
    return ActionRisk.LOW
