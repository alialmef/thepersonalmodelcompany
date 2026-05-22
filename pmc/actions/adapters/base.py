"""Base contracts for MCP/local action adapters."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from pmc.actions.schema import ActionProposal, ActionRisk


class ActionExecutionMode(StrEnum):
    SIMULATE = "simulate"
    STAGE = "stage"
    EXECUTE = "execute"
    UNDO = "undo"


class ActionAdapterCapability(BaseModel):
    surface: str
    operation: str
    risk_level: ActionRisk = ActionRisk.MEDIUM
    supports_simulate: bool = True
    supports_stage: bool = False
    supports_execute: bool = False
    supports_undo: bool = False
    requires_confirmation: bool = True
    description: str = ""

    @property
    def key(self) -> str:
        return f"{self.surface}:{self.operation}"


class ActionExecutionRequest(BaseModel):
    user_id: str
    proposal: ActionProposal
    mode: ActionExecutionMode = ActionExecutionMode.SIMULATE
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionExecutionReceipt(BaseModel):
    id: str = Field(default_factory=lambda: f"receipt-{uuid.uuid4().hex[:12]}")
    user_id: str
    proposal_id: str
    surface: str
    operation: str
    mode: ActionExecutionMode
    ok: bool
    preview: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[str] = Field(default_factory=list)
    undo_token: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class ActionAdapter(Protocol):
    @property
    def capabilities(self) -> list[ActionAdapterCapability]:
        ...

    def can_handle(self, surface: str, operation: str) -> bool:
        ...

    def run(self, request: ActionExecutionRequest) -> ActionExecutionReceipt:
        ...
