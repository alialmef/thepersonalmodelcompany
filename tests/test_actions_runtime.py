"""Tests for executable action adapters."""

from __future__ import annotations

from pathlib import Path

from pmc.actions.adapters.base import ActionExecutionMode
from pmc.actions.registry import build_default_action_registry
from pmc.actions.service import ActionService
from pmc.storage.action_store import ActionStore
from pmc.storage.audit import AuditLog
from pmc.storage.verification_store import VerificationStore


def _service(root: Path) -> ActionService:
    return ActionService(
        VerificationStore(root),
        AuditLog(root),
        action_store=ActionStore(root),
        adapter_registry=build_default_action_registry(root),
    )


def test_action_runtime_simulate_execute_and_undo_file_write(tmp_path: Path):
    service = _service(tmp_path / "storage")
    target = tmp_path / "laptop" / "note.md"

    created = service.create_proposal(
        "alex",
        {
            "surface": "files",
            "operation": "write_text",
            "risk_level": "medium",
            "proposed_text": "write local note",
            "proposed_payload": {
                "path": str(target),
                "content": "frontier memory note",
            },
        },
    )

    simulated = service.run_proposal(
        "alex",
        created.proposal.id,
        ActionExecutionMode.SIMULATE,
    )
    assert simulated.receipt.ok is True
    assert target.exists() is False

    blocked = None
    try:
        service.run_proposal("alex", created.proposal.id, ActionExecutionMode.EXECUTE)
    except Exception as e:
        blocked = e
    assert blocked is not None

    service.review_proposal("alex", created.proposal.id, {"decision": "approved"})
    executed = service.run_proposal("alex", created.proposal.id, ActionExecutionMode.EXECUTE)
    assert executed.receipt.ok is True
    assert target.read_text(encoding="utf-8") == "frontier memory note"
    assert executed.receipt.undo_token

    undone = service.run_proposal(
        "alex",
        created.proposal.id,
        ActionExecutionMode.UNDO,
        {"undo_token": executed.receipt.undo_token},
    )
    assert undone.receipt.ok is True
    assert target.exists() is False


def test_action_runtime_lists_default_capabilities(tmp_path: Path):
    capabilities = _service(tmp_path / "storage").list_capabilities("alex").capabilities
    keys = {capability["key"] for capability in capabilities}
    assert "files:write_text" in keys
    assert "notes:create" in keys
