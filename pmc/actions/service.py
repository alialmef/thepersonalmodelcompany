"""Application service for action proposals and review traces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from pmc.actions.adapters.base import ActionExecutionMode, ActionExecutionReceipt
from pmc.actions.policy import ActionReviewGate, action_review_gate, infer_action_risk
from pmc.actions.registry import ActionAdapterRegistry
from pmc.actions.schema import (
    ActionDecision,
    ActionProposal,
    ActionProposalStatus,
    ActionRisk,
    ActionTrace,
)
from pmc.schema.conversation import Message, Role
from pmc.schema.verification import TrustReport
from pmc.storage.action_store import ActionStore
from pmc.storage.verification_store import VerificationStore


class AuditSink(Protocol):
    def log(
        self,
        user_id: str,
        *,
        stage: str,
        event: str,
        data: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> Any:
        ...


class ActionServiceError(ValueError):
    """Base error for action service validation and lookup failures."""

    status_code = 400


class ActionValidationError(ActionServiceError):
    status_code = 400


class ActionNotFoundError(ActionServiceError):
    status_code = 404


@dataclass(frozen=True)
class ActionProposalResult:
    proposal: ActionProposal
    review: ActionReviewGate
    trust_report: TrustReport

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": True,
            "proposal": self.proposal.model_dump(mode="json"),
            "review": self.review.model_dump(mode="json"),
            "trust_report": self.trust_report.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class ActionProposalListResult:
    proposals: list[ActionProposal]
    trust_report: TrustReport

    def model_dump(self) -> dict[str, Any]:
        return {
            "proposals": [p.model_dump(mode="json") for p in self.proposals],
            "trust_report": self.trust_report.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class ActionReviewResult:
    proposal: ActionProposal
    trace: ActionTrace
    review: ActionReviewGate
    trust_report: TrustReport

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": True,
            "proposal": self.proposal.model_dump(mode="json"),
            "trace": self.trace.model_dump(mode="json"),
            "review": self.review.model_dump(mode="json"),
            "trust_report": self.trust_report.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class ActionTraceResult:
    trace: ActionTrace
    trust_report: TrustReport

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": True,
            "trace": self.trace.model_dump(mode="json"),
            "trust_report": self.trust_report.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class ActionCapabilityListResult:
    capabilities: list[dict[str, Any]]
    trust_report: TrustReport

    def model_dump(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
            "trust_report": self.trust_report.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class ActionExecutionResult:
    proposal: ActionProposal
    receipt: ActionExecutionReceipt
    review: ActionReviewGate
    trust_report: TrustReport

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": self.receipt.ok,
            "proposal": self.proposal.model_dump(mode="json"),
            "receipt": self.receipt.model_dump(mode="json"),
            "review": self.review.model_dump(mode="json"),
            "trust_report": self.trust_report.model_dump(mode="json"),
        }


class ActionService:
    """Coordinates action proposal review with durable training traces."""

    def __init__(
        self,
        verification_store: VerificationStore,
        audit_log: AuditSink,
        *,
        action_store: ActionStore | None = None,
        adapter_registry: ActionAdapterRegistry | None = None,
    ) -> None:
        self.verification_store = verification_store
        self.audit_log = audit_log
        self.action_store = action_store
        self.adapter_registry = adapter_registry

    def create_proposal(self, user_id: str, payload: dict[str, Any]) -> ActionProposalResult:
        surface = str(payload.get("surface") or "").strip()
        operation = str(payload.get("operation") or "").strip()
        if not surface or not operation:
            raise ActionValidationError("surface and operation are required")

        proposed_payload = payload.get("proposed_payload") or payload.get("payload") or {}
        if not isinstance(proposed_payload, dict):
            raise ActionValidationError("proposed_payload must be an object")

        risk = self._risk_from_payload(payload, surface, operation, proposed_payload)
        metadata = self._metadata_from_payload(payload)
        proposal = ActionProposal(
            user_id=user_id,
            surface=surface,
            operation=operation,
            prompt=self._messages_from_payload(payload.get("prompt")),
            proposed_text=str(payload.get("proposed_text") or payload.get("preview") or ""),
            proposed_payload=proposed_payload,
            rationale=str(payload.get("rationale") or ""),
            required_capability=str(
                payload.get("required_capability") or f"{surface}:{operation}"
            ),
            risk_level=risk,
            model=payload.get("model"),
            run_id=payload.get("run_id"),
            metadata={
                "dry_run": True,
                **metadata,
            },
        )
        self.verification_store.append_action_proposal(user_id, proposal)
        report = self.verification_store.trust_report(user_id)
        self.audit_log.log(
            user_id,
            stage="action",
            event="action_proposal_created",
            data={
                "proposal_id": proposal.id,
                "surface": proposal.surface,
                "operation": proposal.operation,
                "risk_level": proposal.risk_level.value,
                "readiness": report.readiness,
            },
        )
        return ActionProposalResult(
            proposal=proposal,
            review=action_review_gate(report, proposal.risk_level),
            trust_report=report,
        )

    def list_proposals(
        self,
        user_id: str,
        *,
        status: str | None = None,
        limit: int | None = 20,
    ) -> ActionProposalListResult:
        proposals = self.verification_store.list_action_proposals(
            user_id,
            status=status,
            limit=limit,
        )
        return ActionProposalListResult(
            proposals=proposals,
            trust_report=self.verification_store.trust_report(user_id),
        )

    def review_proposal(
        self,
        user_id: str,
        proposal_id: str,
        payload: dict[str, Any],
    ) -> ActionReviewResult:
        proposal = self.verification_store.get_action_proposal(user_id, proposal_id)
        if proposal is None:
            raise ActionNotFoundError(f"No action proposal {proposal_id!r}")

        decision = self._decision_from_payload(payload)
        edited_text = payload.get("edited_text") or payload.get("editedText")
        final_payload = payload.get("final_payload") or payload.get("finalPayload")
        if final_payload is None:
            final_payload = proposal.proposed_payload
        if not isinstance(final_payload, dict):
            raise ActionValidationError("final_payload must be an object")

        status_by_decision = {
            ActionDecision.APPROVED: ActionProposalStatus.APPROVED,
            ActionDecision.EDITED: ActionProposalStatus.EDITED,
            ActionDecision.REJECTED: ActionProposalStatus.REJECTED,
            ActionDecision.UNDONE: ActionProposalStatus.UNDONE,
            ActionDecision.IGNORED: ActionProposalStatus.EXPIRED,
        }
        proposal = proposal.model_copy(
            update={
                "status": status_by_decision[decision],
                "reviewed_at": datetime.now(),
            }
        )
        self.verification_store.update_action_proposal(user_id, proposal)

        metadata = self._metadata_from_payload(payload)
        trace = ActionTrace(
            user_id=user_id,
            surface=proposal.surface,
            operation=proposal.operation,
            prompt=proposal.prompt,
            proposed_text=proposal.proposed_text,
            proposed_payload=proposal.proposed_payload,
            decision=decision,
            edited_text=edited_text,
            final_payload=final_payload,
            risk_level=proposal.risk_level,
            proposal_id=proposal.id,
            run_id=proposal.run_id,
            metadata={
                "required_capability": proposal.required_capability,
                "rationale": proposal.rationale,
                **metadata,
            },
        )
        self.verification_store.append_action_trace(user_id, trace)
        report = self.verification_store.trust_report(user_id)
        self.audit_log.log(
            user_id,
            stage="action",
            event="action_proposal_reviewed",
            data={
                "proposal_id": proposal.id,
                "trace_id": trace.id,
                "decision": trace.decision.value,
                "risk_level": trace.risk_level.value,
                "readiness": report.readiness,
            },
        )
        return ActionReviewResult(
            proposal=proposal,
            trace=trace,
            review=action_review_gate(report, proposal.risk_level),
            trust_report=report,
        )

    def record_trace(self, user_id: str, payload: dict[str, Any]) -> ActionTraceResult:
        body = dict(payload)
        body["user_id"] = user_id
        try:
            trace = ActionTrace.model_validate(body)
        except Exception as e:
            raise ActionValidationError(f"Invalid action trace: {e}") from e

        self.verification_store.append_action_trace(user_id, trace)
        report = self.verification_store.trust_report(user_id)
        self.audit_log.log(
            user_id,
            stage="action",
            event="action_trace_recorded",
            data={
                "surface": trace.surface,
                "operation": trace.operation,
                "decision": trace.decision.value,
                "readiness": report.readiness,
            },
        )
        return ActionTraceResult(trace=trace, trust_report=report)

    def list_capabilities(self, user_id: str) -> ActionCapabilityListResult:
        registry = self._require_registry()
        capabilities = []
        for capability in registry.capabilities():
            item = capability.model_dump(mode="json")
            item["key"] = capability.key
            capabilities.append(item)
        return ActionCapabilityListResult(
            capabilities=capabilities,
            trust_report=self.verification_store.trust_report(user_id),
        )

    def run_proposal(
        self,
        user_id: str,
        proposal_id: str,
        mode: ActionExecutionMode | str,
        payload: dict[str, Any] | None = None,
    ) -> ActionExecutionResult:
        proposal = self.verification_store.get_action_proposal(user_id, proposal_id)
        if proposal is None:
            raise ActionNotFoundError(f"No action proposal {proposal_id!r}")
        try:
            execution_mode = (
                mode if isinstance(mode, ActionExecutionMode) else ActionExecutionMode(str(mode))
            )
        except ValueError as e:
            raise ActionValidationError(f"Unknown execution mode: {mode!r}") from e

        if execution_mode == ActionExecutionMode.EXECUTE:
            self._require_execution_review(proposal)

        registry = self._require_registry()
        adapter = registry.find(proposal.surface, proposal.operation)
        if adapter is None:
            raise ActionValidationError(
                f"No adapter registered for {proposal.surface}:{proposal.operation}"
            )

        from pmc.actions.adapters.base import ActionExecutionRequest

        request = ActionExecutionRequest(
            user_id=user_id,
            proposal=proposal,
            mode=execution_mode,
            payload=payload or {},
        )
        try:
            receipt = adapter.run(request)
        except ValueError as e:
            raise ActionValidationError(str(e)) from e

        action_store = self._require_action_store()
        action_store.append_receipt(user_id, receipt)

        if receipt.ok and execution_mode in {ActionExecutionMode.EXECUTE, ActionExecutionMode.UNDO}:
            status = (
                ActionProposalStatus.EXECUTED
                if execution_mode == ActionExecutionMode.EXECUTE
                else ActionProposalStatus.UNDONE
            )
            proposal = proposal.model_copy(
                update={
                    "status": status,
                    "reviewed_at": proposal.reviewed_at or datetime.now(),
                }
            )
            self.verification_store.update_action_proposal(user_id, proposal)

        report = self.verification_store.trust_report(user_id)
        self.audit_log.log(
            user_id,
            stage="action",
            event=f"action_{execution_mode.value}",
            data={
                "proposal_id": proposal.id,
                "receipt_id": receipt.id,
                "surface": proposal.surface,
                "operation": proposal.operation,
                "ok": receipt.ok,
                "risk_level": proposal.risk_level.value,
                "readiness": report.readiness,
            },
        )
        return ActionExecutionResult(
            proposal=proposal,
            receipt=receipt,
            review=action_review_gate(report, proposal.risk_level),
            trust_report=report,
        )

    @staticmethod
    def _messages_from_payload(raw: Any) -> list[Message]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [Message(role=Role.USER, content=raw)]
        if not isinstance(raw, list):
            raise ActionValidationError("prompt must be a string or list")
        messages: list[Message] = []
        for i, item in enumerate(raw):
            try:
                messages.append(Message.model_validate(item))
            except Exception as e:
                raise ActionValidationError(f"Invalid prompt message at index {i}: {e}") from e
        return messages

    @staticmethod
    def _risk_from_payload(
        payload: dict[str, Any],
        surface: str,
        operation: str,
        proposed_payload: dict[str, Any],
    ) -> ActionRisk:
        try:
            return (
                ActionRisk(str(payload["risk_level"]))
                if payload.get("risk_level")
                else infer_action_risk(surface, operation, proposed_payload)
            )
        except ValueError as e:
            raise ActionValidationError(
                f"Unknown risk_level: {payload.get('risk_level')!r}"
            ) from e

    @staticmethod
    def _decision_from_payload(payload: dict[str, Any]) -> ActionDecision:
        try:
            return ActionDecision(str(payload.get("decision") or "ignored"))
        except ValueError as e:
            raise ActionValidationError(
                f"Unknown action decision: {payload.get('decision')!r}"
            ) from e

    @staticmethod
    def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ActionValidationError("metadata must be an object")
        return metadata

    def _require_registry(self) -> ActionAdapterRegistry:
        if self.adapter_registry is None:
            raise ActionValidationError("No action adapter registry is configured")
        return self.adapter_registry

    def _require_action_store(self) -> ActionStore:
        if self.action_store is None:
            raise ActionValidationError("No action receipt store is configured")
        return self.action_store

    @staticmethod
    def _require_execution_review(proposal: ActionProposal) -> None:
        if proposal.risk_level == ActionRisk.LOW:
            return
        if proposal.status not in {ActionProposalStatus.APPROVED, ActionProposalStatus.EDITED}:
            raise ActionValidationError(
                "Medium and high risk actions must be approved or edited before execution"
            )
