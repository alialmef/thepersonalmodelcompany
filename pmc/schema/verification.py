"""Personal verification schemas.

This is the proving-ground layer: private probes, user judgments, and action
traces that tell us whether a model sounds like, thinks like, and acts like
the user. Raw data teaches voice; these objects teach the correction function.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from pmc.schema.annotations import PreferenceAnnotation, SourceAnnotation
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
    SourceType,
)


class ProbeKind(StrEnum):
    """What a probe is trying to verify."""

    VOICE = "voice"          # Does it sound like the user?
    DECISION = "decision"    # Would the user choose this?
    ACTION = "action"        # Would the user do this in a tool surface?
    FACTUAL = "factual"      # Is user-specific knowledge correct?
    PRIVACY = "privacy"      # Does it avoid leaking sensitive data?
    MEMORY = "memory"        # Does retrieved memory look true and useful?


class CandidateOrigin(StrEnum):
    """Where one answer/action candidate came from."""

    PERSONAL_MODEL = "personal_model"
    BASE_MODEL = "base_model"
    REAL_USER = "real_user"
    USER_EDIT = "user_edit"
    SYNTHETIC = "synthetic"
    TOOL_PLAN = "tool_plan"


class JudgmentVerdict(StrEnum):
    """User-facing judgment labels normalized for training."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    CHOOSE = "choose"
    UNSURE = "unsure"
    TOO_FORMAL = "too_formal"
    TOO_CASUAL = "too_casual"
    NOT_ME = "not_me"
    PRIVATE = "private"
    WRONG = "wrong"


class ActionDecision(StrEnum):
    """What happened to a proposed tool action."""

    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    UNDONE = "undone"
    IGNORED = "ignored"


class ActionRisk(StrEnum):
    """Risk tier for a proposed action."""

    LOW = "low"          # draft/preview only
    MEDIUM = "medium"    # local modification or reversible write
    HIGH = "high"        # send/delete/share/pay/shell/network side effect


class ActionProposalStatus(StrEnum):
    """Lifecycle state for a dry-run action proposal."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    EXECUTED = "executed"
    UNDONE = "undone"
    EXPIRED = "expired"


class ProbeCandidate(BaseModel):
    """One candidate response/action inside a probe."""

    id: str = Field(default_factory=lambda: f"cand-{uuid.uuid4().hex[:12]}")
    origin: CandidateOrigin
    text: str
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionProposal(BaseModel):
    """A dry-run action that must be reviewed before execution.

    This is the action sandbox's core object. The model can propose an action,
    but the system persists a preview and waits for user review. Review creates
    an ActionTrace, which becomes training data.
    """

    id: str = Field(default_factory=lambda: f"prop-{uuid.uuid4().hex[:12]}")
    user_id: str
    surface: str
    operation: str
    prompt: list[Message] = Field(default_factory=list)
    proposed_text: str = ""
    proposed_payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    required_capability: str = ""
    risk_level: ActionRisk = ActionRisk.MEDIUM
    status: ActionProposalStatus = ActionProposalStatus.PROPOSED
    model: str | None = None
    run_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def can_execute_without_step_up(self) -> bool:
        return self.risk_level == ActionRisk.LOW


class PersonalProbe(BaseModel):
    """A single private benchmark item for one user.

    Voice probes look like held-out messages; action probes look like proposed
    tool actions; decision probes are scenario choices. All are shaped so a
    user judgment can become training signal.
    """

    id: str = Field(default_factory=lambda: f"probe-{uuid.uuid4().hex[:12]}")
    user_id: str
    kind: ProbeKind
    prompt: list[Message]
    candidates: list[ProbeCandidate] = Field(default_factory=list)
    reference: str | None = None
    source_completion_id: str | None = None
    dataset_version: str | None = None
    run_id: str | None = None
    surface: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_text(self) -> str:
        """Compact prompt text for UI labels."""
        for msg in reversed(self.prompt):
            if msg.role == Role.USER and msg.content.strip():
                return msg.content.strip()
        return self.prompt[-1].content.strip() if self.prompt else ""


class UserJudgment(BaseModel):
    """A user's judgment on a probe.

    This is the high-value data asset. It can be converted into preference
    pairs, edited-response SFT examples, action-policy supervision, or privacy
    constraints.
    """

    id: str = Field(default_factory=lambda: f"judge-{uuid.uuid4().hex[:12]}")
    user_id: str
    probe_id: str
    verdict: JudgmentVerdict
    chosen_candidate_id: str | None = None
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    edited_text: str | None = None
    reason: str | None = None
    dimension: str = "overall"
    score: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionTrace(BaseModel):
    """One proposed action and how the user handled it.

    MCP access is not the moat by itself. The trace of proposed action →
    edit/approval/rejection is the agency dataset.
    """

    id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:12]}")
    user_id: str
    surface: str
    operation: str
    prompt: list[Message] = Field(default_factory=list)
    proposed_text: str = ""
    proposed_payload: dict[str, Any] = Field(default_factory=dict)
    decision: ActionDecision
    edited_text: str | None = None
    final_payload: dict[str, Any] = Field(default_factory=dict)
    risk_level: ActionRisk = ActionRisk.MEDIUM
    proposal_id: str | None = None
    probe_id: str | None = None
    run_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrustReport(BaseModel):
    """Snapshot of how well the current model has been proven."""

    user_id: str
    total_probes: int = 0
    total_judgments: int = 0
    total_action_traces: int = 0
    voice_approved: int = 0
    voice_total: int = 0
    action_approved: int = 0
    action_total: int = 0
    privacy_flags: int = 0
    scores: dict[str, float] = Field(default_factory=dict)
    readiness: Literal["unproven", "voice", "sandbox", "supervised"] = "unproven"
    generated_at: datetime = Field(default_factory=datetime.now)


def probe_to_preference_completion(
    probe: PersonalProbe,
    judgment: UserJudgment,
) -> Completion | None:
    """Convert a probe judgment into a DPO-ready Completion when possible."""
    if judgment.verdict == JudgmentVerdict.UNSURE:
        return None

    chosen_text: str | None = None
    rejected: list[ProbeCandidate] = []

    if judgment.edited_text and judgment.edited_text.strip():
        chosen_text = judgment.edited_text.strip()
        rejected = [
            c for c in probe.candidates
            if judgment.chosen_candidate_id is None or c.id == judgment.chosen_candidate_id
        ] or probe.candidates[:1]
    elif judgment.chosen_candidate_id:
        chosen = next((c for c in probe.candidates if c.id == judgment.chosen_candidate_id), None)
        if chosen is None:
            return None
        chosen_text = chosen.text
        rejected_ids = set(judgment.rejected_candidate_ids)
        rejected = [c for c in probe.candidates if c.id in rejected_ids and c.id != chosen.id]
        if not rejected:
            rejected = [c for c in probe.candidates if c.id != chosen.id][:1]
    elif judgment.verdict in {JudgmentVerdict.REJECT, JudgmentVerdict.NOT_ME}:
        return None

    if not chosen_text or not rejected:
        return None

    candidates = [
        CompletionCandidate(
            messages=[Message(role=Role.ASSISTANT, content=chosen_text)],
            annotations=[PreferenceAnnotation(chosen=True, dimension=judgment.dimension)],
        )
    ]
    candidates.extend(
        CompletionCandidate(
            messages=[Message(role=Role.ASSISTANT, content=c.text)],
            annotations=[PreferenceAnnotation(chosen=False, dimension=judgment.dimension)],
        )
        for c in rejected
        if c.text.strip()
    )
    if len(candidates) < 2:
        return None

    return Completion(
        conversation=Conversation(
            messages=probe.prompt,
            source_type=SourceType.MANUAL,
        ),
        candidates=candidates,
        annotations=[
            SourceAnnotation(
                source_type="verification",
                source_id=probe.id,
                timestamp=judgment.created_at,
                metadata={
                    "judgment_id": judgment.id,
                    "probe_kind": probe.kind.value,
                    "dimension": judgment.dimension,
                },
            )
        ],
        user_id=probe.user_id,
    )


def action_trace_to_sft_completion(trace: ActionTrace) -> Completion | None:
    """Convert approved/edited tool actions into SFT supervision."""
    if trace.decision not in {ActionDecision.APPROVED, ActionDecision.EDITED}:
        return None
    target = (trace.edited_text or trace.proposed_text).strip()
    if not target:
        return None
    prompt = trace.prompt or [
        Message(
            role=Role.USER,
            content=f"Handle this {trace.surface} action: {trace.operation}.",
        )
    ]
    return Completion(
        conversation=Conversation(messages=prompt, source_type=SourceType.MANUAL),
        candidates=[
            CompletionCandidate(
                messages=[Message(role=Role.ASSISTANT, content=target)],
                annotations=[PreferenceAnnotation(chosen=True, dimension="action")],
            )
        ],
        annotations=[
            SourceAnnotation(
                source_type="action_trace",
                source_id=trace.id,
                timestamp=trace.created_at,
                metadata={
                    "surface": trace.surface,
                    "operation": trace.operation,
                    "decision": trace.decision.value,
                    "proposal_id": trace.proposal_id or "",
                },
            )
        ],
        user_id=trace.user_id,
    )
