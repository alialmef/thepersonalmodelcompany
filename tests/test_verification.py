"""Tests for private verification artifacts and training-signal conversion."""

from __future__ import annotations

from pathlib import Path

from pmc.schema import (
    ActionDecision,
    ActionProposal,
    ActionProposalStatus,
    ActionRisk,
    ActionTrace,
    CandidateOrigin,
    JudgmentVerdict,
    Message,
    PersonalProbe,
    ProbeCandidate,
    ProbeKind,
    Role,
    UserJudgment,
    action_trace_to_sft_completion,
    probe_to_preference_completion,
)
from pmc.storage import VerificationStore
from pmc.train.formatter import completion_to_dpo_pair, completion_to_messages


def _voice_probe(user_id: str = "u") -> PersonalProbe:
    candidate = ProbeCandidate(
        id="cand-model",
        origin=CandidateOrigin.PERSONAL_MODEL,
        text="Sounds good, I can do Thursday.",
        model="mock/base",
    )
    return PersonalProbe(
        id="probe-voice",
        user_id=user_id,
        kind=ProbeKind.VOICE,
        prompt=[Message(role=Role.USER, content="free for dinner thursday?")],
        candidates=[candidate],
        reference="yeah thursday works",
    )


def test_probe_edit_judgment_becomes_dpo_completion():
    probe = _voice_probe()
    judgment = UserJudgment(
        user_id="u",
        probe_id=probe.id,
        verdict=JudgmentVerdict.EDIT,
        chosen_candidate_id="cand-model",
        edited_text="yeah thursday works",
        dimension="voice",
    )

    completion = probe_to_preference_completion(probe, judgment)
    assert completion is not None
    pair = completion_to_dpo_pair(completion)
    assert pair is not None
    assert pair["chosen"][0]["content"] == "yeah thursday works"
    assert pair["rejected"][0]["content"] == "Sounds good, I can do Thursday."


def test_action_trace_becomes_sft_completion():
    trace = ActionTrace(
        user_id="u",
        surface="mail",
        operation="draft_reply",
        prompt=[Message(role=Role.USER, content="Reply to Maya")],
        proposed_text="Sure, Thursday works.",
        decision=ActionDecision.EDITED,
        edited_text="yeah thursday works",
    )

    completion = action_trace_to_sft_completion(trace)
    assert completion is not None
    messages = completion_to_messages(completion)
    assert messages is not None
    assert messages[-1]["content"] == "yeah thursday works"


def test_verification_store_persists_and_reports_training_signal(tmp_path: Path):
    store = VerificationStore(tmp_path)
    probes = [
        _voice_probe("u").model_copy(update={"id": f"probe-{i}"})
        for i in range(3)
    ]
    assert store.save_probes("u", probes) == 3

    for probe in probes:
        store.append_judgment(
            "u",
            UserJudgment(
                user_id="u",
                probe_id=probe.id,
                verdict=JudgmentVerdict.EDIT,
                chosen_candidate_id="cand-model",
                edited_text="yeah thursday works",
                dimension="voice",
            ),
        )

    store.append_action_trace(
        "u",
        ActionTrace(
            user_id="u",
            surface="messages",
            operation="draft_reply",
            proposed_text="yeah thursday works",
            decision=ActionDecision.APPROVED,
        ),
    )

    report = store.trust_report("u")
    assert report.voice_total == 3
    assert report.voice_approved == 3
    assert report.readiness == "voice"
    assert len(store.preference_completions("u")) == 3
    assert len(store.action_sft_completions("u")) == 1


def test_action_proposal_updates_and_review_trace_training_signal(tmp_path: Path):
    store = VerificationStore(tmp_path)
    proposal = ActionProposal(
        user_id="u",
        surface="mail",
        operation="draft_reply",
        proposed_text="Sure, Thursday works.",
        risk_level=ActionRisk.LOW,
    )
    store.append_action_proposal("u", proposal)

    proposal = proposal.model_copy(update={"status": ActionProposalStatus.EDITED})
    store.update_action_proposal("u", proposal)
    loaded = store.get_action_proposal("u", proposal.id)
    assert loaded is not None
    assert loaded.status == ActionProposalStatus.EDITED

    store.append_action_trace(
        "u",
        ActionTrace(
            user_id="u",
            surface=proposal.surface,
            operation=proposal.operation,
            proposed_text=proposal.proposed_text,
            decision=ActionDecision.EDITED,
            edited_text="yeah thursday works",
            risk_level=proposal.risk_level,
            proposal_id=proposal.id,
        ),
    )
    completion = store.action_sft_completions("u")[0]
    assert completion.candidates[0].messages[0].content == "yeah thursday works"
