"""Durable verification, judgment, and action-trace storage.

This store is intentionally simple: append-only JSONL ledgers under the user's
directory. The artifacts are valuable training data, so they live beside the
curated datasets and bundles instead of as temporary UI state.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from pmc.schema.conversation import Completion
from pmc.schema.verification import (
    ActionDecision,
    ActionProposal,
    ActionProposalStatus,
    ActionTrace,
    JudgmentVerdict,
    PersonalProbe,
    ProbeKind,
    TrustReport,
    UserJudgment,
    action_trace_to_sft_completion,
    probe_to_preference_completion,
)
from pmc.storage.paths import StoragePaths


class VerificationStore:
    """Per-user verification artifacts and derived training signal."""

    def __init__(self, root: Path | str) -> None:
        self.paths = StoragePaths(root)

    # -- probes -----------------------------------------------------------

    def save_probes(
        self,
        user_id: str,
        probes: Iterable[PersonalProbe],
        *,
        append: bool = False,
    ) -> int:
        path = self.paths.verification_probes_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and path.is_file() else "w"
        count = 0
        with path.open(mode, encoding="utf-8") as f:
            for probe in probes:
                f.write(probe.model_dump_json() + "\n")
                count += 1
        return count

    def append_probe(self, user_id: str, probe: PersonalProbe) -> None:
        self.save_probes(user_id, [probe], append=True)

    def list_probes(
        self,
        user_id: str,
        *,
        kind: ProbeKind | str | None = None,
        limit: int | None = None,
    ) -> list[PersonalProbe]:
        probes = list(self._read_jsonl(self.paths.verification_probes_file(user_id), PersonalProbe))
        if kind is not None:
            kind_value = kind.value if isinstance(kind, ProbeKind) else str(kind)
            probes = [p for p in probes if p.kind.value == kind_value]
        if limit is not None and limit >= 0:
            probes = probes[-limit:]
        return probes

    def get_probe(self, user_id: str, probe_id: str) -> PersonalProbe | None:
        for probe in self.list_probes(user_id):
            if probe.id == probe_id:
                return probe
        return None

    # -- judgments --------------------------------------------------------

    def append_judgment(self, user_id: str, judgment: UserJudgment) -> None:
        path = self.paths.verification_judgments_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(judgment.model_dump_json() + "\n")

    def list_judgments(
        self,
        user_id: str,
        *,
        probe_id: str | None = None,
        limit: int | None = None,
    ) -> list[UserJudgment]:
        judgments = list(
            self._read_jsonl(self.paths.verification_judgments_file(user_id), UserJudgment)
        )
        if probe_id is not None:
            judgments = [j for j in judgments if j.probe_id == probe_id]
        if limit is not None and limit >= 0:
            judgments = judgments[-limit:]
        return judgments

    # -- action proposals -------------------------------------------------

    def append_action_proposal(self, user_id: str, proposal: ActionProposal) -> None:
        path = self.paths.action_proposals_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(proposal.model_dump_json() + "\n")

    def list_action_proposals(
        self,
        user_id: str,
        *,
        status: ActionProposalStatus | str | None = None,
        limit: int | None = None,
    ) -> list[ActionProposal]:
        proposals = list(
            self._read_jsonl(self.paths.action_proposals_file(user_id), ActionProposal)
        )
        if status is not None:
            status_value = status.value if isinstance(status, ActionProposalStatus) else str(status)
            proposals = [p for p in proposals if p.status.value == status_value]
        if limit is not None and limit >= 0:
            proposals = proposals[-limit:]
        return proposals

    def get_action_proposal(self, user_id: str, proposal_id: str) -> ActionProposal | None:
        for proposal in self.list_action_proposals(user_id):
            if proposal.id == proposal_id:
                return proposal
        return None

    def update_action_proposal(self, user_id: str, proposal: ActionProposal) -> None:
        proposals = self.list_action_proposals(user_id)
        replaced = False
        for i, existing in enumerate(proposals):
            if existing.id == proposal.id:
                proposals[i] = proposal
                replaced = True
                break
        if not replaced:
            proposals.append(proposal)
        self._write_jsonl(self.paths.action_proposals_file(user_id), proposals)

    # -- action traces ----------------------------------------------------

    def append_action_trace(self, user_id: str, trace: ActionTrace) -> None:
        path = self.paths.action_traces_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def list_action_traces(
        self,
        user_id: str,
        *,
        limit: int | None = None,
    ) -> list[ActionTrace]:
        traces = list(self._read_jsonl(self.paths.action_traces_file(user_id), ActionTrace))
        if limit is not None and limit >= 0:
            traces = traces[-limit:]
        return traces

    # -- derived training signal -----------------------------------------

    def preference_completions(self, user_id: str) -> list[Completion]:
        probes = {p.id: p for p in self.list_probes(user_id)}
        completions: list[Completion] = []
        for judgment in self.list_judgments(user_id):
            probe = probes.get(judgment.probe_id)
            if probe is None:
                continue
            completion = probe_to_preference_completion(probe, judgment)
            if completion is not None:
                completions.append(completion)
        return completions

    def action_sft_completions(self, user_id: str) -> list[Completion]:
        completions: list[Completion] = []
        for trace in self.list_action_traces(user_id):
            completion = action_trace_to_sft_completion(trace)
            if completion is not None:
                completions.append(completion)
        return completions

    def trust_report(self, user_id: str) -> TrustReport:
        probes = {p.id: p for p in self.list_probes(user_id)}
        judgments = self.list_judgments(user_id)
        traces = self.list_action_traces(user_id)

        latest_by_probe: dict[str, UserJudgment] = {}
        for judgment in judgments:
            current = latest_by_probe.get(judgment.probe_id)
            if current is None or judgment.created_at >= current.created_at:
                latest_by_probe[judgment.probe_id] = judgment

        positive_verdicts = {
            JudgmentVerdict.APPROVE,
            JudgmentVerdict.CHOOSE,
            JudgmentVerdict.EDIT,
        }
        privacy_verdicts = {JudgmentVerdict.PRIVATE}

        voice_total = 0
        voice_approved = 0
        action_probe_total = 0
        action_probe_approved = 0
        privacy_flags = 0

        for judgment in latest_by_probe.values():
            probe = probes.get(judgment.probe_id)
            if judgment.verdict in privacy_verdicts:
                privacy_flags += 1
            if probe is None:
                continue
            if probe.kind == ProbeKind.VOICE:
                voice_total += 1
                if judgment.verdict in positive_verdicts:
                    voice_approved += 1
            elif probe.kind == ProbeKind.ACTION:
                action_probe_total += 1
                if judgment.verdict in positive_verdicts:
                    action_probe_approved += 1

        action_total = action_probe_total + len(traces)
        action_approved = action_probe_approved + sum(
            1
            for trace in traces
            if trace.decision in {ActionDecision.APPROVED, ActionDecision.EDITED}
        )

        voice_acceptance = voice_approved / voice_total if voice_total else 0.0
        action_acceptance = action_approved / action_total if action_total else 0.0
        correction_rate = (
            sum(1 for j in judgments if j.verdict == JudgmentVerdict.EDIT) / len(judgments)
            if judgments
            else 0.0
        )

        readiness = "unproven"
        if voice_total >= 3 and voice_acceptance >= 0.7 and privacy_flags == 0:
            readiness = "voice"
        if readiness == "voice" and action_total >= 3 and action_acceptance >= 0.7:
            readiness = "sandbox"
        if (
            readiness == "sandbox"
            and len(traces) >= 10
            and action_acceptance >= 0.85
            and privacy_flags == 0
        ):
            readiness = "supervised"

        return TrustReport(
            user_id=user_id,
            total_probes=len(probes),
            total_judgments=len(judgments),
            total_action_traces=len(traces),
            voice_approved=voice_approved,
            voice_total=voice_total,
            action_approved=action_approved,
            action_total=action_total,
            privacy_flags=privacy_flags,
            scores={
                "voice_acceptance": round(voice_acceptance, 4),
                "action_acceptance": round(action_acceptance, 4),
                "correction_rate": round(correction_rate, 4),
            },
            readiness=readiness,
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _read_jsonl(path: Path, model_cls) -> Iterator:
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield model_cls.model_validate_json(line)

    @staticmethod
    def _write_jsonl(path: Path, items: Iterable) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")


__all__ = ["VerificationStore"]
