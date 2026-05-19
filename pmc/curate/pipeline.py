"""End-to-end curate pipeline.

Conversations come in (from ingest/normalize); curated Completions come out,
each annotated with source, quality, and PII metadata. Filtering decisions are
explicit: configure thresholds, and the pipeline tracks counts of what was
dropped at each stage so the user can see why their training set is the size
it is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pmc.curate.dedup import Deduplicator
from pmc.curate.pii import SEVERE_PII, detect_pii, redact_text
from pmc.curate.quality import HeuristicQualityScorer
from pmc.curate.splitter import split_conversation
from pmc.curate.style_profile import extract_style_profile
from pmc.curate.synthesize import (
    HeuristicSyntheticPrompter,
    SyntheticPrompter,
    attach_synthetic_prompt,
)
from pmc.schema.annotations import PIIType, QualityAnnotation
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
)
from pmc.schema.user import StyleProfile


@dataclass
class CurateConfig:
    clean_boilerplate: bool = True
    min_response_chars: int = 20
    detect_pii: bool = True
    redact_severe_pii: bool = True
    redact_pii_types: set[PIIType] | None = None
    dedup: bool = True
    dedup_threshold: float = 0.85
    min_quality_score: float = 0.3
    synthesize_prompts: bool = True


@dataclass
class CurateStats:
    input_conversations: int = 0
    split_completions: int = 0
    dropped_short: int = 0
    dropped_duplicate: int = 0
    dropped_low_quality: int = 0
    redacted_severe: int = 0
    pii_spans_detected: int = 0
    output_completions: int = 0
    quality_by_bucket: dict[str, int] = field(
        default_factory=lambda: {"high": 0, "mid": 0, "low": 0}
    )


@dataclass
class CurateResult:
    completions: list[Completion]
    style_profile: StyleProfile
    stats: CurateStats


class CuratePipeline:
    """Orchestrates clean → split → PII → dedup → quality → filter → profile."""

    def __init__(
        self,
        config: CurateConfig | None = None,
        prompter: SyntheticPrompter | None = None,
        quality_scorer: HeuristicQualityScorer | None = None,
    ) -> None:
        self.config = config or CurateConfig()
        self.prompter = prompter or HeuristicSyntheticPrompter()
        self.quality_scorer = quality_scorer or HeuristicQualityScorer()
        self.dedup = Deduplicator(threshold=self.config.dedup_threshold)

    def curate(self, conversations: Iterable[Conversation]) -> CurateResult:
        stats = CurateStats()
        kept: list[Completion] = []

        for conv in conversations:
            stats.input_conversations += 1
            completions = split_conversation(
                conv,
                clean=self.config.clean_boilerplate,
                min_response_chars=self.config.min_response_chars,
                include_empty_context=self.config.synthesize_prompts,
            )
            stats.split_completions += len(completions)

            for completion in completions:
                processed = self._process(completion, stats)
                if processed is not None:
                    kept.append(processed)

        stats.output_completions = len(kept)
        for c in kept:
            quality = _find_quality(c)
            if quality is None:
                continue
            bucket = (
                "high" if quality.overall >= 0.7
                else "mid" if quality.overall >= 0.4
                else "low"
            )
            stats.quality_by_bucket[bucket] += 1

        return CurateResult(
            completions=kept,
            style_profile=extract_style_profile(kept),
            stats=stats,
        )

    def _process(self, completion: Completion, stats: CurateStats) -> Completion | None:
        if self.config.synthesize_prompts:
            completion = attach_synthetic_prompt(completion, self.prompter)

        candidate_text = _candidate_text(completion)
        if len(candidate_text) < self.config.min_response_chars:
            stats.dropped_short += 1
            return None

        if self.config.detect_pii:
            completion = self._apply_pii(completion, stats)

        similarity = 0.0
        if self.config.dedup:
            text_for_dedup = _candidate_text(completion)
            is_dup, similarity = self.dedup.check(text_for_dedup)
            if is_dup:
                stats.dropped_duplicate += 1
                return None

        quality = self.quality_scorer.score(completion, similarity=similarity)
        completion = _attach_annotation(completion, quality)

        if quality.overall < self.config.min_quality_score:
            stats.dropped_low_quality += 1
            return None

        return completion

    def _apply_pii(self, completion: Completion, stats: CurateStats) -> Completion:
        types_to_redact: set[PIIType] = set()
        if self.config.redact_severe_pii:
            types_to_redact |= SEVERE_PII
        if self.config.redact_pii_types:
            types_to_redact |= self.config.redact_pii_types

        new_candidates: list[CompletionCandidate] = []
        for cand in completion.candidates:
            new_messages: list[Message] = []
            for msg in cand.messages:
                annotations = detect_pii(msg.content)
                stats.pii_spans_detected += len(annotations)
                content = msg.content
                if types_to_redact and annotations:
                    content, annotations = redact_text(
                        content, annotations, only_types=types_to_redact
                    )
                    stats.redacted_severe += sum(1 for a in annotations if a.redacted)
                new_messages.append(
                    Message(
                        role=msg.role,
                        content=content,
                        timestamp=msg.timestamp,
                        annotations=list(msg.annotations) + list(annotations),
                    )
                )
            new_candidates.append(
                CompletionCandidate(
                    id=cand.id,
                    messages=new_messages,
                    annotations=cand.annotations,
                )
            )
        return Completion(
            id=completion.id,
            conversation=completion.conversation,
            candidates=new_candidates,
            annotations=completion.annotations,
            user_id=completion.user_id,
        )


def _candidate_text(completion: Completion) -> str:
    if not completion.candidates or not completion.candidates[0].messages:
        return ""
    return " ".join(m.content for m in completion.candidates[0].messages)


def _attach_annotation(completion: Completion, quality: QualityAnnotation) -> Completion:
    return Completion(
        id=completion.id,
        conversation=completion.conversation,
        candidates=completion.candidates,
        annotations=list(completion.annotations) + [quality],
        user_id=completion.user_id,
    )


def _find_quality(completion: Completion) -> QualityAnnotation | None:
    for ann in completion.annotations:
        if isinstance(ann, QualityAnnotation):
            return ann
    return None
