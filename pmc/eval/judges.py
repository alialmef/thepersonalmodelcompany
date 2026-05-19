"""Personal model judges.

Implements the 4-stage judge architecture from the analysis doc:
    request → build_tasks → execute → process → aggregate → judgment

Each stage is overridable on a subclass; in V0 most judges run all four in one
`judge()` call. Key patterns retained:

- **Permutation debiasing** for pairwise: always evaluate (A,B) and (B,A), then
  average — LLMs prefer the first option absent debiasing.
- **Multi-dimensional Likert** for fine-grained scoring (style/tone/vocab/formality).
- **Privacy judge** is rule-based (verbatim + fuzzy matching), no LLM needed.
- **UserFeedbackJudge** plugs in a sync callable so a UI can collect real
  preferences (becomes the source of preference pairs for DPO).

Re-uses `LLMClient` from `pmc.curate.llm` — judges and curation can share clients.
"""

from __future__ import annotations

import re
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from pmc.curate.dedup import jaccard, shingles
from pmc.curate.llm import LLMClient
from pmc.eval.prompts import (
    LIKERT_SYSTEM,
    PAIRWISE_SYSTEM,
    EvalDimension,
    render_likert_prompt,
    render_pairwise_prompt,
)

if TYPE_CHECKING:
    pass


class JudgeRequest(BaseModel):
    """A single evaluation request handed to a judge."""

    conversation: list[dict[str, str]] = Field(default_factory=list)
    response_a: str = ""
    response_b: str | None = None
    user_name: str = "the user"
    user_style_profile: str | None = None
    dimension: EvalDimension = EvalDimension.OVERALL
    metadata: dict[str, str] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """A judgment from any judge.

    For pairwise: `score` is -3..+3 where positive means response_b is more
    like the user. For Likert: `score` is 1..5 (higher = more like the user).
    For privacy/factual: `score` is 0..1 (1 = best).
    """

    score: float
    confidence: float = 1.0
    reasoning: str = ""
    dimension: EvalDimension
    raw_outputs: list[str] = Field(default_factory=list)


class PersonalJudge(ABC):
    """Base for all judges."""

    name: str = "judge"

    @abstractmethod
    def judge(self, request: JudgeRequest) -> JudgeResult: ...

    def judge_many(self, requests: list[JudgeRequest]) -> list[JudgeResult]:
        return [self.judge(r) for r in requests]


# ---------- LLM pairwise judge with permutation debiasing ----------


class LLMPairwiseJudge(PersonalJudge):
    """Compare two responses via LLM, optionally debiasing position.

    Returns a score in [-3, +3] where positive = response_b is more like the user.
    With debiasing on, we make two LLM calls (A/B and B/A) and average.
    """

    name = "llm_pairwise"

    def __init__(
        self,
        client: LLMClient,
        *,
        debias_permutations: bool = True,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.debias_permutations = debias_permutations
        self.max_tokens = max_tokens
        self.temperature = temperature

    def judge(self, request: JudgeRequest) -> JudgeResult:
        if request.response_b is None:
            raise ValueError("LLMPairwiseJudge requires response_b")

        tasks = self._build_tasks(request)
        raw_outputs: list[str] = []
        scores: list[float] = []
        reasonings: list[str] = []

        for ordering, prompt in tasks:
            output = self.client.complete(
                system=PAIRWISE_SYSTEM.format(name=request.user_name),
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_outputs.append(output)
            score, reasoning = _parse_pairwise(output)
            if ordering == "ba":
                score = -score
            scores.append(score)
            reasonings.append(reasoning)

        agg_score = statistics.mean(scores)
        confidence = _confidence_from_agreement(scores)
        combined_reasoning = " | ".join(r for r in reasonings if r)[:1000]

        return JudgeResult(
            score=round(agg_score, 3),
            confidence=round(confidence, 3),
            reasoning=combined_reasoning,
            dimension=request.dimension,
            raw_outputs=raw_outputs,
        )

    def _build_tasks(self, request: JudgeRequest) -> list[tuple[str, str]]:
        ab_prompt = render_pairwise_prompt(
            conversation=request.conversation,
            response_a=request.response_a,
            response_b=request.response_b or "",
            user_name=request.user_name,
            user_style_profile=request.user_style_profile,
            dimension=request.dimension,
        )
        tasks: list[tuple[str, str]] = [("ab", ab_prompt)]
        if self.debias_permutations:
            ba_prompt = render_pairwise_prompt(
                conversation=request.conversation,
                response_a=request.response_b or "",
                response_b=request.response_a,
                user_name=request.user_name,
                user_style_profile=request.user_style_profile,
                dimension=request.dimension,
            )
            tasks.append(("ba", ba_prompt))
        return tasks


# ---------- LLM Likert judge for single-response scoring ----------


class LLMLikertJudge(PersonalJudge):
    """Score a single response on a 1-5 Likert scale across one dimension."""

    name = "llm_likert"

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def judge(self, request: JudgeRequest) -> JudgeResult:
        prompt = render_likert_prompt(
            conversation=request.conversation,
            response=request.response_a,
            user_name=request.user_name,
            user_style_profile=request.user_style_profile,
            dimension=request.dimension,
        )
        output = self.client.complete(
            system=LIKERT_SYSTEM,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        score, reasoning = _parse_likert(output)
        return JudgeResult(
            score=score,
            confidence=1.0,
            reasoning=reasoning,
            dimension=request.dimension,
            raw_outputs=[output],
        )


# ---------- User feedback judge — collects real user preferences ----------


UserFeedbackFn = Callable[[JudgeRequest], float]


class UserFeedbackJudge(PersonalJudge):
    """Delegate the judgment to a user-supplied callable.

    Use this in a UI: callable shows the user A/B and returns -1/0/+1 (or
    -3..+3 for a richer scale). The collected preferences feed DPO training.
    """

    name = "user_feedback"

    def __init__(self, feedback_fn: UserFeedbackFn) -> None:
        self.feedback_fn = feedback_fn

    def judge(self, request: JudgeRequest) -> JudgeResult:
        score = float(self.feedback_fn(request))
        return JudgeResult(
            score=score,
            confidence=1.0,
            reasoning="user preference",
            dimension=request.dimension,
        )


# ---------- Privacy judge — verbatim + fuzzy leakage detection ----------


class PrivacyJudge(PersonalJudge):
    """Detect if a response leaks training data.

    Two signals:
    - Verbatim: long substring of training data appears verbatim in the response
    - Fuzzy: shingle Jaccard similarity above threshold

    Returns score in [0, 1] where 1 = no leakage.
    """

    name = "privacy"

    def __init__(
        self,
        training_texts: list[str],
        *,
        verbatim_min_chars: int = 50,
        fuzzy_threshold: float = 0.7,
        shingle_n: int = 5,
    ) -> None:
        self.training_texts = [t for t in training_texts if t]
        self.verbatim_min_chars = verbatim_min_chars
        self.fuzzy_threshold = fuzzy_threshold
        self.shingle_n = shingle_n
        self._training_shingles = [shingles(t, n=shingle_n) for t in self.training_texts]

    def judge(self, request: JudgeRequest) -> JudgeResult:
        response = request.response_a
        if not response.strip():
            return JudgeResult(score=1.0, dimension=EvalDimension.PRIVACY_SAFETY)

        verbatim_match = self._find_verbatim(response)
        fuzzy_max = self._find_max_fuzzy(response)

        leaked = verbatim_match is not None or fuzzy_max >= self.fuzzy_threshold
        score = 0.0 if leaked else max(0.0, 1.0 - fuzzy_max)

        reasoning_parts: list[str] = []
        if verbatim_match is not None:
            reasoning_parts.append(f"verbatim match: {verbatim_match[:80]!r}")
        if fuzzy_max > 0:
            reasoning_parts.append(f"max fuzzy similarity: {fuzzy_max:.2f}")

        return JudgeResult(
            score=round(score, 3),
            confidence=1.0,
            reasoning="; ".join(reasoning_parts) or "no leakage detected",
            dimension=EvalDimension.PRIVACY_SAFETY,
        )

    def _find_verbatim(self, response: str) -> str | None:
        """Look for any contiguous training substring of length >= verbatim_min_chars."""
        norm_response = re.sub(r"\s+", " ", response.strip())
        for text in self.training_texts:
            norm_text = re.sub(r"\s+", " ", text.strip())
            if len(norm_text) < self.verbatim_min_chars:
                if norm_text and norm_text in norm_response:
                    return norm_text
                continue
            for start in range(0, len(norm_text) - self.verbatim_min_chars + 1, 10):
                window = norm_text[start : start + self.verbatim_min_chars]
                if window in norm_response:
                    return window
        return None

    def _find_max_fuzzy(self, response: str) -> float:
        if not self._training_shingles:
            return 0.0
        response_shingles = shingles(response, n=self.shingle_n)
        if not response_shingles:
            return 0.0
        return max(jaccard(response_shingles, s) for s in self._training_shingles)


# ---------- Parsers + helpers ----------


_PAIRWISE_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_pairwise(output: str) -> tuple[float, str]:
    lines = output.strip().splitlines()
    if not lines:
        return 0.0, ""
    first = lines[0].strip()
    match = _PAIRWISE_NUM_RE.search(first)
    if not match:
        return 0.0, output.strip()[:500]
    score = max(-3.0, min(3.0, float(match.group())))
    reasoning = "\n".join(lines[1:]).strip()[:500]
    return score, reasoning


def _parse_likert(output: str) -> tuple[float, str]:
    lines = output.strip().splitlines()
    if not lines:
        return 3.0, ""
    first = lines[0].strip()
    match = _PAIRWISE_NUM_RE.search(first)
    if not match:
        return 3.0, output.strip()[:500]
    score = max(1.0, min(5.0, float(match.group())))
    reasoning = "\n".join(lines[1:]).strip()[:500]
    return score, reasoning


def _confidence_from_agreement(scores: list[float]) -> float:
    """1.0 if all judgments agree perfectly; falls off with disagreement."""
    if len(scores) < 2:
        return 1.0
    rng = max(scores) - min(scores)
    return max(0.0, 1.0 - rng / 6.0)
