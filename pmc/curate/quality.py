"""Quality scoring for training Completions.

Two scorers:
- HeuristicQualityScorer: cheap, deterministic, no API calls. Always runs.
- LLMQualityScorer: uses a judge model to score style signal and coherence.
  Use it on a sampled subset (LLM calls are expensive) — pass a sampler in.

Both produce a QualityAnnotation with normalized [0, 1] scores.
"""

from __future__ import annotations

import re

from pmc.curate.llm import LLMClient
from pmc.schema.annotations import QualityAnnotation
from pmc.schema.conversation import Completion

BOILERPLATE_PHRASES = [
    "thanks in advance",
    "looking forward to hearing",
    "please let me know if",
    "feel free to reach out",
    "do not hesitate to contact",
    "best regards",
    "kind regards",
    "warm regards",
    "to whom it may concern",
    "as per my last email",
]

AUTO_REPLY_MARKERS = [
    "out of office",
    "auto reply",
    "automatic reply",
    "i am currently out",
    "i'm currently away",
    "i will be out",
    "vacation responder",
]


def _candidate_text(completion: Completion) -> str:
    if not completion.candidates or not completion.candidates[0].messages:
        return ""
    return " ".join(m.content for m in completion.candidates[0].messages)


def _context_chars(completion: Completion) -> int:
    return sum(len(m.content) for m in completion.conversation.messages)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?]+\s+", text.strip()) if s]


class HeuristicQualityScorer:
    """Heuristic scorer — fast, no LLM. Run on every completion."""

    def __init__(
        self,
        min_chars: int = 20,
        ideal_chars: int = 400,
        max_chars: int = 4000,
    ) -> None:
        self.min_chars = min_chars
        self.ideal_chars = ideal_chars
        self.max_chars = max_chars

    def score(self, completion: Completion, similarity: float = 0.0) -> QualityAnnotation:
        text = _candidate_text(completion)
        text_lower = text.lower()
        n_chars = len(text)

        style_signal = self._length_score(n_chars)
        coherence = self._coherence_score(text)
        sufficient_context = self._context_score(completion)
        boilerplate = self._boilerplate_score(text_lower)
        duplicate_risk = max(0.0, min(similarity, 1.0))

        overall = max(
            0.0,
            (style_signal * 0.3 + coherence * 0.3 + sufficient_context * 0.2)
            * (1.0 - boilerplate)
            * (1.0 - duplicate_risk),
        )

        return QualityAnnotation(
            style_signal=round(style_signal, 4),
            coherence=round(coherence, 4),
            sufficient_context=round(sufficient_context, 4),
            duplicate_risk=round(duplicate_risk, 4),
            boilerplate_score=round(boilerplate, 4),
            overall=round(overall, 4),
        )

    def _length_score(self, n_chars: int) -> float:
        if n_chars < self.min_chars:
            return 0.0
        if n_chars >= self.max_chars:
            return 0.3
        # Triangular peak at ideal_chars
        if n_chars <= self.ideal_chars:
            return n_chars / self.ideal_chars
        decay = (n_chars - self.ideal_chars) / max(1, self.max_chars - self.ideal_chars)
        return max(0.3, 1.0 - 0.7 * decay)

    def _coherence_score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        sents = _sentences(text)
        if not sents:
            return 0.2
        # Penalize all-caps text, excessive repetition
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        caps_penalty = 1.0 if caps_ratio < 0.4 else max(0.0, 1.5 - caps_ratio * 2)

        words = text.split()
        if len(words) < 3:
            return 0.3 * caps_penalty
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        diversity = min(1.0, unique_ratio * 1.5)

        avg_sent_len = sum(len(s.split()) for s in sents) / len(sents)
        sent_score = 1.0 if 4 <= avg_sent_len <= 30 else 0.6

        return min(1.0, (diversity * 0.5 + sent_score * 0.5) * caps_penalty)

    def _context_score(self, completion: Completion) -> float:
        n = _context_chars(completion)
        if n == 0:
            return 0.2  # standalone writing — usable but needs a synthetic prompt
        if n < 20:
            return 0.4
        return min(1.0, n / 500)

    def _boilerplate_score(self, text_lower: str) -> float:
        if not text_lower:
            return 0.0
        if any(marker in text_lower for marker in AUTO_REPLY_MARKERS):
            return 1.0
        hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in text_lower)
        text_chars = max(1, len(text_lower))
        # Proportion of text taken up by boilerplate phrases
        boilerplate_chars = sum(
            len(p) for p in BOILERPLATE_PHRASES if p in text_lower
        )
        density = boilerplate_chars / text_chars
        return min(1.0, density * 3 + 0.15 * hits)


class LLMQualityScorer:
    """LLM judge for style signal + coherence. Use sparingly (one call per completion)."""

    SYSTEM = (
        "You are evaluating whether a writing sample looks like authentic, "
        "personal writing worth using to train a personal AI model. "
        "Reply with two numbers between 0 and 1, separated by a comma: "
        "style_signal, coherence. Style signal = how much personal voice/style "
        "is present. Coherence = how well-formed the writing is. "
        "Output ONLY the two numbers, nothing else."
    )

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def score(self, completion: Completion) -> tuple[float, float]:
        text = _candidate_text(completion)
        if not text.strip():
            return 0.0, 0.0
        response = self.client.complete(
            system=self.SYSTEM,
            prompt=f"Writing sample:\n\n{text[:2000]}",
            max_tokens=20,
            temperature=0.0,
        )
        return _parse_two_scores(response)


def _parse_two_scores(response: str) -> tuple[float, float]:
    nums = re.findall(r"-?\d*\.?\d+", response)
    if len(nums) < 2:
        return 0.5, 0.5
    a = max(0.0, min(1.0, float(nums[0])))
    b = max(0.0, min(1.0, float(nums[1])))
    return a, b
