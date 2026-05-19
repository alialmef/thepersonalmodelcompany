"""Style match benchmark.

The headline eval — "does it sound like me?" Generate responses to held-out
prompts with both the personal model and the base model, then have an LLM
pairwise judge decide which one is more like the user.

Score is the personal model's win rate (0..1). >0.6 is the deploy threshold
from the analysis doc.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pmc.eval.benchmarks import Benchmark, BenchmarkExample, BenchmarkResult
from pmc.eval.judges import JudgeRequest, PersonalJudge
from pmc.eval.prompts import EvalDimension

if TYPE_CHECKING:
    from pmc.eval.generator import ModelGenerator


class StyleProbe:
    """A single held-out prompt + the user's actual response (if we have it)."""

    def __init__(
        self,
        conversation: list[dict[str, str]],
        reference: str | None = None,
    ) -> None:
        self.conversation = conversation
        self.reference = reference


class StyleMatchBenchmark(Benchmark):
    """Pairwise personal-vs-base style match.

    Requires:
    - `probes`: held-out prompts the user has not been trained on
    - `judge`: usually an LLMPairwiseJudge with debiasing on
    - a base model generator passed at evaluate() time for comparison

    If `base` is None, falls back to comparing against the reference response
    (if probes carry one) — useful for seeing how the model matches actual
    user responses on held-out conversations.
    """

    name = "style_match"
    dimension = EvalDimension.STYLE_MATCH

    def __init__(
        self,
        probes: Sequence[StyleProbe],
        judge: PersonalJudge,
        *,
        user_name: str = "the user",
        user_style_profile: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
    ) -> None:
        self.probes = list(probes)
        self.judge = judge
        self.user_name = user_name
        self.user_style_profile = user_style_profile
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def evaluate(
        self,
        personal: ModelGenerator,
        base: ModelGenerator | None = None,
    ) -> BenchmarkResult:
        t0 = time.time()
        examples: list[BenchmarkExample] = []
        wins = 0
        ties = 0
        losses = 0

        for probe in self.probes:
            personal_resp = personal.generate(
                probe.conversation,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            if base is not None:
                comparison = base.generate(
                    probe.conversation,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                comparison_label = "base"
            elif probe.reference is not None:
                comparison = probe.reference
                comparison_label = "reference"
            else:
                # Nothing to compare against — score this probe as 0
                examples.append(
                    BenchmarkExample(
                        prompt=probe.conversation,
                        personal_response=personal_resp,
                        score=0.0,
                        details={"skipped": True},
                    )
                )
                continue

            # response_b = personal model. score > 0 means personal won.
            judgment = self.judge.judge(
                JudgeRequest(
                    conversation=probe.conversation,
                    response_a=comparison,
                    response_b=personal_resp,
                    user_name=self.user_name,
                    user_style_profile=self.user_style_profile,
                    dimension=self.dimension,
                )
            )
            if judgment.score > 0.5:
                wins += 1
                outcome = "win"
            elif judgment.score < -0.5:
                losses += 1
                outcome = "loss"
            else:
                ties += 1
                outcome = "tie"

            examples.append(
                BenchmarkExample(
                    prompt=probe.conversation,
                    personal_response=personal_resp,
                    base_response=comparison,
                    score=judgment.score,
                    details={
                        "outcome": outcome,
                        "comparison": comparison_label,
                        "confidence": judgment.confidence,
                        "reasoning": judgment.reasoning,
                    },
                )
            )

        scored = wins + ties + losses
        if scored == 0:
            score = 0.0
            win_rate = 0.0
        else:
            # Win rate adjusted for ties (ties count as half)
            score = (wins + 0.5 * ties) / scored
            win_rate = wins / scored

        return BenchmarkResult(
            name=self.name,
            dimension=self.dimension,
            score=round(score, 4),
            num_examples=len(self.probes),
            details={
                "win_rate": round(win_rate, 4),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "scored": scored,
            },
            examples=examples,
            elapsed_seconds=round(time.time() - t0, 2),
        )
