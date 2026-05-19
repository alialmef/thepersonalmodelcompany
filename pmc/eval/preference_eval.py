"""Preference alignment benchmark.

If the user has a trained reward model (see `pmc/train/reward_model.py`), this
benchmark scores model outputs with it and tracks how well-aligned they are.

For V0, the RewardScorer abstraction is a Protocol — callers pass any callable
that returns a scalar reward for a (prompt, response) pair. A real RM-backed
scorer is a one-liner once `train/reward_model.py` is wired to inference.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pmc.eval.benchmarks import Benchmark, BenchmarkExample, BenchmarkResult
from pmc.eval.prompts import EvalDimension

if TYPE_CHECKING:
    from pmc.eval.generator import ModelGenerator


@runtime_checkable
class RewardScorer(Protocol):
    """Score how well a response matches the user's preferences."""

    def score(self, conversation: list[dict[str, str]], response: str) -> float: ...


class PreferenceAlignBenchmark(Benchmark):
    """Average reward-model score across held-out prompts.

    Score is normalized: assumes reward model outputs in roughly [-5, +5] and
    rescales to [0, 1]. Override `_normalize` if your RM uses a different range.
    """

    name = "preference_align"
    dimension = EvalDimension.OVERALL

    def __init__(
        self,
        prompts: Sequence[list[dict[str, str]]],
        scorer: RewardScorer,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        reward_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.prompts = list(prompts)
        self.scorer = scorer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.reward_range = reward_range

    def evaluate(
        self,
        personal: ModelGenerator,
        base: ModelGenerator | None = None,
    ) -> BenchmarkResult:
        t0 = time.time()
        examples: list[BenchmarkExample] = []
        rewards: list[float] = []
        base_rewards: list[float] = []

        for prompt in self.prompts:
            response = personal.generate(
                prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature
            )
            reward = float(self.scorer.score(prompt, response))
            rewards.append(reward)

            base_resp: str | None = None
            base_r: float | None = None
            if base is not None:
                base_resp = base.generate(
                    prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature
                )
                base_r = float(self.scorer.score(prompt, base_resp))
                base_rewards.append(base_r)

            examples.append(
                BenchmarkExample(
                    prompt=prompt,
                    personal_response=response,
                    base_response=base_resp,
                    score=reward,
                    details={
                        "raw_reward": reward,
                        "base_reward": base_r,
                    },
                )
            )

        if not rewards:
            return BenchmarkResult(
                name=self.name,
                dimension=self.dimension,
                score=0.0,
                num_examples=0,
                elapsed_seconds=round(time.time() - t0, 2),
            )

        avg_reward = sum(rewards) / len(rewards)
        details: dict[str, float] = {
            "mean_reward": round(avg_reward, 4),
            "max_reward": round(max(rewards), 4),
            "min_reward": round(min(rewards), 4),
        }
        if base_rewards:
            details["mean_base_reward"] = round(sum(base_rewards) / len(base_rewards), 4)
            details["margin_vs_base"] = round(
                avg_reward - details["mean_base_reward"], 4
            )

        return BenchmarkResult(
            name=self.name,
            dimension=self.dimension,
            score=round(self._normalize(avg_reward), 4),
            num_examples=len(self.prompts),
            details=details,
            examples=examples,
            elapsed_seconds=round(time.time() - t0, 2),
        )

    def _normalize(self, reward: float) -> float:
        lo, hi = self.reward_range
        if hi <= lo:
            return 0.5
        clamped = max(lo, min(hi, reward))
        return (clamped - lo) / (hi - lo)
