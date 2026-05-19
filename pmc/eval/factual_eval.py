"""Factual accuracy benchmark.

Does the model get user-specific facts right? V0 uses a simple keyword-match
probe: the test author supplies questions and a list of acceptable
phrases/keywords; the score is the fraction of probes where the response
contains at least one required phrase.

For richer factual eval later, swap in an LLM judge that checks entailment
against a reference fact — same Benchmark interface, just a different
checking function.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from pmc.eval.benchmarks import Benchmark, BenchmarkExample, BenchmarkResult
from pmc.eval.prompts import EvalDimension


class FactualProbe:
    """A factual question with a list of accepted answer phrases.

    `must_contain`: at least one of these phrases must appear in the response
    (case-insensitive substring match) for the probe to count as passed.

    `must_not_contain`: any of these phrases appearing causes a fail (use for
    common wrong answers).
    """

    def __init__(
        self,
        question: str,
        must_contain: Sequence[str],
        must_not_contain: Sequence[str] = (),
        prior_context: list[dict[str, str]] | None = None,
    ) -> None:
        self.question = question
        self.must_contain = list(must_contain)
        self.must_not_contain = list(must_not_contain)
        self.prior_context = prior_context or []

    def build_conversation(self) -> list[dict[str, str]]:
        return [*self.prior_context, {"role": "user", "content": self.question}]


CheckFn = Callable[[FactualProbe, str], bool]


def keyword_check(probe: FactualProbe, response: str) -> bool:
    """Default check: at least one must_contain present, no must_not_contain."""
    text = response.lower()
    for bad in probe.must_not_contain:
        if bad.lower() in text:
            return False
    if not probe.must_contain:
        return True
    return any(good.lower() in text for good in probe.must_contain)


class FactualAccuracyBenchmark(Benchmark):
    name = "factual_accuracy"
    dimension = EvalDimension.FACTUAL_ACCURACY

    def __init__(
        self,
        probes: Sequence[FactualProbe],
        *,
        check_fn: CheckFn = keyword_check,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.probes = list(probes)
        self.check_fn = check_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def evaluate(self, personal, base=None) -> BenchmarkResult:
        t0 = time.time()
        examples: list[BenchmarkExample] = []
        passes = 0

        for probe in self.probes:
            conversation = probe.build_conversation()
            response = personal.generate(
                conversation,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            passed = self.check_fn(probe, response)
            if passed:
                passes += 1
            examples.append(
                BenchmarkExample(
                    prompt=conversation,
                    personal_response=response,
                    score=1.0 if passed else 0.0,
                    details={
                        "passed": passed,
                        "must_contain": probe.must_contain,
                        "must_not_contain": probe.must_not_contain,
                    },
                )
            )

        total = len(self.probes)
        score = passes / total if total else 0.0
        return BenchmarkResult(
            name=self.name,
            dimension=self.dimension,
            score=round(score, 4),
            num_examples=total,
            details={"passes": passes, "fails": total - passes},
            examples=examples,
            elapsed_seconds=round(time.time() - t0, 2),
        )
