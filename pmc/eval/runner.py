"""PersonalEvalRunner — run all benchmarks against a personal model.

Holds an ordered list of benchmarks, calls each one, and returns an
EvalSuiteResult that serializes naturally into the eval_report.json of an
ArtifactBundle.
"""

from __future__ import annotations

import time
from datetime import datetime

from pydantic import BaseModel, Field

from pmc.eval.benchmarks import Benchmark, BenchmarkResult
from pmc.eval.generator import ModelGenerator


class EvalSuiteResult(BaseModel):
    """Aggregated output of running all benchmarks. Drops cleanly into bundle.eval_report."""

    user_id: str | None = None
    adapter_dir: str | None = None
    base_model: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    elapsed_seconds: float = 0.0
    results: list[BenchmarkResult] = Field(default_factory=list)

    def score_by_name(self, name: str) -> float | None:
        for r in self.results:
            if r.name == name:
                return r.score
        return None

    def to_summary(self) -> dict[str, float]:
        return {r.name: r.score for r in self.results}


class PersonalEvalRunner:
    """Run a set of benchmarks against a personal model."""

    def __init__(
        self,
        benchmarks: list[Benchmark],
        *,
        user_id: str | None = None,
        adapter_dir: str | None = None,
        base_model: str | None = None,
    ) -> None:
        self.benchmarks = benchmarks
        self.user_id = user_id
        self.adapter_dir = adapter_dir
        self.base_model = base_model

    def run(
        self,
        personal: ModelGenerator,
        base: ModelGenerator | None = None,
    ) -> EvalSuiteResult:
        started = datetime.now()
        t0 = time.time()
        results: list[BenchmarkResult] = []
        for benchmark in self.benchmarks:
            results.append(benchmark.evaluate(personal, base))
        return EvalSuiteResult(
            user_id=self.user_id,
            adapter_dir=self.adapter_dir,
            base_model=self.base_model,
            started_at=started,
            completed_at=datetime.now(),
            elapsed_seconds=round(time.time() - t0, 2),
            results=results,
        )
