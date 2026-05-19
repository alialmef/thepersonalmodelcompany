"""Benchmark protocol and shared result types.

Every benchmark takes a personal-model generator (and optionally a base-model
generator for comparison), runs against a set of probes, and returns a
`BenchmarkResult` with a 0..1 normalized score and per-example details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from pmc.eval.generator import ModelGenerator
from pmc.eval.prompts import EvalDimension


class BenchmarkExample(BaseModel):
    """One example from a benchmark run — what was asked, what came back, how it scored."""

    prompt: list[dict[str, str]] = Field(default_factory=list)
    personal_response: str = ""
    base_response: str | None = None
    score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """The output of a single benchmark."""

    name: str
    dimension: EvalDimension
    score: float                                # 0..1, higher is better
    num_examples: int
    details: dict[str, Any] = Field(default_factory=dict)
    examples: list[BenchmarkExample] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class Benchmark(ABC):
    """Base for all PMC benchmarks."""

    name: str
    dimension: EvalDimension

    @abstractmethod
    def evaluate(
        self,
        personal: ModelGenerator,
        base: ModelGenerator | None = None,
    ) -> BenchmarkResult: ...
