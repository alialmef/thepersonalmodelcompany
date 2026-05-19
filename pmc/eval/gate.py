"""EvalGate — pass/fail decision for deploying an adapter.

From the analysis doc:

    Train adapter
      → Run eval suite
        → Style match > 0.6? ✓
        → Privacy score > 0.95? ✓
        → Factual accuracy > 0.7? ✓
      → All pass? → Deploy adapter
      → Any fail? → Flag for review, don't deploy

This file is the gate. It takes an EvalSuiteResult + thresholds, returns a
GateDecision the orchestrator (or web app) can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pmc.eval.runner import EvalSuiteResult


@dataclass
class EvalGateConfig:
    """Per-benchmark deploy thresholds. Defaults match the analysis doc."""

    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "style_match": 0.6,
            "privacy": 0.95,
            "factual_accuracy": 0.7,
        }
    )
    required: list[str] = field(default_factory=lambda: ["style_match", "privacy"])
    treat_missing_as: Literal["pass", "fail", "ignore"] = "ignore"


@dataclass
class GateCheck:
    name: str
    score: float | None
    threshold: float
    passed: bool


@dataclass
class GateDecision:
    deploy: bool
    checks: list[GateCheck]
    failed: list[str] = field(default_factory=list)
    reason: str = ""


class EvalGate:
    """Decide whether an adapter is good enough to deploy."""

    def __init__(self, config: EvalGateConfig | None = None) -> None:
        self.config = config or EvalGateConfig()

    def decide(self, result: EvalSuiteResult) -> GateDecision:
        scores = result.to_summary()
        checks: list[GateCheck] = []
        failed: list[str] = []

        for name, threshold in self.config.thresholds.items():
            score = scores.get(name)
            if score is None:
                if name in self.config.required:
                    handling = self.config.treat_missing_as
                    if handling == "fail":
                        checks.append(GateCheck(name=name, score=None, threshold=threshold, passed=False))
                        failed.append(f"{name} (missing)")
                        continue
                    if handling == "pass":
                        checks.append(GateCheck(name=name, score=None, threshold=threshold, passed=True))
                        continue
                checks.append(GateCheck(name=name, score=None, threshold=threshold, passed=True))
                continue

            passed = score >= threshold
            checks.append(GateCheck(name=name, score=score, threshold=threshold, passed=passed))
            if not passed and name in self.config.required:
                failed.append(f"{name} ({score:.3f} < {threshold:.3f})")

        deploy = not failed
        reason = "All required checks passed." if deploy else f"Failed: {', '.join(failed)}"
        return GateDecision(deploy=deploy, checks=checks, failed=failed, reason=reason)
