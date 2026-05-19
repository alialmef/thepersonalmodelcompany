"""Evaluation harness: style match, factual accuracy, privacy probes."""

from pmc.eval.benchmarks import Benchmark, BenchmarkExample, BenchmarkResult
from pmc.eval.factual_eval import FactualAccuracyBenchmark, FactualProbe, keyword_check
from pmc.eval.gate import EvalGate, EvalGateConfig, GateCheck, GateDecision
from pmc.eval.generator import (
    CallableGenerator,
    HFGenerator,
    MockGenerator,
    ModelGenerator,
)
from pmc.eval.judges import (
    JudgeRequest,
    JudgeResult,
    LLMLikertJudge,
    LLMPairwiseJudge,
    PersonalJudge,
    PrivacyJudge,
    UserFeedbackJudge,
)
from pmc.eval.preference_eval import PreferenceAlignBenchmark, RewardScorer
from pmc.eval.privacy_eval import PrivacyBenchmark
from pmc.eval.prompts import (
    DIMENSION_PROMPTS,
    LIKERT_SYSTEM,
    PAIRWISE_SYSTEM,
    EvalDimension,
    render_likert_prompt,
    render_pairwise_prompt,
)
from pmc.eval.runner import EvalSuiteResult, PersonalEvalRunner
from pmc.eval.style_eval import StyleMatchBenchmark, StyleProbe

__all__ = [
    "Benchmark",
    "BenchmarkExample",
    "BenchmarkResult",
    "CallableGenerator",
    "DIMENSION_PROMPTS",
    "EvalDimension",
    "EvalGate",
    "EvalGateConfig",
    "EvalSuiteResult",
    "FactualAccuracyBenchmark",
    "FactualProbe",
    "GateCheck",
    "GateDecision",
    "HFGenerator",
    "JudgeRequest",
    "JudgeResult",
    "LIKERT_SYSTEM",
    "LLMLikertJudge",
    "LLMPairwiseJudge",
    "MockGenerator",
    "ModelGenerator",
    "PAIRWISE_SYSTEM",
    "PersonalEvalRunner",
    "PersonalJudge",
    "PreferenceAlignBenchmark",
    "PrivacyBenchmark",
    "PrivacyJudge",
    "RewardScorer",
    "StyleMatchBenchmark",
    "StyleProbe",
    "UserFeedbackJudge",
    "keyword_check",
    "render_likert_prompt",
    "render_pairwise_prompt",
]
