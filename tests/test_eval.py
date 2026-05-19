"""Tests for the eval layer.

All tests run without torch — model generation uses MockGenerator, LLM judges
use MockLLMClient. Real-model integration is tested separately when a GPU is
available.
"""

from __future__ import annotations

import pytest

from pmc.curate.llm import MockLLMClient
from pmc.eval import (
    EvalDimension,
    EvalGate,
    EvalGateConfig,
    EvalSuiteResult,
    FactualAccuracyBenchmark,
    FactualProbe,
    JudgeRequest,
    LLMLikertJudge,
    LLMPairwiseJudge,
    MockGenerator,
    PersonalEvalRunner,
    PreferenceAlignBenchmark,
    PrivacyBenchmark,
    PrivacyJudge,
    StyleMatchBenchmark,
    StyleProbe,
    UserFeedbackJudge,
    render_likert_prompt,
    render_pairwise_prompt,
)
from pmc.eval.judges import _confidence_from_agreement, _parse_likert, _parse_pairwise


# ---------- Prompt templates ----------


def test_render_pairwise_prompt_contains_both_responses():
    prompt = render_pairwise_prompt(
        conversation=[{"role": "user", "content": "What do you think?"}],
        response_a="formal answer here",
        response_b="casual answer here",
        user_name="Alex",
        user_style_profile="warm and direct",
        dimension=EvalDimension.STYLE_MATCH,
    )
    assert "formal answer here" in prompt
    assert "casual answer here" in prompt
    assert "Alex" in prompt
    assert "warm and direct" in prompt
    assert "-3" in prompt and "+3" in prompt


def test_render_likert_prompt_uses_dimension():
    prompt = render_likert_prompt(
        conversation=[{"role": "user", "content": "hi"}],
        response="some response",
        user_name="Sam",
        user_style_profile=None,
        dimension=EvalDimension.TONE_MATCH,
    )
    assert "Sam" in prompt
    assert "tone" in prompt.lower()
    assert "Score 1-5" in prompt


def test_render_pairwise_handles_empty_conversation():
    prompt = render_pairwise_prompt(
        conversation=[],
        response_a="a",
        response_b="b",
        user_name="Alex",
        user_style_profile=None,
        dimension=EvalDimension.OVERALL,
    )
    assert "no prior context" in prompt


# ---------- Parsers ----------


def test_parse_pairwise_clamps_to_range():
    score, _ = _parse_pairwise("-5\nReason here")
    assert score == -3.0
    score, _ = _parse_pairwise("99\n")
    assert score == 3.0


def test_parse_pairwise_extracts_reasoning():
    score, reasoning = _parse_pairwise("+2\nBecause it sounds more like them.")
    assert score == 2.0
    assert "sounds more like them" in reasoning


def test_parse_pairwise_garbage_returns_zero():
    score, _ = _parse_pairwise("hmm not sure")
    assert score == 0.0


def test_parse_likert_clamps():
    score, _ = _parse_likert("0\n")
    assert score == 1.0
    score, _ = _parse_likert("9\n")
    assert score == 5.0


def test_confidence_decays_with_disagreement():
    assert _confidence_from_agreement([2.0, 2.0]) == 1.0
    assert _confidence_from_agreement([1.0, 2.0]) == pytest.approx(1 - 1 / 6)
    assert _confidence_from_agreement([-3.0, 3.0]) == 0.0


# ---------- LLMPairwiseJudge ----------


def test_pairwise_judge_debiases_by_averaging_two_orderings():
    """If LLM always says '+2' regardless of order, debiased score should be 0."""
    client = MockLLMClient(default="+2\nReasoning")
    judge = LLMPairwiseJudge(client, debias_permutations=True)
    result = judge.judge(
        JudgeRequest(
            conversation=[{"role": "user", "content": "Q"}],
            response_a="response one",
            response_b="response two",
            user_name="Alex",
            dimension=EvalDimension.STYLE_MATCH,
        )
    )
    # forward says +2 (B better), backward says +2 (now A is the "real" B), so we invert → -2
    # average is 0
    assert result.score == 0.0
    assert len(client.calls) == 2


def test_pairwise_judge_with_consistent_judgment():
    """Real judgment: B is genuinely better in both orderings."""
    client = MockLLMClient(
        responses={
            "response one\n\nResponse 2:\nresponse two": "+2\nB is better",
            "response two\n\nResponse 2:\nresponse one": "-2\nA was better",
        },
        default="0\n",
    )
    judge = LLMPairwiseJudge(client, debias_permutations=True)
    result = judge.judge(
        JudgeRequest(
            conversation=[{"role": "user", "content": "Q"}],
            response_a="response one",
            response_b="response two",
            user_name="Alex",
            dimension=EvalDimension.STYLE_MATCH,
        )
    )
    # Forward: +2 (B better). Backward: -2 (A=original B better), inverted to +2.
    # Mean = +2 → B genuinely wins.
    assert result.score == 2.0
    assert result.confidence == 1.0


def test_pairwise_judge_no_debias_single_call():
    client = MockLLMClient(default="+1\n")
    judge = LLMPairwiseJudge(client, debias_permutations=False)
    result = judge.judge(
        JudgeRequest(
            response_a="a",
            response_b="b",
            dimension=EvalDimension.STYLE_MATCH,
        )
    )
    assert len(client.calls) == 1
    assert result.score == 1.0


def test_pairwise_judge_requires_response_b():
    judge = LLMPairwiseJudge(MockLLMClient())
    with pytest.raises(ValueError):
        judge.judge(JudgeRequest(response_a="only one", dimension=EvalDimension.STYLE_MATCH))


# ---------- LLMLikertJudge ----------


def test_likert_judge_returns_score_and_reasoning():
    client = MockLLMClient(default="4\nVery much like them")
    judge = LLMLikertJudge(client)
    result = judge.judge(
        JudgeRequest(
            response_a="some text",
            user_name="Alex",
            dimension=EvalDimension.STYLE_MATCH,
        )
    )
    assert result.score == 4.0
    assert "like them" in result.reasoning


# ---------- UserFeedbackJudge ----------


def test_user_feedback_judge_uses_callable():
    judge = UserFeedbackJudge(lambda req: 1.0)
    result = judge.judge(JudgeRequest(response_a="a", dimension=EvalDimension.OVERALL))
    assert result.score == 1.0
    assert result.reasoning == "user preference"


# ---------- PrivacyJudge ----------


def test_privacy_judge_detects_verbatim_leak():
    training = ["My credit card number is 4111 2222 3333 4444 and I should not share it"]
    judge = PrivacyJudge(training_texts=training, verbatim_min_chars=20)
    result = judge.judge(
        JudgeRequest(
            response_a="I cannot tell you. My credit card number is 4111 2222 3333 4444 anyway.",
            dimension=EvalDimension.PRIVACY_SAFETY,
        )
    )
    assert result.score < 0.5
    assert "verbatim" in result.reasoning


def test_privacy_judge_detects_fuzzy_leak():
    training = ["The launch is scheduled for October fifteenth at noon eastern time precisely"]
    judge = PrivacyJudge(training_texts=training, fuzzy_threshold=0.5, verbatim_min_chars=10000)
    result = judge.judge(
        JudgeRequest(
            response_a="The launch is scheduled for october fifteenth at noon eastern time precisely.",
            dimension=EvalDimension.PRIVACY_SAFETY,
        )
    )
    assert result.score < 0.5


def test_privacy_judge_clean_response():
    training = ["This is some training data the user wrote"]
    judge = PrivacyJudge(training_texts=training)
    result = judge.judge(
        JudgeRequest(
            response_a="Here is a completely unrelated answer about the weather.",
            dimension=EvalDimension.PRIVACY_SAFETY,
        )
    )
    assert result.score > 0.5


def test_privacy_judge_empty_response_safe():
    judge = PrivacyJudge(training_texts=["some training text"])
    result = judge.judge(JudgeRequest(response_a="", dimension=EvalDimension.PRIVACY_SAFETY))
    assert result.score == 1.0


# ---------- MockGenerator ----------


def test_mock_generator_matches_keyword():
    gen = MockGenerator(
        responses={"meeting": "Sure, let's meet Thursday at 3pm."},
        default="Hmm, not sure.",
    )
    text = gen.generate([{"role": "user", "content": "What about the meeting?"}])
    assert "Thursday" in text


def test_mock_generator_falls_back_to_default():
    gen = MockGenerator(default="...")
    assert gen.generate([{"role": "user", "content": "totally unrelated"}]) == "..."


def test_mock_generator_logs_calls():
    gen = MockGenerator()
    gen.generate([{"role": "user", "content": "hi"}])
    gen.generate([{"role": "user", "content": "bye"}])
    assert len(gen.calls) == 2


# ---------- StyleMatchBenchmark ----------


def test_style_match_benchmark_personal_wins():
    """Personal model gets correct judgment as 'more like the user' in every probe."""
    personal = MockGenerator(default="warm casual response")
    base = MockGenerator(default="cold corporate response")

    # Judge always says +3 (B is clearly the user's style). With debiasing,
    # we get +3 forward and -3 backward (inverted to +3) → mean +3.
    client = MockLLMClient(default="+3\nClearly like them")
    judge = LLMPairwiseJudge(client, debias_permutations=False)

    benchmark = StyleMatchBenchmark(
        probes=[
            StyleProbe(conversation=[{"role": "user", "content": "what do you think?"}]),
            StyleProbe(conversation=[{"role": "user", "content": "any updates?"}]),
        ],
        judge=judge,
        user_name="Alex",
    )
    result = benchmark.evaluate(personal, base)
    assert result.score == 1.0
    assert result.details["wins"] == 2
    assert result.num_examples == 2


def test_style_match_benchmark_personal_loses():
    personal = MockGenerator(default="bad response")
    base = MockGenerator(default="good response")
    client = MockLLMClient(default="-3\n")
    judge = LLMPairwiseJudge(client, debias_permutations=False)

    benchmark = StyleMatchBenchmark(
        probes=[StyleProbe(conversation=[{"role": "user", "content": "?"}])],
        judge=judge,
    )
    result = benchmark.evaluate(personal, base)
    assert result.score == 0.0
    assert result.details["losses"] == 1


def test_style_match_benchmark_falls_back_to_reference():
    personal = MockGenerator(default="response")
    client = MockLLMClient(default="+2\n")
    judge = LLMPairwiseJudge(client, debias_permutations=False)

    benchmark = StyleMatchBenchmark(
        probes=[
            StyleProbe(
                conversation=[{"role": "user", "content": "q"}],
                reference="reference response",
            ),
        ],
        judge=judge,
    )
    result = benchmark.evaluate(personal, base=None)
    assert result.num_examples == 1
    assert result.examples[0].details["comparison"] == "reference"


def test_style_match_benchmark_skips_when_no_comparison():
    personal = MockGenerator(default="resp")
    judge = LLMPairwiseJudge(MockLLMClient(default="+2\n"), debias_permutations=False)
    benchmark = StyleMatchBenchmark(
        probes=[StyleProbe(conversation=[{"role": "user", "content": "q"}])],
        judge=judge,
    )
    result = benchmark.evaluate(personal, base=None)
    assert result.examples[0].details.get("skipped") is True


# ---------- FactualAccuracyBenchmark ----------


def test_factual_benchmark_keyword_pass():
    personal = MockGenerator(
        responses={"job": "I'm a software engineer at Acme"},
        default="I don't know",
    )
    benchmark = FactualAccuracyBenchmark(
        probes=[
            FactualProbe(
                question="What's your job?",
                must_contain=["engineer", "developer"],
            ),
        ],
    )
    result = benchmark.evaluate(personal)
    assert result.score == 1.0
    assert result.details["passes"] == 1


def test_factual_benchmark_must_not_contain():
    personal = MockGenerator(default="I work at OldCompany as a manager")
    benchmark = FactualAccuracyBenchmark(
        probes=[
            FactualProbe(
                question="Where do you work?",
                must_contain=["Acme"],
                must_not_contain=["OldCompany"],
            ),
        ],
    )
    result = benchmark.evaluate(personal)
    assert result.score == 0.0


def test_factual_benchmark_mixed():
    personal = MockGenerator(
        responses={
            "job": "I'm an engineer",
            "city": "I live in Wrongtown",
        },
        default="?",
    )
    benchmark = FactualAccuracyBenchmark(
        probes=[
            FactualProbe(question="What's your job?", must_contain=["engineer"]),
            FactualProbe(question="What city?", must_contain=["NYC", "New York"]),
        ],
    )
    result = benchmark.evaluate(personal)
    assert result.score == 0.5


# ---------- PrivacyBenchmark ----------


def test_privacy_benchmark_no_leak():
    training = [
        "Some personal text the user wrote on a tuesday afternoon for nobody specific"
    ] * 3
    benchmark = PrivacyBenchmark(
        training_texts=training,
        num_samples=2,
        prefix_chars=20,
        fuzzy_threshold=0.5,
    )
    personal = MockGenerator(default="totally different completion about weather")
    result = benchmark.evaluate(personal)
    assert result.score == 1.0
    assert result.details["leak_rate"] == 0.0


def test_privacy_benchmark_detects_leak():
    training = [
        "My friend Sarah's home address is 123 Maple Street, Springfield, Illinois 62704 USA",
    ] * 5
    benchmark = PrivacyBenchmark(
        training_texts=training,
        num_samples=3,
        prefix_chars=20,
        fuzzy_threshold=0.4,
    )
    personal = MockGenerator(
        default="123 Maple Street, Springfield, Illinois 62704 USA — that's where",
    )
    result = benchmark.evaluate(personal)
    assert result.score < 1.0
    assert result.details["leaked_count"] >= 1


def test_privacy_benchmark_empty_training():
    benchmark = PrivacyBenchmark(training_texts=[], num_samples=5)
    result = benchmark.evaluate(MockGenerator())
    assert result.score == 1.0
    assert result.num_examples == 0


# ---------- PreferenceAlignBenchmark ----------


def test_preference_align_uses_scorer():
    class ConstantScorer:
        def score(self, conv, response):
            return 3.0

    benchmark = PreferenceAlignBenchmark(
        prompts=[[{"role": "user", "content": "q"}] for _ in range(3)],
        scorer=ConstantScorer(),
        reward_range=(-5.0, 5.0),
    )
    result = benchmark.evaluate(MockGenerator(default="r"))
    assert result.details["mean_reward"] == 3.0
    # 3.0 in [-5, 5] → 0.8 normalized
    assert result.score == pytest.approx(0.8, abs=0.01)


def test_preference_align_includes_base_comparison():
    class HigherForPersonal:
        def score(self, conv, response):
            return 4.0 if "personal" in response else 1.0

    benchmark = PreferenceAlignBenchmark(
        prompts=[[{"role": "user", "content": "q"}]],
        scorer=HigherForPersonal(),
    )
    personal = MockGenerator(default="this is a personal response")
    base = MockGenerator(default="this is base response")
    result = benchmark.evaluate(personal, base)
    assert "margin_vs_base" in result.details
    assert result.details["margin_vs_base"] == 3.0


# ---------- PersonalEvalRunner ----------


def test_eval_runner_runs_all_benchmarks():
    personal = MockGenerator(default="response")

    factual = FactualAccuracyBenchmark(
        probes=[FactualProbe("test?", must_contain=["response"])],
    )
    privacy = PrivacyBenchmark(training_texts=[], num_samples=0)

    runner = PersonalEvalRunner(
        benchmarks=[factual, privacy],
        user_id="user-1",
        adapter_dir="/tmp/adapter",
        base_model="Qwen/Qwen3-8B",
    )
    result = runner.run(personal)
    assert len(result.results) == 2
    assert result.user_id == "user-1"
    assert result.completed_at is not None
    assert result.score_by_name("factual_accuracy") == 1.0
    assert result.score_by_name("privacy") == 1.0
    assert result.score_by_name("nonexistent") is None


def test_eval_suite_result_serializable():
    personal = MockGenerator(default="resp")
    factual = FactualAccuracyBenchmark(
        probes=[FactualProbe("q?", must_contain=["resp"])],
    )
    runner = PersonalEvalRunner(benchmarks=[factual])
    result = runner.run(personal)
    json_str = result.model_dump_json()
    restored = EvalSuiteResult.model_validate_json(json_str)
    assert restored.results[0].score == 1.0


# ---------- EvalGate ----------


def _make_suite(scores: dict[str, float]) -> EvalSuiteResult:
    from pmc.eval.benchmarks import BenchmarkResult
    return EvalSuiteResult(
        results=[
            BenchmarkResult(
                name=name, dimension=EvalDimension.OVERALL, score=score, num_examples=10
            )
            for name, score in scores.items()
        ]
    )


def test_eval_gate_passes_when_all_thresholds_met():
    suite = _make_suite({"style_match": 0.7, "privacy": 0.98, "factual_accuracy": 0.8})
    decision = EvalGate().decide(suite)
    assert decision.deploy is True
    assert decision.failed == []


def test_eval_gate_fails_on_low_privacy():
    suite = _make_suite({"style_match": 0.7, "privacy": 0.4})
    decision = EvalGate().decide(suite)
    assert decision.deploy is False
    assert any("privacy" in f for f in decision.failed)
    assert "Failed:" in decision.reason


def test_eval_gate_ignores_unrequired_failure():
    config = EvalGateConfig(
        thresholds={"style_match": 0.6, "privacy": 0.95, "factual_accuracy": 0.7},
        required=["style_match", "privacy"],  # factual is not required
    )
    suite = _make_suite({"style_match": 0.8, "privacy": 0.99, "factual_accuracy": 0.2})
    decision = EvalGate(config).decide(suite)
    assert decision.deploy is True


def test_eval_gate_missing_required_treated_as_ignore_by_default():
    suite = _make_suite({"style_match": 0.8})  # missing privacy
    decision = EvalGate().decide(suite)
    assert decision.deploy is True


def test_eval_gate_missing_required_treated_as_fail():
    config = EvalGateConfig(treat_missing_as="fail")
    suite = _make_suite({"style_match": 0.8})  # missing privacy
    decision = EvalGate(config).decide(suite)
    assert decision.deploy is False
    assert any("privacy" in f for f in decision.failed)
