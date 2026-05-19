"""Tests for the base model registry — Try/Personal/Frontier tier shape."""

from __future__ import annotations

import pytest

from pmc.schema import (
    AdapterConfig,
    AgenticGrade,
    BaseModelSpec,
    MODEL_REGISTRY,
    ModelTier,
    TrainingConfig,
    default_spec,
    get_spec,
    list_specs,
    register,
    spec_for_tier,
    try_get_spec,
)


# ---------- Registry shape ----------


def test_registry_has_three_product_tiers():
    """Three entries, one per product tier."""
    assert len(MODEL_REGISTRY) == 3


def test_default_spec_is_personal_tier():
    """Default = Personal tier (Qwen 3.6-27B Dense) — the main product line."""
    spec = default_spec()
    assert spec.tier == ModelTier.PERSONAL
    assert spec.hf_id == "Qwen/Qwen3.6-27B"
    assert spec.subscription_usd_per_month == 79.00


def test_every_tier_has_exactly_one_spec():
    """Each product tier maps to exactly one base model in V1."""
    for tier in (ModelTier.TRY, ModelTier.PERSONAL, ModelTier.FRONTIER):
        matches = list_specs(tier=tier)
        assert len(matches) == 1, f"Tier {tier} has {len(matches)} specs"


def test_every_spec_has_required_fields():
    for spec in MODEL_REGISTRY.values():
        assert spec.hf_id
        assert spec.display_name
        assert spec.family
        assert spec.total_params_b > 0
        assert spec.context_length > 0
        assert spec.default_adapter_rank > 0
        assert spec.default_adapter_modules
        assert spec.estimated_training_usd > 0
        assert spec.estimated_per_million_tokens_usd > 0
        assert spec.subscription_usd_per_month > 0
        assert spec.license


def test_pricing_ladder_makes_sense():
    """Subscription price ascends with tier — Try < Personal < Frontier."""
    try_spec = spec_for_tier(ModelTier.TRY)
    personal_spec = spec_for_tier(ModelTier.PERSONAL)
    frontier_spec = spec_for_tier(ModelTier.FRONTIER)
    assert try_spec.subscription_usd_per_month < personal_spec.subscription_usd_per_month
    assert personal_spec.subscription_usd_per_month < frontier_spec.subscription_usd_per_month
    # Committed product prices
    assert try_spec.subscription_usd_per_month == 19.00
    assert personal_spec.subscription_usd_per_month == 79.00
    assert frontier_spec.subscription_usd_per_month == 299.00


def test_training_cost_below_first_month_revenue():
    """Every tier's training cost must be recouped in <1 month of subscription
    so we don't lose money on a same-month canceller."""
    for spec in MODEL_REGISTRY.values():
        assert spec.estimated_training_usd < spec.subscription_usd_per_month, (
            f"{spec.display_name}: training ${spec.estimated_training_usd} > "
            f"sub ${spec.subscription_usd_per_month}"
        )


def test_agentic_grade_ascends_with_tier():
    """Higher tier → more agentic. Try=POOR, Personal=GOOD, Frontier=FRONTIER."""
    grade_order = {
        AgenticGrade.POOR: 0,
        AgenticGrade.BASIC: 1,
        AgenticGrade.GOOD: 2,
        AgenticGrade.FRONTIER: 3,
    }
    try_spec = spec_for_tier(ModelTier.TRY)
    personal_spec = spec_for_tier(ModelTier.PERSONAL)
    frontier_spec = spec_for_tier(ModelTier.FRONTIER)
    assert grade_order[try_spec.agentic_grade] < grade_order[personal_spec.agentic_grade]
    assert grade_order[personal_spec.agentic_grade] < grade_order[frontier_spec.agentic_grade]


# ---------- Lookup ----------


def test_get_spec_by_hf_id():
    assert get_spec("meta-llama/Llama-3.1-8B-Instruct").tier == ModelTier.TRY
    assert get_spec("Qwen/Qwen3.6-27B").tier == ModelTier.PERSONAL
    assert get_spec("moonshotai/Kimi-K2.6").tier == ModelTier.FRONTIER


def test_get_spec_by_tier_alias():
    """The tier names themselves are aliases — `try`, `personal`, `frontier`."""
    assert get_spec("try").tier == ModelTier.TRY
    assert get_spec("personal").tier == ModelTier.PERSONAL
    assert get_spec("frontier").tier == ModelTier.FRONTIER


def test_get_spec_by_friendly_alias():
    assert get_spec("llama-8b").family == "Llama"
    assert get_spec("qwen-27b").family == "Qwen"
    assert get_spec("kimi").family.startswith("Kimi")
    assert get_spec("kimi-k2").tier == ModelTier.FRONTIER


def test_get_spec_alias_case_insensitive():
    assert get_spec("KIMI-K2.6").tier == ModelTier.FRONTIER


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("non-existent-model")


def test_try_get_spec_returns_none_on_miss():
    assert try_get_spec("missing") is None
    assert try_get_spec("personal") is not None


def test_spec_for_tier():
    assert spec_for_tier(ModelTier.TRY).hf_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec_for_tier(ModelTier.PERSONAL).hf_id == "Qwen/Qwen3.6-27B"
    assert spec_for_tier(ModelTier.FRONTIER).hf_id == "moonshotai/Kimi-K2.6"


def test_spec_for_tier_unknown_raises():
    with pytest.raises(KeyError):
        spec_for_tier(ModelTier.ENTERPRISE)  # not in V1 registry


# ---------- Filtering ----------


def test_list_specs_no_filter_returns_all():
    assert len(list_specs()) == 3


def test_list_specs_filter_by_tier():
    assert len(list_specs(tier=ModelTier.TRY)) == 1
    assert len(list_specs(tier=ModelTier.PERSONAL)) == 1
    assert len(list_specs(tier=ModelTier.FRONTIER)) == 1
    assert len(list_specs(tier=ModelTier.ENTERPRISE)) == 0


def test_list_specs_filter_by_agentic_grade():
    frontier_grade = list_specs(agentic_grade=AgenticGrade.FRONTIER)
    assert len(frontier_grade) == 1
    assert frontier_grade[0].tier == ModelTier.FRONTIER


def test_list_specs_filter_by_multimodal():
    """Only the Personal tier (Qwen 3.6-27B) is multimodal in V1."""
    multimodal = list_specs(multimodal=True)
    assert len(multimodal) == 1
    assert multimodal[0].tier == ModelTier.PERSONAL


def test_list_specs_filter_combined():
    specs = list_specs(tier=ModelTier.PERSONAL, multimodal=True)
    assert len(specs) == 1


# ---------- Adapter defaults ----------


def test_adapter_config_from_spec_uses_per_model_rank():
    small = spec_for_tier(ModelTier.TRY)
    big = spec_for_tier(ModelTier.FRONTIER)
    small_adapter = AdapterConfig.from_spec(small)
    big_adapter = AdapterConfig.from_spec(big)
    assert small_adapter.rank == small.default_adapter_rank
    assert big_adapter.rank == big.default_adapter_rank
    assert big_adapter.rank >= small_adapter.rank


def test_adapter_config_from_spec_qlora_toggle():
    spec = default_spec()
    qlora = AdapterConfig.from_spec(spec, use_qlora=True)
    full = AdapterConfig.from_spec(spec, use_qlora=False)
    assert qlora.use_qlora is True and qlora.bits == 4
    assert full.use_qlora is False and full.bits == 16


def test_adapter_alpha_is_2x_rank():
    spec = default_spec()
    adapter = AdapterConfig.from_spec(spec)
    assert adapter.alpha == adapter.rank * 2


# ---------- TrainingConfig integration ----------


def test_training_config_from_spec_basic():
    spec = spec_for_tier(ModelTier.PERSONAL)
    cfg = TrainingConfig.from_spec(spec, user_id="alex")
    assert cfg.user_id == "alex"
    assert cfg.base_model == spec.hf_id
    assert cfg.adapter.rank == spec.default_adapter_rank


def test_training_config_from_spec_with_overrides():
    spec = spec_for_tier(ModelTier.FRONTIER)
    cfg = TrainingConfig.from_spec(
        spec, user_id="alex", num_epochs=1, learning_rate=1e-4, seed=99,
    )
    assert cfg.num_epochs == 1
    assert cfg.learning_rate == 1e-4
    assert cfg.seed == 99
    assert cfg.base_model == spec.hf_id


def test_training_config_default_unchanged():
    """Backwards compat: TrainingConfig() default still produces a valid config."""
    cfg = TrainingConfig(user_id="u")
    # Default stays on the cheapest model for dev/test sanity — production
    # callers should always use TrainingConfig.from_spec(spec, ...).
    assert cfg.base_model == "meta-llama/Llama-3.1-8B-Instruct"


# ---------- Dynamic registration ----------


def test_register_new_spec():
    spec = BaseModelSpec(
        hf_id="example/test-model-1",
        display_name="Test 1",
        family="Test",
        tier=ModelTier.PERSONAL,
        total_params_b=1.0,
        context_length=4096,
        agentic_grade=AgenticGrade.POOR,
        together_supported=False,
        serving_compute="A100",
        default_adapter_rank=8,
        estimated_training_usd=0.5,
        estimated_per_million_tokens_usd=0.05,
        subscription_usd_per_month=29.00,
        license="MIT",
    )
    register(spec, aliases=["test-1"])
    assert get_spec("example/test-model-1").display_name == "Test 1"
    assert get_spec("test-1").display_name == "Test 1"

    # Cleanup so the registry stays at three for other tests
    MODEL_REGISTRY.pop("example/test-model-1", None)
    from pmc.schema.base_models import ALIASES
    ALIASES.pop("test-1", None)


def test_serialization_roundtrip():
    spec = default_spec()
    json_str = spec.model_dump_json()
    restored = BaseModelSpec.model_validate_json(json_str)
    assert restored.hf_id == spec.hf_id
    assert restored.tier == spec.tier
    assert restored.default_adapter_rank == spec.default_adapter_rank
    assert restored.subscription_usd_per_month == spec.subscription_usd_per_month
