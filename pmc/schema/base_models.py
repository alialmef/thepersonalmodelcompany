"""Base model registry — what PMC knows about each model it can fine-tune.

The registry maps directly to the **three product tiers**:

  Try        $19/mo   Llama 3.1 8B          See it work on your own writing
  Personal   $79/mo   Qwen 3.6-27B Dense    A real personal agent
  Frontier   $299/mo  Kimi K2.6              The best one. Sub-agent swarms.

Each entry is a `BaseModelSpec` carrying everything we need to make a sensible
default: tier label, params, context length, agentic grade, hosting story,
LoRA defaults, cost estimates, monthly subscription price, license.

Lookup:
    spec = get_spec("Qwen/Qwen3.6-27B")
    spec = get_spec("qwen-3.6-27b")  # alias

Build a TrainingConfig with the right adapter defaults for a model:
    config = TrainingConfig.from_spec(spec, user_id="alex")

Adding a new model is a one-line code change — just append to MODEL_REGISTRY.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class ModelTier(StrEnum):
    """Product-facing tier — maps directly to subscription price + base model."""

    TRY = "try"                # $19/mo — see how it works
    PERSONAL = "personal"      # $79/mo — a real agent
    FRONTIER = "frontier"      # $299/mo — the best
    ENTERPRISE = "enterprise"  # per-seat custom, min $5K/mo


class AgenticGrade(StrEnum):
    """How well the model handles tool use, multi-step planning, code, etc."""

    POOR = "poor"              # 8B-class — basic chat only
    BASIC = "basic"            # Mid-size — single-tool use OK
    GOOD = "good"              # Large open — multi-tool, multi-step
    FRONTIER = "frontier"      # Claude Sonnet 4 / GPT-5 class


class BaseModelSpec(BaseModel):
    """Everything PMC knows about a base model + the product tier it serves."""

    # Identity
    hf_id: str
    display_name: str
    family: str
    tier: ModelTier

    # Architecture
    total_params_b: float
    active_params_b: Optional[float] = None   # For MoE; None for dense
    context_length: int

    # Capability
    agentic_grade: AgenticGrade
    multimodal: bool = False

    # Hosting / serving
    together_supported: bool
    serving_compute: str                       # "A100", "H100", "1xH100", "8xH100", etc.

    # LoRA defaults
    default_adapter_rank: int
    default_adapter_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Economics (USD, refined as real billing data lands)
    estimated_training_usd: float              # Per training run on Modal preemptible
    estimated_per_million_tokens_usd: float    # Inference token cost
    subscription_usd_per_month: float          # What we charge the user for this tier

    # Legal
    license: str
    license_url: Optional[str] = None

    # Free-form
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — three product tiers, three entries.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, BaseModelSpec] = {
    "meta-llama/Llama-3.1-8B-Instruct": BaseModelSpec(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B Instruct",
        family="Llama",
        tier=ModelTier.TRY,
        total_params_b=8.0,
        context_length=131072,
        agentic_grade=AgenticGrade.POOR,
        multimodal=False,
        together_supported=True,
        serving_compute="A100",
        default_adapter_rank=32,
        estimated_training_usd=2.50,
        estimated_per_million_tokens_usd=0.20,
        subscription_usd_per_month=19.00,
        license="Llama 3.1 Community License",
        license_url="https://www.llama.com/llama3_1/license/",
        notes=(
            "TRY tier ($19/mo). Cheapest entry point. Together has confirmed "
            "serverless multi-LoRA support. Weak agent — good for chat + basic tools, "
            "users will upgrade for real agentic work."
        ),
    ),
    "Qwen/Qwen3.6-27B": BaseModelSpec(
        hf_id="Qwen/Qwen3.6-27B",
        display_name="Qwen 3.6 27B Dense",
        family="Qwen",
        tier=ModelTier.PERSONAL,
        total_params_b=27.0,
        context_length=262144,
        agentic_grade=AgenticGrade.GOOD,
        multimodal=True,
        together_supported=True,  # Verify before launch — research says first-class
        serving_compute="1xH100",
        default_adapter_rank=32,
        estimated_training_usd=10.00,
        estimated_per_million_tokens_usd=0.88,
        subscription_usd_per_month=79.00,
        license="Apache 2.0",
        notes=(
            "PERSONAL tier ($79/mo). The default product. 27B dense beating its own "
            "397B MoE predecessor on SWE-bench Verified (77.2). Matches Claude 4.5 Opus "
            "on Terminal-Bench. Single H100 serves hundreds of LoRAs. Apache 2.0 — "
            "no license review. Vision-language unified. The right launch base."
        ),
    ),
    "moonshotai/Kimi-K2.6": BaseModelSpec(
        hf_id="moonshotai/Kimi-K2.6",
        display_name="Kimi K2.6",
        family="Kimi (Moonshot)",
        tier=ModelTier.FRONTIER,
        total_params_b=1000.0,
        active_params_b=32.0,
        context_length=262144,
        agentic_grade=AgenticGrade.FRONTIER,
        multimodal=False,
        together_supported=True,
        serving_compute="serverless (Together FP4)",
        default_adapter_rank=64,
        # Together-side cost of one LoRA fine-tune. Roughly tokens-of-data
        # × $X per million; the exact figure depends on dataset size and
        # epochs. $40 is a comfortable budget cap for V1 datasets (~10k
        # curated conversations, 1-2 epochs).
        estimated_training_usd=40.00,
        # Weighted blended rate. Together (2026-05): input $1.20/M,
        # cached input $0.20/M, output $4.50/M. Real-world traffic skews
        # heavily to cached input (memory recall) so the blended cost
        # per million is closer to ~$1.50.
        estimated_per_million_tokens_usd=1.50,
        subscription_usd_per_month=50.00,
        license="Modified MIT",
        license_url="https://github.com/moonshotai/Kimi-K2",
        notes=(
            "Single PMC product ($50/mo). Best open agentic model in the world: "
            "ties Opus 4.7 on HLE-with-tools, leads GPT-5.4 on SWE-Pro, 200-300 "
            "sequential tool calls, 300 sub-agent swarms. 256K context. "
            "Together hosts the FP4 quantization SERVERLESSLY (pay-per-token, no "
            "dedicated endpoint) and accepts LoRA fine-tunes against it — "
            "verified 2026-05-27 via /v1/fine-tunes probe. Pricing: $1.20 input / "
            "$4.50 output / $0.20 cached input per million tokens, $0/hr base. "
            "Margin holds for ~95% of users with a 10M-output/mo soft cap."
        ),
    ),
}

# Friendly short aliases. Resolved by get_spec() before registry lookup.
ALIASES: dict[str, str] = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "try": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-3.6-27b": "Qwen/Qwen3.6-27B",
    "qwen-27b": "Qwen/Qwen3.6-27B",
    "personal": "Qwen/Qwen3.6-27B",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
    "kimi-k2": "moonshotai/Kimi-K2.6",
    "kimi": "moonshotai/Kimi-K2.6",
    "frontier": "moonshotai/Kimi-K2.6",
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_spec(model_id_or_alias: str) -> BaseModelSpec:
    """Resolve a spec by HF id, friendly alias, or tier name.

    Raises KeyError if no spec is registered.
    """
    resolved = ALIASES.get(model_id_or_alias.lower(), model_id_or_alias)
    if resolved not in MODEL_REGISTRY:
        raise KeyError(
            f"No spec for {model_id_or_alias!r}. "
            f"Known: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[resolved]


def try_get_spec(model_id_or_alias: str) -> BaseModelSpec | None:
    """Same as get_spec but returns None instead of raising."""
    try:
        return get_spec(model_id_or_alias)
    except KeyError:
        return None


def list_specs(
    *,
    tier: ModelTier | None = None,
    together_supported: bool | None = None,
    agentic_grade: AgenticGrade | None = None,
    multimodal: bool | None = None,
) -> list[BaseModelSpec]:
    """List registered specs, optionally filtered."""
    specs = list(MODEL_REGISTRY.values())
    if tier is not None:
        specs = [s for s in specs if s.tier == tier]
    if together_supported is not None:
        specs = [s for s in specs if s.together_supported == together_supported]
    if agentic_grade is not None:
        specs = [s for s in specs if s.agentic_grade == agentic_grade]
    if multimodal is not None:
        specs = [s for s in specs if s.multimodal == multimodal]
    return specs


def default_spec() -> BaseModelSpec:
    """The PMC product default — the Personal tier (Qwen 3.6-27B Dense).

    This is the main product line. The Try tier is an on-ramp; Frontier is the
    upgrade. When code asks "the default model," they mean the Personal tier.
    """
    return get_spec("Qwen/Qwen3.6-27B")


def spec_for_tier(tier: ModelTier) -> BaseModelSpec:
    """Get the single canonical spec for a product tier.

    Each tier maps to exactly one base model in V1. If multiple specs share a
    tier (future), returns the first one.
    """
    matches = list_specs(tier=tier)
    if not matches:
        raise KeyError(f"No spec registered for tier {tier!r}")
    return matches[0]


def register(spec: BaseModelSpec, *, aliases: list[str] | None = None) -> None:
    """Add or replace a spec in the registry."""
    MODEL_REGISTRY[spec.hf_id] = spec
    if aliases:
        for alias in aliases:
            ALIASES[alias.lower()] = spec.hf_id


__all__ = [
    "ALIASES",
    "AgenticGrade",
    "BaseModelSpec",
    "MODEL_REGISTRY",
    "ModelTier",
    "default_spec",
    "get_spec",
    "list_specs",
    "register",
    "spec_for_tier",
    "try_get_spec",
]
