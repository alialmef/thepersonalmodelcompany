"""Training-time configs and result types.

The core hyperparameter config (`TrainingConfig`, `AdapterConfig`) lives in
`pmc/schema/training.py` because it's shared across pipeline stages and
serialized into the artifact bundle. This module adds:

- `DPOConfig` — DPO-specific hyperparams (beta, loss type)
- `RewardModelConfig` — pairwise reward model training hyperparams
- `TrainingPlan` — dry-run estimate (steps, time, params) returned before training
- `SFTRunResult` / `DPORunResult` / `RewardRunResult` — what a completed run returns
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pmc.schema.training import AdapterConfig, TrainingConfig


class DPOConfig(BaseModel):
    """DPO hyperparams. Pair with a TrainingConfig for shared model/adapter/optim settings."""

    beta: float = 0.1
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    label_smoothing: float = 0.0
    max_prompt_length: int = 1024
    max_length: int = 2048
    reference_free: bool = False


class RewardModelConfig(BaseModel):
    """Pairwise reward model (TRL RewardTrainer) hyperparams."""

    base_model: str = "Qwen/Qwen3-0.6B"
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_length: int = 1024
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42


class TrainingPlan(BaseModel):
    """Dry-run estimate produced before kicking off a job.

    Returned by `plan_sft()` / `plan_dpo()` so the orchestrator (or the user)
    can sanity-check size/cost before committing GPU time.
    """

    job_type: Literal["sft", "dpo", "reward"]
    base_model: str
    num_train_examples: int
    num_eval_examples: int = 0
    effective_batch_size: int
    estimated_steps: int
    estimated_trainable_params: int
    estimated_minutes: float
    estimated_adapter_mb: float
    warnings: list[str] = Field(default_factory=list)


class SFTRunResult(BaseModel):
    """What a completed SFT run produces."""

    job_type: Literal["sft"] = "sft"
    user_id: str
    base_model: str
    adapter_dir: Path
    num_train_examples: int
    num_eval_examples: int = 0
    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    elapsed_seconds: float = 0.0
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    config: TrainingConfig

    model_config = {"arbitrary_types_allowed": True}


class DPORunResult(BaseModel):
    job_type: Literal["dpo"] = "dpo"
    user_id: str
    base_model: str
    adapter_dir: Path
    base_adapter_dir: Path | None = None
    num_pairs: int
    final_loss: float | None = None
    final_reward_margin: float | None = None
    elapsed_seconds: float = 0.0
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    sft_config: TrainingConfig
    dpo_config: DPOConfig

    model_config = {"arbitrary_types_allowed": True}


class RewardRunResult(BaseModel):
    job_type: Literal["reward"] = "reward"
    user_id: str
    base_model: str
    model_dir: Path
    num_pairs: int
    final_loss: float | None = None
    final_accuracy: float | None = None
    elapsed_seconds: float = 0.0
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    config: RewardModelConfig

    model_config = {"arbitrary_types_allowed": True}


# --- Plan estimation helpers (pure, no torch) ---


def estimate_trainable_params(
    adapter: AdapterConfig,
    hidden_dim: int = 4096,
    num_layers: int = 32,
) -> int:
    """Rough LoRA parameter count: rank * (in_dim + out_dim) per target module, per layer.

    For attention projections at hidden_dim=4096, each LoRA matrix pair adds
    `rank * 2 * 4096` trainable params per layer.
    """
    per_module_per_layer = adapter.rank * 2 * hidden_dim
    total = per_module_per_layer * len(adapter.target_modules) * num_layers
    return total


def estimate_adapter_size_mb(trainable_params: int, bytes_per_param: int = 2) -> float:
    """LoRA adapters typically save as bf16 — 2 bytes per param."""
    return round(trainable_params * bytes_per_param / (1024 * 1024), 2)


def estimate_steps(
    num_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
) -> tuple[int, int]:
    """Returns (effective_batch_size, total_steps)."""
    effective = max(1, batch_size * gradient_accumulation_steps)
    per_epoch = math.ceil(num_examples / effective) if num_examples else 0
    return effective, per_epoch * num_epochs


def estimate_training_minutes(
    total_steps: int,
    seconds_per_step: float = 1.5,
) -> float:
    """Rough wall-clock estimate. ~1.5s/step is typical for 8B QLoRA on A100."""
    return round(total_steps * seconds_per_step / 60.0, 2)


def warnings_for_dataset(num_examples: int) -> list[str]:
    out: list[str] = []
    if num_examples < 50:
        out.append(
            f"Only {num_examples} examples — model is likely to overfit. "
            "Aim for 1,000+ examples for meaningful style learning."
        )
    elif num_examples < 500:
        out.append(
            f"{num_examples} examples is on the low end. Style signal may be weak."
        )
    if num_examples > 100_000:
        out.append(
            f"{num_examples} examples is large for a personal model — "
            "consider sampling for faster iteration."
        )
    return out
