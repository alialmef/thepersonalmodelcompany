"""Training pipeline: LoRA/QLoRA SFT, DPO, reward model, and the artifact bundle.

Pure-Python parts (formatter, config, bundle, plan_*) are importable without
torch. The actual runners (`run_sft`, `run_dpo`, `run_reward_model`) and
checkpoint helpers lazy-import torch/transformers/peft/trl/bitsandbytes — install
those with `pip install pmc[train]`.
"""

from pmc.train.bundle import ArtifactBundle, AuditEvent, BundleMetadata
from pmc.train.config import (
    DPOConfig,
    DPORunResult,
    RewardModelConfig,
    RewardRunResult,
    SFTRunResult,
    TrainingPlan,
    estimate_adapter_size_mb,
    estimate_steps,
    estimate_trainable_params,
    estimate_training_minutes,
)
from pmc.train.formatter import (
    build_inference_system_prompt,
    completion_to_dpo_pair,
    completion_to_messages,
    completions_to_messages,
)

__all__ = [
    "ArtifactBundle",
    "AuditEvent",
    "BundleMetadata",
    "DPOConfig",
    "DPORunResult",
    "RewardModelConfig",
    "RewardRunResult",
    "SFTRunResult",
    "TrainingPlan",
    "build_inference_system_prompt",
    "completion_to_dpo_pair",
    "completion_to_messages",
    "completions_to_messages",
    "estimate_adapter_size_mb",
    "estimate_steps",
    "estimate_trainable_params",
    "estimate_training_minutes",
]
