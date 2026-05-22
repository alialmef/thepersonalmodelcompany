"""Training-specific data types: configs, SFT examples, preference pairs."""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from pmc.schema.annotations import SourceAnnotation
from pmc.schema.conversation import Message

if TYPE_CHECKING:
    from pmc.schema.base_models import BaseModelSpec


class BaseModel_(StrEnum):
    """Legacy: kept for backwards compatibility with existing code paths.

    New code should use the `pmc.schema.base_models` registry — call
    `get_spec(...)` or `default_spec()` for richer metadata than just the HF id.
    """

    # Default — Llama 3.1 8B Instruct is natively supported by Together AI's
    # serverless multi-LoRA, which is our V0 serving target.
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
    QWEN3_8B = "Qwen/Qwen3-8B"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"


class AdapterConfig(BaseModel):
    """LoRA/QLoRA adapter hyperparameters."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    use_qlora: bool = True
    bits: int = 4

    @classmethod
    def from_spec(cls, spec: BaseModelSpec, *, use_qlora: bool = True) -> AdapterConfig:
        """Build an AdapterConfig using the per-model defaults from the registry."""
        return cls(
            rank=spec.default_adapter_rank,
            alpha=spec.default_adapter_rank * 2,
            target_modules=list(spec.default_adapter_modules),
            use_qlora=use_qlora,
            bits=4 if use_qlora else 16,
        )


class TrainingConfig(BaseModel):
    """Full training job configuration."""

    user_id: str
    base_model: str = BaseModel_.LLAMA_3_1_8B
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    output_dir: str = ""
    seed: int = 42

    # Cost-control levers used by remote (Together) trainers. Local
    # trainers ignore these. Defaults are the "legendary recipe":
    #   - cap each example at 2k tokens so voice signal isn't drowned in context
    #   - subsample to 1,500 highest-importance examples
    # See pmc/train/together_trainer.py for the rationale.
    max_tokens_per_example: int = 2000
    max_examples: int = 1500

    # Convenience accessors for the Together trainer; mirror adapter
    # config so the trainer doesn't have to dig into nested objects.
    @property
    def lora_r(self) -> int:
        return self.adapter.rank

    @property
    def lora_alpha(self) -> int:
        return self.adapter.alpha

    @classmethod
    def from_spec(
        cls,
        spec: BaseModelSpec,
        *,
        user_id: str,
        use_qlora: bool = True,
        **overrides: Any,
    ) -> TrainingConfig:
        """Build a TrainingConfig from a BaseModelSpec.

        Uses the spec's HF id as `base_model` and the spec's adapter defaults
        (rank, target modules) for the adapter config. Caller can pass any
        TrainingConfig field as a kwarg to override.
        """
        defaults: dict[str, Any] = {
            "user_id": user_id,
            "base_model": spec.hf_id,
            "adapter": AdapterConfig.from_spec(spec, use_qlora=use_qlora),
        }
        defaults.update(overrides)
        return cls(**defaults)


class SFTExample(BaseModel):
    """A single supervised fine-tuning example in chat format."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    messages: list[Message]
    source: SourceAnnotation | None = None
    train_on_last_n: int = 1


class PreferencePair(BaseModel):
    """A preference pair for DPO training."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    conversation: list[Message]
    chosen: str
    rejected: str
    dimension: str = "style"
    source: SourceAnnotation | None = None
