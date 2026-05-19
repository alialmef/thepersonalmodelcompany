"""Pairwise reward model — small "sounds like me" scorer.

The yolo Likert RM pattern trains a model that outputs scores per dimension.
For PMC V0, we use TRL's `RewardTrainer` to train a much simpler pairwise
preference model: given two responses to the same prompt, predict which one
the user prefers (i.e. which sounds more like them).

Use cases:
- Filter/rank candidate outputs at inference time
- Provide a reward signal for DPO without needing fresh user labels
- Surface confidence on "is this output actually like me?"

Heavy ML imports are deferred until `run_reward_model()` is called.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pmc.schema.conversation import Completion
from pmc.train.config import (
    RewardModelConfig,
    RewardRunResult,
    TrainingPlan,
    estimate_adapter_size_mb,
    estimate_steps,
    estimate_trainable_params,
    estimate_training_minutes,
    warnings_for_dataset,
)
from pmc.train.formatter import completion_to_dpo_pair


def plan_reward_model(
    config: RewardModelConfig,
    pairs: Sequence[Completion],
) -> TrainingPlan:
    usable = sum(1 for c in pairs if completion_to_dpo_pair(c) is not None)
    effective_bs, total_steps = estimate_steps(
        num_examples=usable,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_epochs=config.num_epochs,
    )
    params = estimate_trainable_params(config.adapter, hidden_dim=2048, num_layers=24)
    warnings = warnings_for_dataset(usable)
    if usable < 100:
        warnings.append(
            f"Reward model needs 100+ labeled pairs for meaningful signal "
            f"(currently {usable})."
        )
    return TrainingPlan(
        job_type="reward",
        base_model=config.base_model,
        num_train_examples=usable,
        effective_batch_size=effective_bs,
        estimated_steps=total_steps,
        estimated_trainable_params=params,
        estimated_minutes=estimate_training_minutes(total_steps, seconds_per_step=1.0),
        estimated_adapter_mb=estimate_adapter_size_mb(params),
        warnings=warnings,
    )


def run_reward_model(
    config: RewardModelConfig,
    pairs: Sequence[Completion],
    output_dir: Path | str,
    *,
    report_to: str | None = None,
) -> RewardRunResult:
    """Train a pairwise reward model with TRL RewardTrainer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch = _import_torch()
    AutoModelForSequenceClassification, AutoTokenizer = _import_hf_seq_cls()
    LoraConfig, get_peft_model = _import_peft()
    RewardTrainer, RewardConfigT = _import_trl_reward()

    from pmc.train.dataset import build_reward_dataset, split_train_eval

    started = datetime.now()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora = LoraConfig(
        r=config.adapter.rank,
        lora_alpha=config.adapter.alpha,
        lora_dropout=config.adapter.dropout,
        target_modules=config.adapter.target_modules,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora)

    train_ds = build_reward_dataset(pairs)
    eval_ds = None
    if len(train_ds) >= 20:
        split = split_train_eval(train_ds, eval_fraction=0.1, seed=config.seed)
        train_ds, eval_ds = split["train"], split["test"]

    args = RewardConfigT(
        output_dir=str(output_path),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_length=config.max_length,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        logging_steps=10,
        bf16=True,
        seed=config.seed,
        report_to=report_to or "none",
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_path))

    loss, acc = _final_reward_metrics(trainer)
    return RewardRunResult(
        user_id="",
        base_model=config.base_model,
        model_dir=output_path,
        num_pairs=len(train_ds),
        final_loss=loss,
        final_accuracy=acc,
        elapsed_seconds=round(time.time() - t0, 2),
        started_at=started,
        completed_at=datetime.now(),
        config=config,
    )


# --- Lazy imports ---


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required. Install with `pip install pmc[train]`.") from e
    return torch


def _import_hf_seq_cls() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required. Install with `pip install pmc[train]`."
        ) from e
    return AutoModelForSequenceClassification, AutoTokenizer


def _import_peft() -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("peft is required. Install with `pip install pmc[train]`.") from e
    return LoraConfig, get_peft_model


def _import_trl_reward() -> tuple[Any, Any]:
    try:
        from trl import RewardConfig as RewardConfigT, RewardTrainer
    except ImportError as e:
        raise ImportError("trl is required. Install with `pip install pmc[train]`.") from e
    return RewardTrainer, RewardConfigT


def _final_reward_metrics(trainer: Any) -> tuple[float | None, float | None]:
    loss: float | None = None
    acc: float | None = None
    for entry in reversed(trainer.state.log_history):
        if loss is None and "loss" in entry:
            loss = float(entry["loss"])
        if acc is None:
            for key in ("eval_accuracy", "accuracy"):
                if key in entry:
                    acc = float(entry[key])
                    break
        if loss is not None and acc is not None:
            break
    return loss, acc
