"""LoRA/QLoRA Supervised Fine-Tuning runner.

The actual training imports (torch, transformers, peft, trl, bitsandbytes) are
deferred until `run_sft()` is called. `plan_sft()` is pure and can be called
without any ML deps installed.

Usage:
    plan = plan_sft(config, train_completions, eval_completions)
    # inspect plan.estimated_minutes, plan.warnings...
    result = run_sft(config, train_completions, eval_completions, output_dir)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pmc.schema.conversation import Completion
from pmc.schema.training import TrainingConfig
from pmc.train.config import (
    SFTRunResult,
    TrainingPlan,
    estimate_adapter_size_mb,
    estimate_steps,
    estimate_trainable_params,
    estimate_training_minutes,
    warnings_for_dataset,
)
from pmc.train.formatter import completion_to_messages


def plan_sft(
    config: TrainingConfig,
    train_completions: Sequence[Completion],
    eval_completions: Sequence[Completion] | None = None,
) -> TrainingPlan:
    """Dry-run estimate. Counts usable examples and predicts steps + time + size."""
    usable_train = sum(
        1 for c in train_completions if completion_to_messages(c) is not None
    )
    usable_eval = (
        sum(1 for c in eval_completions if completion_to_messages(c) is not None)
        if eval_completions
        else 0
    )
    effective_bs, total_steps = estimate_steps(
        num_examples=usable_train,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_epochs=config.num_epochs,
    )
    params = estimate_trainable_params(config.adapter)
    return TrainingPlan(
        job_type="sft",
        base_model=config.base_model,
        num_train_examples=usable_train,
        num_eval_examples=usable_eval,
        effective_batch_size=effective_bs,
        estimated_steps=total_steps,
        estimated_trainable_params=params,
        estimated_minutes=estimate_training_minutes(total_steps),
        estimated_adapter_mb=estimate_adapter_size_mb(params),
        warnings=warnings_for_dataset(usable_train),
    )


def run_sft(
    config: TrainingConfig,
    train_completions: Sequence[Completion],
    output_dir: Path | str,
    eval_completions: Sequence[Completion] | None = None,
    *,
    report_to: str | None = None,
) -> SFTRunResult:
    """Run LoRA/QLoRA SFT on a single GPU. Heavy imports happen here."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_hf()
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = _import_peft()
    SFTTrainer, SFTConfigT = _import_trl_sft()

    from pmc.train.dataset import build_sft_dataset, split_train_eval

    started = datetime.now()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(
        config, AutoModelForCausalLM, BitsAndBytesConfig, prepare_model_for_kbit_training, torch
    )
    lora_cfg = LoraConfig(
        r=config.adapter.rank,
        lora_alpha=config.adapter.alpha,
        lora_dropout=config.adapter.dropout,
        target_modules=config.adapter.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = build_sft_dataset(train_completions)
    eval_ds = build_sft_dataset(eval_completions) if eval_completions else None
    if eval_ds is None and len(train_ds) >= 20:
        split = split_train_eval(train_ds, eval_fraction=0.1, seed=config.seed)
        train_ds, eval_ds = split["train"], split["test"]

    sft_args = SFTConfigT(
        output_dir=str(output_path),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_seq_length=config.max_seq_length,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        logging_steps=10,
        bf16=True,
        seed=config.seed,
        report_to=report_to or "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_path))

    train_loss, eval_loss = _final_losses(trainer)

    return SFTRunResult(
        user_id=config.user_id,
        base_model=config.base_model,
        adapter_dir=output_path,
        num_train_examples=len(train_ds),
        num_eval_examples=len(eval_ds) if eval_ds is not None else 0,
        final_train_loss=train_loss,
        final_eval_loss=eval_loss,
        elapsed_seconds=round(time.time() - t0, 2),
        started_at=started,
        completed_at=datetime.now(),
        config=config,
    )


# --- Lazy imports of heavy ML deps ---


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required. Install with `pip install pmc[train]`.") from e
    return torch


def _import_hf() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required. Install with `pip install pmc[train]`."
        ) from e
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _import_peft() -> tuple[Any, Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError("peft is required. Install with `pip install pmc[train]`.") from e
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _import_trl_sft() -> tuple[Any, Any]:
    try:
        from trl import SFTConfig as SFTConfigT, SFTTrainer
    except ImportError as e:
        raise ImportError("trl is required. Install with `pip install pmc[train]`.") from e
    return SFTTrainer, SFTConfigT


def _load_model(
    config: TrainingConfig,
    AutoModelForCausalLM: Any,
    BitsAndBytesConfig: Any,
    prepare_model_for_kbit_training: Any,
    torch: Any,
) -> Any:
    if config.adapter.use_qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=(config.adapter.bits == 4),
            load_in_8bit=(config.adapter.bits == 8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb,
            device_map="auto",
        )
        return prepare_model_for_kbit_training(model)
    return AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def _final_losses(trainer: Any) -> tuple[float | None, float | None]:
    train_loss: float | None = None
    eval_loss: float | None = None
    for entry in reversed(trainer.state.log_history):
        if train_loss is None and "loss" in entry:
            train_loss = float(entry["loss"])
        if eval_loss is None and "eval_loss" in entry:
            eval_loss = float(entry["eval_loss"])
        if train_loss is not None and eval_loss is not None:
            break
    return train_loss, eval_loss
