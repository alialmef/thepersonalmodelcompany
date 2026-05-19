"""DPO (Direct Preference Optimization) training on preference pairs.

DPO refines an SFT-trained adapter using "which sounds more like the user?"
preference pairs. The flow is: collect pairs (in `eval/` or `serve/`), then run
DPO starting from the existing SFT adapter to nudge the model toward the chosen
responses.

Like sft.py, heavy ML imports are deferred until `run_dpo()` is called.
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
    DPOConfig,
    DPORunResult,
    TrainingPlan,
    estimate_adapter_size_mb,
    estimate_steps,
    estimate_trainable_params,
    estimate_training_minutes,
    warnings_for_dataset,
)
from pmc.train.formatter import completion_to_dpo_pair


def plan_dpo(
    sft_config: TrainingConfig,
    dpo_config: DPOConfig,
    pairs: Sequence[Completion],
) -> TrainingPlan:
    """Dry-run estimate for a DPO run."""
    usable = sum(1 for c in pairs if completion_to_dpo_pair(c) is not None)
    effective_bs, total_steps = estimate_steps(
        num_examples=usable,
        batch_size=sft_config.batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        num_epochs=sft_config.num_epochs,
    )
    params = estimate_trainable_params(sft_config.adapter)
    warnings = warnings_for_dataset(usable)
    if usable < 100:
        warnings.append(
            f"DPO is noisy with few pairs ({usable}). 200+ pairs recommended."
        )
    # DPO is ~2x slower than SFT because it forwards both chosen and rejected.
    return TrainingPlan(
        job_type="dpo",
        base_model=sft_config.base_model,
        num_train_examples=usable,
        effective_batch_size=effective_bs,
        estimated_steps=total_steps,
        estimated_trainable_params=params,
        estimated_minutes=estimate_training_minutes(total_steps, seconds_per_step=3.0),
        estimated_adapter_mb=estimate_adapter_size_mb(params),
        warnings=warnings,
    )


def run_dpo(
    sft_config: TrainingConfig,
    dpo_config: DPOConfig,
    pairs: Sequence[Completion],
    output_dir: Path | str,
    *,
    base_adapter_dir: Path | str | None = None,
    report_to: str | None = None,
) -> DPORunResult:
    """Run DPO. `base_adapter_dir` should point at the SFT adapter — DPO loads
    it as the policy starting point and uses a frozen copy as the reference.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_hf()
    LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training = _import_peft()
    DPOTrainer, DPOConfigT = _import_trl_dpo()

    from pmc.train.dataset import build_dpo_dataset, split_train_eval

    started = datetime.now()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(sft_config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _load_base(
        sft_config, AutoModelForCausalLM, BitsAndBytesConfig,
        prepare_model_for_kbit_training, torch,
    )

    if base_adapter_dir is not None:
        policy = PeftModel.from_pretrained(base_model, str(base_adapter_dir), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=sft_config.adapter.rank,
            lora_alpha=sft_config.adapter.alpha,
            lora_dropout=sft_config.adapter.dropout,
            target_modules=sft_config.adapter.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base_model, lora_cfg)

    train_ds = build_dpo_dataset(pairs)
    eval_ds = None
    if len(train_ds) >= 20:
        split = split_train_eval(train_ds, eval_fraction=0.1, seed=sft_config.seed)
        train_ds, eval_ds = split["train"], split["test"]

    args = DPOConfigT(
        output_dir=str(output_path),
        num_train_epochs=sft_config.num_epochs,
        per_device_train_batch_size=sft_config.batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        learning_rate=sft_config.learning_rate / 10,  # DPO wants lower LR than SFT
        warmup_ratio=sft_config.warmup_ratio,
        weight_decay=sft_config.weight_decay,
        beta=dpo_config.beta,
        loss_type=dpo_config.loss_type,
        max_length=dpo_config.max_length,
        max_prompt_length=dpo_config.max_prompt_length,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        logging_steps=10,
        bf16=True,
        seed=sft_config.seed,
        report_to=report_to or "none",
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,  # PEFT model — TRL builds reference by disabling adapter
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_path))

    loss, margin = _final_dpo_metrics(trainer)

    return DPORunResult(
        user_id=sft_config.user_id,
        base_model=sft_config.base_model,
        adapter_dir=output_path,
        base_adapter_dir=Path(base_adapter_dir) if base_adapter_dir else None,
        num_pairs=len(train_ds),
        final_loss=loss,
        final_reward_margin=margin,
        elapsed_seconds=round(time.time() - t0, 2),
        started_at=started,
        completed_at=datetime.now(),
        sft_config=sft_config,
        dpo_config=dpo_config,
    )


# --- Lazy imports ---


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


def _import_peft() -> tuple[Any, Any, Any, Any]:
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as e:
        raise ImportError("peft is required. Install with `pip install pmc[train]`.") from e
    return LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


def _import_trl_dpo() -> tuple[Any, Any]:
    try:
        from trl import DPOConfig as DPOConfigT, DPOTrainer
    except ImportError as e:
        raise ImportError("trl is required. Install with `pip install pmc[train]`.") from e
    return DPOTrainer, DPOConfigT


def _load_base(
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
            config.base_model, quantization_config=bnb, device_map="auto"
        )
        return prepare_model_for_kbit_training(model)
    return AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )


def _final_dpo_metrics(trainer: Any) -> tuple[float | None, float | None]:
    loss: float | None = None
    margin: float | None = None
    for entry in reversed(trainer.state.log_history):
        if loss is None and "loss" in entry:
            loss = float(entry["loss"])
        if margin is None and "rewards/margins" in entry:
            margin = float(entry["rewards/margins"])
        if loss is not None and margin is not None:
            break
    return loss, margin
