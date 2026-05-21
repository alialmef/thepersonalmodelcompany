"""Adapter save/load/merge utilities.

PEFT handles the actual file format (`adapter_config.json` +
`adapter_model.safetensors`). This module wraps PEFT's interface with helpers
for the cases PMC cares about:

- Verifying an adapter directory is well-formed (no torch required)
- Loading an adapter onto a base model (lazy import)
- Merging an adapter into the base model to produce a standalone model
  (for users who want a self-contained export, no PEFT runtime needed)
- Reporting adapter on-disk size for the user's manifest
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ADAPTER_CONFIG_FILE = "adapter_config.json"
# PEFT/HF use adapter_model.* ; MLX-LM writes adapters.safetensors. Accept both
# so the local Apple-Silicon training path validates the same as the cloud path.
ADAPTER_WEIGHTS_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapters.safetensors",
)


@dataclass
class AdapterInfo:
    path: Path
    rank: int | None
    alpha: int | None
    target_modules: list[str]
    base_model_name: str | None
    size_bytes: int


def adapter_info(adapter_dir: Path | str) -> AdapterInfo:
    """Inspect an adapter directory. Pure Python, no torch dep."""
    path = Path(adapter_dir)
    config_path = path / ADAPTER_CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"No {ADAPTER_CONFIG_FILE} at {path}")

    config = json.loads(config_path.read_text())
    weights_size = 0
    for name in ADAPTER_WEIGHTS_FILES:
        f = path / name
        if f.is_file():
            weights_size = f.stat().st_size
            break

    return AdapterInfo(
        path=path,
        rank=config.get("r"),
        alpha=config.get("lora_alpha"),
        target_modules=list(config.get("target_modules") or []),
        base_model_name=config.get("base_model_name_or_path"),
        size_bytes=weights_size,
    )


def is_valid_adapter(adapter_dir: Path | str) -> bool:
    """Quick check without throwing — for callers that need a boolean."""
    try:
        info = adapter_info(adapter_dir)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return info.size_bytes > 0 or any(
        (Path(adapter_dir) / w).is_file() for w in ADAPTER_WEIGHTS_FILES
    )


def load_adapter(
    base_model_name: str,
    adapter_dir: Path | str,
    *,
    use_4bit: bool = True,
) -> Any:
    """Load base model + adapter as a single PEFT model ready for inference."""
    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_hf()
    PeftModel = _import_peft_model()

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb, device_map="auto"
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

    return PeftModel.from_pretrained(base, str(adapter_dir))


def merge_adapter_into_base(
    base_model_name: str,
    adapter_dir: Path | str,
    output_dir: Path | str,
    *,
    save_tokenizer: bool = True,
) -> Path:
    """Merge a LoRA adapter into the base model and save a standalone model.

    Useful for users who want to export a model they can run anywhere without
    needing PEFT at inference time (e.g. Ollama/GGUF conversion).
    """
    torch, AutoModelForCausalLM, AutoTokenizer, _ = _import_hf()
    PeftModel = _import_peft_model()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output), safe_serialization=True)

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(str(output))

    return output


def _import_hf() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "torch + transformers required. Install with `pip install pmc[train]`."
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _import_peft_model() -> Any:
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft is required. Install with `pip install pmc[train]`.") from e
    return PeftModel
