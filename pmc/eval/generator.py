"""Model generator abstraction.

Eval needs to call "generate me a response" against both the personal model
and (for comparison evals) a base model. The actual generation backend is
swappable — we use a Protocol so tests can mock it without torch/vLLM/HF.

V0 ships:
- `ModelGenerator` protocol
- `MockGenerator` — deterministic responses from a dict, for tests
- `CallableGenerator` — wraps any `messages → str` function
- `HFGenerator` — lazy-imports transformers/peft, loads base + adapter

Add a vLLM or Anthropic generator alongside these later if needed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelGenerator(Protocol):
    """Generate a single response from a list of chat messages."""

    name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...


class MockGenerator:
    """Deterministic generator for tests.

    `responses` maps a substring → response. If no key matches the last user
    message, returns `default`. Every call is logged in `calls`.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "(mock response)",
        name: str = "mock",
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.name = name
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(messages)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        for key, response in self.responses.items():
            if key in last_user:
                return response
        return self.default


class CallableGenerator:
    """Adapt any function `(messages, **kwargs) → str` into a ModelGenerator."""

    def __init__(
        self,
        fn: Callable[..., str],
        name: str = "callable",
    ) -> None:
        self.fn = fn
        self.name = name

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return self.fn(messages, max_new_tokens=max_new_tokens, temperature=temperature)


class HFGenerator:
    """HuggingFace generator that loads base model + optional LoRA adapter.

    Lazy-imports torch/transformers/peft so the module is importable without
    them. Use `pip install pmc[train,serve]` to enable.
    """

    def __init__(
        self,
        base_model: str,
        adapter_dir: Path | str | None = None,
        *,
        device: str = "auto",
        use_4bit: bool = True,
        name: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.device = device
        self.use_4bit = use_4bit
        self.name = name or (f"hf:{adapter_dir}" if adapter_dir else f"hf:{base_model}")
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_hf()

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: dict[str, Any] = {"device_map": self.device}
        if self.use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        if self.adapter_dir is not None:
            PeftModel = _import_peft_model()
            self._model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        else:
            self._model = base
        self._model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self._load()
        torch, _, _, _ = _import_hf()
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs.shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _import_hf() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "torch + transformers required for HFGenerator. "
            "Install with `pip install pmc[train]`."
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _import_peft_model() -> Any:
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft is required. Install with `pip install pmc[train]`.") from e
    return PeftModel
