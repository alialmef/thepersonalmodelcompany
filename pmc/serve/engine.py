"""Inference engine abstraction.

The engine is what actually runs a model and produces text. We have:

- `InferenceEngine` Protocol — the interface (sync `chat()` + lifecycle hooks)
- `MockEngine` — deterministic, no deps. For tests and offline development.
- `VLLMEngine` — lazy-imports vLLM. Single base model loaded once; LoRA
  adapters loaded per-request via vLLM's multi-LoRA support. This matches
  the architecture from the analysis doc.

Warmth (which adapters are currently in GPU memory) is the engine's concern,
not the registry's — vLLM handles its own LRU under the hood. We expose
`warm()` and `evict()` for explicit control.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pmc.serve.registry import AdapterRecord


@runtime_checkable
class InferenceEngine(Protocol):
    """Sync interface every engine implements."""

    base_model: str

    def chat(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Return (response_text, usage_counts) where usage = {prompt_tokens, completion_tokens}."""

    def chat_stream(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield incremental text chunks as they're generated.

        The final iteration MUST exhaust the stream — caller assumes when the
        iterator stops, generation is done.
        """
        ...

    def warm(self, record: AdapterRecord) -> None: ...
    def evict(self, user_id: str) -> bool: ...
    def shutdown(self) -> None: ...


class MockEngine:
    """Deterministic engine for tests.

    `responses` maps a substring of the last user message → response. Falls back
    to `default`. Tracks calls and warmth state.
    """

    def __init__(
        self,
        base_model: str = "mock/base",
        responses: dict[str, str] | None = None,
        default: str = "(mock response)",
    ) -> None:
        self.base_model = base_model
        self.responses = responses or {}
        self.default = default
        self.calls: list[dict[str, Any]] = []
        self.warm_users: set[str] = set()

    def chat(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        self.warm_users.add(record.user_id)
        self.calls.append({
            "user_id": record.user_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        response = self.default
        for key, text in self.responses.items():
            if key in last_user:
                response = text
                break

        # naive token approximation: 1 token per 4 chars
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        usage = {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": max(1, len(response) // 4),
        }
        return response, usage

    def chat_stream(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield the chosen response word-by-word, for testing the SSE path."""
        text, _usage = self.chat(record, messages, max_tokens, temperature, top_p, stop)
        if not text:
            return
        # yield with whitespace preserved so the join is lossless
        import re
        parts = re.findall(r"\S+\s*|\s+", text)
        for part in parts:
            yield part

    def warm(self, record: AdapterRecord) -> None:
        self.warm_users.add(record.user_id)

    def evict(self, user_id: str) -> bool:
        if user_id in self.warm_users:
            self.warm_users.remove(user_id)
            return True
        return False

    def shutdown(self) -> None:
        self.warm_users.clear()


class VLLMEngine:
    """vLLM-backed engine with multi-LoRA serving.

    Loads the base model once at construction time, then serves any number of
    adapters by passing a `LoRARequest` per generate call. vLLM caches recently
    used adapters in GPU memory.

    Heavy imports happen in `__init__`. Don't construct this on a machine
    without a CUDA-capable GPU.
    """

    def __init__(
        self,
        base_model: str,
        *,
        max_lora_rank: int = 64,
        max_loras: int = 8,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ) -> None:
        self.base_model = base_model
        LLM, _SamplingParams, _LoRARequest = _import_vllm()
        self._SamplingParams = _SamplingParams
        self._LoRARequest = _LoRARequest
        self._llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=max_loras,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        # vLLM identifies adapters by an integer ID; we map user_id → int.
        self._lora_ids: dict[str, int] = {}
        self._lora_paths: dict[str, Path] = {}

    def _lora_request(self, record: AdapterRecord) -> Any:
        if record.user_id not in self._lora_ids:
            self._lora_ids[record.user_id] = len(self._lora_ids) + 1
            self._lora_paths[record.user_id] = Path(record.adapter_dir)
        return self._LoRARequest(
            lora_name=record.user_id,
            lora_int_id=self._lora_ids[record.user_id],
            lora_path=str(self._lora_paths[record.user_id]),
        )

    def chat(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        sampling = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        tokenizer = self._llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        outputs = self._llm.generate(
            [prompt],
            sampling_params=sampling,
            lora_request=self._lora_request(record),
        )
        output = outputs[0]
        text = output.outputs[0].text
        usage = {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
        }
        return text, usage

    def chat_stream(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """True token-by-token streaming via vLLM's add_request + step loop.

        We submit a single request to the engine then drive it forward with
        step() calls, emitting only the new text since the previous step.
        """
        _LLM, SamplingParams, LoRARequest = _import_vllm()
        sampling = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop
        )
        tokenizer = self._llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        request_id = f"stream-{uuid.uuid4().hex[:12]}"
        engine = self._llm.llm_engine  # internal sync engine
        engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=sampling,
            lora_request=self._lora_request(record),
        )
        emitted = ""
        while True:
            step_outputs = engine.step()
            for out in step_outputs:
                if out.request_id != request_id:
                    continue
                latest = out.outputs[0].text
                if len(latest) > len(emitted):
                    delta = latest[len(emitted):]
                    emitted = latest
                    yield delta
                if out.finished:
                    return
            if not engine.has_unfinished_requests():
                return

    def warm(self, record: AdapterRecord) -> None:
        # vLLM warms on first use; nothing to do here.
        self._lora_request(record)

    def evict(self, user_id: str) -> bool:
        # vLLM evicts via its own LRU; we just drop our local mapping.
        return self._lora_ids.pop(user_id, None) is not None

    def shutdown(self) -> None:
        try:
            self._llm.shutdown()  # vLLM ≥0.6
        except AttributeError:
            pass


def _import_vllm() -> tuple[Any, Any, Any]:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as e:
        raise ImportError(
            "vllm is required for VLLMEngine. Install with `pip install pmc[serve-gpu]`."
        ) from e
    return LLM, SamplingParams, LoRARequest
