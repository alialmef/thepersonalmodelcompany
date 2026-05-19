"""Together AI engine — multi-tenant LoRA serving via Together's serverless API.

Together AI offers an OpenAI-compatible API with native serverless multi-LoRA:
you upload a LoRA adapter (or reference one in HuggingFace), then route
inference requests to it via the `model` field. Pricing is per-token of the
base model, regardless of how many adapters you serve. This is our V0 serving
target — zero GPU operations on our end.

The user's training pipeline produces an adapter at
`{bundle_dir}/adapter/`. After training, `upload_adapter()` pushes it to
Together (HF dataset repo or direct upload — Together supports both) and stores
the returned ID on the AdapterRecord. Chat then routes by that ID.

Lazy import of `openai` (Together uses the OpenAI client pointed at their base URL).
Set `TOGETHER_API_KEY` in env.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pmc.serve.registry import AdapterRecord

DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct-Reference"


class TogetherEngine:
    """InferenceEngine impl backed by Together AI's hosted multi-LoRA serving.

    Adapter routing: when serving a user's request, we send `model=base_model`
    and pass the adapter reference via Together's `lora` extra field. The
    AdapterRecord's `metadata['together_adapter_id']` carries the upload ID
    returned by `upload_adapter()`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.base_model = base_model
        self.base_url = base_url
        self._api_key = api_key or os.environ.get("TOGETHER_API_KEY") or ""
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            # Check API key first — fails fast without needing the openai package
            if not self._api_key:
                raise RuntimeError(
                    "Together API key missing. Set TOGETHER_API_KEY or pass api_key="
                )
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai is required for TogetherEngine. "
                    "Install with `pip install openai`."
                ) from e
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    def chat(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        client = self._get_client()
        kwargs = self._build_request_kwargs(record, messages, max_tokens, temperature, top_p, stop)
        response = client.chat.completions.create(stream=False, **kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "completion_tokens": (
                getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            ),
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
        client = self._get_client()
        kwargs = self._build_request_kwargs(record, messages, max_tokens, temperature, top_p, stop)
        stream = client.chat.completions.create(stream=True, **kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    def _build_request_kwargs(
        self,
        record: AdapterRecord,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        adapter_id = record.metadata.get("together_adapter_id")
        # Together passes LoRA reference via extra_body — adjust if their API changes.
        extra_body: dict[str, Any] = {}
        if adapter_id:
            extra_body["lora"] = adapter_id
        kwargs: dict[str, Any] = {
            "model": self.base_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            kwargs["stop"] = stop
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    # -- adapter upload (called by orchestrator after training) -----------

    def upload_adapter(self, adapter_dir: Path | str, name: str) -> str:
        """Upload a LoRA adapter to Together. Returns the adapter ID for routing.

        Together accepts adapters via their `/v1/files` endpoint or directly from
        HuggingFace — exact API may evolve. For V0 we assume the user uploads
        manually or via Together's CLI and stores the returned ID in
        AdapterRecord.metadata['together_adapter_id'].
        """
        # Placeholder — Together's adapter upload API isn't fully spec'd here.
        # When we have a Together account, fill in:
        #   1. Tar/zip adapter_dir
        #   2. POST to Together's adapter upload endpoint
        #   3. Return the adapter ID
        raise NotImplementedError(
            "upload_adapter not yet implemented. For V0, upload adapters via "
            "Together's CLI or web console and store the ID in "
            "AdapterRecord.metadata['together_adapter_id']."
        )

    def warm(self, record: AdapterRecord) -> None:
        # Together handles all warmth internally; nothing to do.
        return

    def evict(self, user_id: str) -> bool:
        # Together manages adapter caching on their side.
        return False

    def shutdown(self) -> None:
        self._client = None


def set_together_adapter_id(record: AdapterRecord, adapter_id: str) -> None:
    """Store a Together adapter ID on a record so the engine can route to it."""
    record.metadata["together_adapter_id"] = adapter_id


__all__ = ["DEFAULT_BASE_MODEL", "DEFAULT_BASE_URL", "TogetherEngine", "set_together_adapter_id"]
