"""PMCServer — top-level orchestrator that ties registry + engine.

Use directly in process for tests / scripts, or wrap in the FastAPI app
(`pmc.serve.api.create_app`) for HTTP serving.

Responsibilities:
- Route incoming chat requests to the right user's adapter
- Track usage via the registry
- Provide model listing for the OpenAI-compatible /v1/models endpoint
- Produce bundle/adapter zips for the export endpoint
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

from pmc.serve.engine import InferenceEngine, MockEngine
from pmc.serve.export import export_adapter_only, export_bundle
from pmc.serve.memory_context import MemoryContextProvider, enrich_messages
from pmc.serve.registry import AdapterRegistry
from pmc.serve.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelList,
    Usage,
)


class PMCServer:
    """High-level server: registry + engine + API methods.

    Pass `memory_provider` to enable identity-prompt + retrieved-context
    injection on every chat. When omitted, chat() behaves exactly as it did
    before memory existed — useful for tests and for engines that don't
    benefit from RAG.
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        engine: InferenceEngine | None = None,
        memory_provider: MemoryContextProvider | None = None,
        retrieval_k: int = 5,
    ) -> None:
        self.registry = registry
        self.engine: InferenceEngine = engine or MockEngine()
        self.memory_provider = memory_provider
        self.retrieval_k = retrieval_k

    def _prepared_messages(
        self,
        user_id: str,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Enrich messages with identity + retrieved memory when configured."""
        if self.memory_provider is None:
            return messages
        try:
            context = self.memory_provider.get(user_id)
        except Exception:
            # Memory is opportunistic — never block a chat because the store
            # is missing or corrupt. Just serve the raw messages.
            return messages
        return enrich_messages(
            messages,
            context,
            retrieval_k=self.retrieval_k,
        )

    # -- chat completions --------------------------------------------------

    def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        user_id = request.user or request.model
        record = self.registry.require(user_id)

        if record.base_model != self.engine.base_model:
            # Engine and adapter must agree on the base. Bail loudly rather than
            # silently producing junk from a mismatched base.
            raise ValueError(
                f"Adapter for {user_id!r} expects base {record.base_model!r}, "
                f"engine is serving {self.engine.base_model!r}"
            )

        stop = [request.stop] if isinstance(request.stop, str) else request.stop
        messages = [m.model_dump() for m in request.messages]
        messages = self._prepared_messages(user_id, messages)

        text, usage = self.engine.chat(
            record=record,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
        )
        self.registry.mark_served(user_id)

        return ChatCompletionResponse(
            model=user_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="length" if usage["completion_tokens"] >= request.max_tokens else "stop",
                )
            ],
            usage=Usage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["prompt_tokens"] + usage["completion_tokens"],
            ),
        )

    def chat_stream(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
        """Yield OpenAI-style streaming chunks for a chat completion."""
        user_id = request.user or request.model
        record = self.registry.require(user_id)
        if record.base_model != self.engine.base_model:
            raise ValueError(
                f"Adapter for {user_id!r} expects base {record.base_model!r}, "
                f"engine is serving {self.engine.base_model!r}"
            )
        stop = [request.stop] if isinstance(request.stop, str) else request.stop
        messages = [m.model_dump() for m in request.messages]
        messages = self._prepared_messages(user_id, messages)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # First chunk announces the role (OpenAI convention)
        yield ChatCompletionChunk(
            id=completion_id,
            model=user_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(role="assistant"),
                )
            ],
        )

        emitted_chars = 0
        try:
            for chunk_text in self.engine.chat_stream(
                record=record,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
            ):
                if not chunk_text:
                    continue
                emitted_chars += len(chunk_text)
                yield ChatCompletionChunk(
                    id=completion_id,
                    model=user_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=chunk_text),
                        )
                    ],
                )
        finally:
            # Final chunk: empty delta + finish_reason. Mark served once.
            self.registry.mark_served(user_id)
            # ~4 chars per token approximation for the finish_reason heuristic
            approx_tokens = max(1, emitted_chars // 4)
            finish = "length" if approx_tokens >= request.max_tokens else "stop"
            yield ChatCompletionChunk(
                id=completion_id,
                model=user_id,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason=finish,
                    )
                ],
            )

    # -- model listing -----------------------------------------------------

    def list_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelInfo(
                    id=r.user_id,
                    created=int(r.registered_at.timestamp()),
                    base_model=r.base_model,
                    adapter_size_mb=r.adapter_size_mb,
                )
                for r in self.registry.list_records()
            ]
        )

    def get_model(self, user_id: str) -> ModelInfo:
        record = self.registry.require(user_id)
        return ModelInfo(
            id=record.user_id,
            created=int(record.registered_at.timestamp()),
            base_model=record.base_model,
            adapter_size_mb=record.adapter_size_mb,
        )

    # -- export ------------------------------------------------------------

    def export_model(
        self,
        user_id: str,
        output_zip: Path | str,
        *,
        adapter_only: bool = False,
    ) -> Path:
        record = self.registry.require(user_id)
        if adapter_only:
            return export_adapter_only(record, output_zip)
        return export_bundle(record, output_zip)

    def delete_model(self, user_id: str, *, delete_files: bool = False) -> bool:
        """Unregister and optionally hard-delete the adapter + bundle on disk."""
        self.engine.evict(user_id)
        return self.registry.unregister(user_id, delete_files=delete_files)
