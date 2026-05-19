"""OpenAI-compatible chat completion schemas.

We mirror the OpenAI Chat Completions shape so existing clients (openai-python,
LangChain, LlamaIndex, etc.) work against the PMC server without changes. The
`model` field is the PMC `user_id` — it routes to that user's adapter.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request body for /v1/chat/completions."""

    model: str  # PMC user_id — names the adapter to use
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] | str | None = None
    user: str | None = None  # ignored; for OpenAI client compat


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------- SSE streaming chunks (OpenAI-compatible) ----------


class ChatCompletionChunkDelta(BaseModel):
    """One incremental piece of a streamed response."""

    role: Literal["user", "assistant", "system"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """One SSE event in a streamed chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    """OpenAI-compatible /v1/models entry."""

    id: str  # user_id
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "pmc"
    base_model: str | None = None
    adapter_size_mb: float | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class APIError(BaseModel):
    error: dict[str, str]


def make_error(message: str, type_: str = "invalid_request_error", code: str = "") -> APIError:
    return APIError(error={"message": message, "type": type_, "code": code or type_})
