"""Inject retrieved memory into chat completion requests.

Sits between the HTTP layer and the model engine. For every chat
turn we:

  1. Pull the most recent user message as the query
  2. Hit `recall.retrieve` against the user's recall.db
  3. Format the top-k MemoryFragments as a structured context block
  4. Prepend that block as a system message so the model reads it
     before composing the response

If retrieval finds nothing, we leave the request untouched. If the
user has no recall.db yet (no consolidation has run), we also pass
through silently. Memory retrieval is a *boost*, never a *gate* — the
model should still respond even if memory is empty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from pmc.memory.recall.retrieve import RetrievalScope, retrieve
from pmc.memory.recall.store import RecallStore
from pmc.serve.schema import ChatCompletionRequest, ChatMessage


log = logging.getLogger(__name__)


# How many fragments to inject. More = richer context, but also more
# tokens / cost / latency on inference. 5 is a reasonable default for
# voice + memory experience without bloating the prompt.
DEFAULT_K = 5

# Block delimiters — give the model a clean signal that this is
# retrieved context, not user input.
CONTEXT_HEADER = "<personal_memory>"
CONTEXT_FOOTER = "</personal_memory>"


def inject_memory(
    request: ChatCompletionRequest,
    storage_root: Path,
    *,
    k: int = DEFAULT_K,
) -> ChatCompletionRequest:
    """Return a new request with a `<personal_memory>` system block prepended.

    Pure function over `request`: never mutates the caller's object.
    If memory is unavailable, returns the request unchanged.
    """
    user_id = request.model
    recall_path = Path(storage_root) / "users" / user_id / "recall.db"
    if not recall_path.is_file():
        return request

    query = _latest_user_message(request.messages)
    if not query:
        return request

    try:
        store = RecallStore(recall_path)
        fragments = retrieve(store, query, scope=RetrievalScope(), k=k)
        store.close()
    except Exception as e:
        log.warning("memory retrieval failed for %s: %s", user_id, e)
        return request

    if not fragments:
        return request

    block = _format_block(fragments)
    new_messages = _prepend_or_merge_system(request.messages, block)
    return request.model_copy(update={"messages": new_messages})


# ----------------------------------------------------------------------


def _latest_user_message(messages: Iterable[ChatMessage]) -> str:
    last_user = None
    for m in messages:
        if m.role == "user":
            last_user = m
    return (last_user.content if last_user else "").strip()


def _format_block(fragments) -> str:
    """Compose a compact, model-friendly memory block."""
    lines = [
        CONTEXT_HEADER,
        "Relevant context from the user's life, ranked by how likely it",
        "matters for this turn. Treat as background knowledge; use it",
        "only when it actually helps the user, never recite verbatim.",
        "",
    ]
    for i, f in enumerate(fragments, 1):
        when = f.time_start.strftime("%Y-%m-%d") if f.time_start else "unknown"
        ppl = (", ".join(f.participants[:3])) if f.participants else ""
        head = f"[{i}] {when} ({f.source})"
        if ppl:
            head += f" with {ppl}"
        lines.append(head)
        lines.append(f"    {f.summary}")
        if f.topics:
            lines.append(f"    topics: {', '.join(f.topics[:5])}")
    lines.append(CONTEXT_FOOTER)
    return "\n".join(lines)


def _prepend_or_merge_system(
    messages: list[ChatMessage], memory_block: str
) -> list[ChatMessage]:
    """If a system message exists, append the memory block to it.
    Otherwise insert a new system message at the top."""
    out = list(messages)
    if out and out[0].role == "system":
        merged = out[0].content.rstrip() + "\n\n" + memory_block
        out[0] = out[0].model_copy(update={"content": merged})
    else:
        out.insert(0, ChatMessage(role="system", content=memory_block))
    return out
