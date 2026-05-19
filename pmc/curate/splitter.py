"""Split a Conversation into one or more Completions (training units).

For each ASSISTANT message in the conversation, we emit a Completion whose
context is everything before that message and whose single candidate is the
ASSISTANT message itself. This is the standard SFT framing — train the model
to produce the user's actual response given the preceding context.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from pmc.curate.clean import clean as clean_text
from pmc.schema.annotations import SourceAnnotation
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
)


def split_conversation(
    conv: Conversation,
    *,
    clean: bool = True,
    min_response_chars: int = 5,
    include_empty_context: bool = True,
) -> list[Completion]:
    """Convert a Conversation into a list of training Completions.

    - clean: if True, run boilerplate stripping on each message's content.
    - min_response_chars: skip completions where the user's response is too short.
    - include_empty_context: standalone writing (single ASSISTANT message) has
      no prior context. If False, those are dropped — useful when you intend to
      synthesize a prompt for them separately.
    """
    completions: list[Completion] = []
    for i, msg in enumerate(conv.messages):
        if msg.role != Role.ASSISTANT:
            continue

        content = clean_text(msg.content) if clean else msg.content
        if len(content) < min_response_chars:
            continue

        context_msgs = [_clean_message(m, clean=clean) for m in conv.messages[:i]]
        context_msgs = [m for m in context_msgs if m.content.strip()]

        if not context_msgs and not include_empty_context:
            continue

        candidate = CompletionCandidate(
            messages=[
                Message(
                    role=msg.role,
                    content=content,
                    timestamp=msg.timestamp,
                    annotations=list(msg.annotations),
                )
            ],
            annotations=[a for a in msg.annotations if isinstance(a, SourceAnnotation)],
        )
        completions.append(
            Completion(
                conversation=Conversation(
                    messages=context_msgs,
                    source_type=conv.source_type,
                ),
                candidates=[candidate],
            )
        )
    return completions


def split_many(
    conversations: Iterable[Conversation],
    **kwargs: bool | int,
) -> Iterator[Completion]:
    for conv in conversations:
        yield from split_conversation(conv, **kwargs)  # type: ignore[arg-type]


def _clean_message(msg: Message, *, clean: bool) -> Message:
    content = clean_text(msg.content) if clean else msg.content
    return Message(
        role=msg.role,
        content=content,
        timestamp=msg.timestamp,
        annotations=list(msg.annotations),
    )
