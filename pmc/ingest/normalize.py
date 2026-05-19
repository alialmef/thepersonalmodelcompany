"""Normalize RawItems into Conversation objects.

Two patterns:
- Threaded sources (email, messages): items with a shared thread_id are grouped
  and sorted by timestamp into a Conversation. Each message's role is mapped:
  is_user=True → ASSISTANT (the model trains to mimic the user), others → USER.
- Standalone sources (documents, notes): each item becomes a single-message
  Conversation with role=ASSISTANT. Curate will synthesize a prompt later.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator

from pmc.ingest.base import RawItem
from pmc.schema.annotations import SourceAnnotation
from pmc.schema.conversation import (
    Conversation,
    Message,
    Role,
    SourceType,
)

THREADED_SOURCES = {SourceType.EMAIL, SourceType.IMESSAGE}


class Normalizer:
    """Group RawItems into Conversation objects.

    min_user_messages: a threaded conversation is only emitted if it contains at
    least this many messages from the user — otherwise there's nothing to train.
    """

    def __init__(self, min_user_messages: int = 1) -> None:
        self.min_user_messages = min_user_messages

    def normalize(self, items: Iterable[RawItem]) -> Iterator[Conversation]:
        threaded: dict[tuple[SourceType, str], list[RawItem]] = defaultdict(list)
        for item in items:
            if item.source_type in THREADED_SOURCES and item.thread_id:
                threaded[(item.source_type, item.thread_id)].append(item)
            else:
                conv = self._standalone_conversation(item)
                if conv is not None:
                    yield conv

        for (source_type, thread_id), thread_items in threaded.items():
            conv = self._threaded_conversation(source_type, thread_id, thread_items)
            if conv is not None:
                yield conv

    def _standalone_conversation(self, item: RawItem) -> Conversation | None:
        if not item.content.strip():
            return None
        message = Message(
            role=Role.ASSISTANT,
            content=item.content,
            timestamp=item.timestamp,
            annotations=[_source_annotation(item)],
        )
        return Conversation(
            messages=[message],
            source_type=item.source_type,
        )

    def _threaded_conversation(
        self,
        source_type: SourceType,
        thread_id: str,
        items: list[RawItem],
    ) -> Conversation | None:
        sorted_items = sorted(
            items, key=lambda i: (i.timestamp is None, i.timestamp)
        )
        messages: list[Message] = []
        user_count = 0
        for item in sorted_items:
            role = Role.ASSISTANT if item.is_user else Role.USER
            if role == Role.ASSISTANT:
                user_count += 1
            messages.append(
                Message(
                    role=role,
                    content=item.content,
                    timestamp=item.timestamp,
                    annotations=[_source_annotation(item)],
                )
            )

        if user_count < self.min_user_messages:
            return None

        messages = _merge_consecutive_same_role(messages)
        return Conversation(messages=messages, source_type=source_type)


def _source_annotation(item: RawItem) -> SourceAnnotation:
    return SourceAnnotation(
        source_type=item.source_type.value,
        source_id=item.source_id,
        timestamp=item.timestamp,
        metadata={**item.metadata, **({"thread_id": item.thread_id} if item.thread_id else {})},
    )


def _merge_consecutive_same_role(messages: list[Message]) -> list[Message]:
    """If the same person sent several messages in a row, collapse them."""
    if not messages:
        return messages
    merged: list[Message] = [messages[0]]
    for msg in messages[1:]:
        last = merged[-1]
        if msg.role == last.role:
            merged[-1] = Message(
                role=last.role,
                content=f"{last.content}\n\n{msg.content}",
                timestamp=last.timestamp,
                annotations=last.annotations + msg.annotations,
            )
        else:
            merged.append(msg)
    return merged
