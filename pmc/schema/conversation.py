"""Core data model: Message → Conversation → Completion.

This is the canonical representation that flows through the entire pipeline:
ingest → curate → train → eval → serve. Adapted from the Conversation/Completion
pattern where Completion = context + candidate responses = the training unit.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from pmc.schema.annotations import Annotation


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceType(StrEnum):
    EMAIL = "email"
    IMESSAGE = "imessage"
    NOTES = "notes"
    DOCUMENT = "document"
    SOCIAL = "social"
    MANUAL = "manual"


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str
    timestamp: datetime | None = None
    annotations: list[Annotation] = Field(default_factory=list)


class Conversation(BaseModel):
    """An ordered sequence of messages — the prompt/context for training."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    messages: list[Message]
    source_type: SourceType | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class CompletionCandidate(BaseModel):
    """A single candidate response, optionally annotated with scores/preferences."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    messages: list[Message]
    annotations: list[Annotation] = Field(default_factory=list)


class Completion(BaseModel):
    """The training unit: context conversation + one or more candidate responses.

    - SFT: 1 candidate, train on the response
    - Preference/DPO: 2+ candidates with PreferenceAnnotation
    - Reward model: N candidates with quality/style scores
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    conversation: Conversation
    candidates: list[CompletionCandidate]
    annotations: list[Annotation] = Field(default_factory=list)
    user_id: str | None = None
