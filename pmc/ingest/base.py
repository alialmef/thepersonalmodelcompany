"""Base abstractions for data ingest.

Each ingestor reads from one source type and yields RawItem objects.
RawItems are pre-normalization: source-specific metadata is preserved so the
Normalizer can group them into conversations and map authorship correctly.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pmc.schema.conversation import SourceType


class RawItem(BaseModel):
    """A single piece of content pulled from a source, pre-normalization.

    - For threaded sources (email, messages): each message in a thread is one
      RawItem and `thread_id` groups them together.
    - For standalone sources (documents, notes): a single RawItem with no
      thread, no author, content is just the body.
    """

    source_type: SourceType
    source_id: str
    content: str
    timestamp: datetime | None = None
    thread_id: str | None = None
    author_identifier: str | None = None
    is_user: bool | None = None
    subject: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable hash of content for deduplication."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


class Ingestor(ABC):
    """Reads raw items from a source path."""

    source_type: SourceType

    @abstractmethod
    def ingest(self, source: Path | str) -> Iterator[RawItem]:
        """Yield RawItem objects from the given source path."""
        ...


def normalize_identifier(identifier: str) -> str:
    """Normalize an email address / phone number / name for comparison."""
    return identifier.strip().lower()
