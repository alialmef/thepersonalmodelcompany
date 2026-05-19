"""DataSource spec — how the orchestrator describes things to ingest.

Each DataSource bundles:
- a `kind` (which ingestor to use)
- either a `path` on disk or pre-built `items`
- a `source_id` for storage partitioning (auto-derived from path if absent)
- per-kind config (e.g. `user_emails` for mbox so we know which side of an
  email thread is the user)
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from pmc.ingest.base import RawItem
from pmc.ingest.documents import DocumentIngestor
from pmc.ingest.email_mbox import MboxIngestor
from pmc.ingest.imessage import IMessageIngestor
from pmc.ingest.text import TextFileIngestor
from pmc.ingest.whatsapp import WhatsAppIngestor


class DataSourceKind(StrEnum):
    TEXT = "text"               # .txt / .md files via TextFileIngestor
    DOCUMENT = "document"       # .pdf / .docx via DocumentIngestor
    EMAIL_MBOX = "email_mbox"   # Gmail / Apple Mail mbox via MboxIngestor
    IMESSAGE = "imessage"       # macOS chat.db via IMessageIngestor
    WHATSAPP = "whatsapp"       # WhatsApp .txt export via WhatsAppIngestor
    RAW = "raw"                 # pre-built RawItems passed in directly


class DataSource(BaseModel):
    """One source of personal data for the pipeline to ingest."""

    kind: DataSourceKind
    path: Path | None = None
    source_id: str | None = None
    user_emails: list[str] = Field(default_factory=list)
    user_names: list[str] = Field(default_factory=list)
    items: list[RawItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> DataSource:
        if self.kind == DataSourceKind.RAW:
            if not self.items:
                raise ValueError("RAW source requires `items`")
        else:
            if self.path is None:
                raise ValueError(f"{self.kind} source requires `path`")
        if self.kind == DataSourceKind.EMAIL_MBOX and not self.user_emails:
            raise ValueError("EMAIL_MBOX requires `user_emails` to identify the user")
        if self.kind == DataSourceKind.WHATSAPP and not self.user_names:
            raise ValueError("WHATSAPP requires `user_names` to identify the user")
        return self

    def derived_source_id(self) -> str:
        """Stable per-source ID for storage partitioning."""
        if self.source_id:
            return self.source_id
        if self.path is not None:
            return f"{self.kind.value}-{self.path.stem}"
        return f"{self.kind.value}-raw"

    def ingest(self) -> Iterable[RawItem]:
        """Run the right ingestor and yield RawItems."""
        if self.kind == DataSourceKind.RAW:
            yield from self.items
            return
        assert self.path is not None
        if self.kind == DataSourceKind.TEXT:
            yield from TextFileIngestor().ingest(self.path)
        elif self.kind == DataSourceKind.DOCUMENT:
            yield from DocumentIngestor().ingest(self.path)
        elif self.kind == DataSourceKind.EMAIL_MBOX:
            yield from MboxIngestor(user_emails=self.user_emails).ingest(self.path)
        elif self.kind == DataSourceKind.IMESSAGE:
            yield from IMessageIngestor().ingest(self.path)
        elif self.kind == DataSourceKind.WHATSAPP:
            yield from WhatsAppIngestor(user_names=self.user_names).ingest(self.path)


def text_source(path: Path | str, source_id: str | None = None) -> DataSource:
    return DataSource(kind=DataSourceKind.TEXT, path=Path(path), source_id=source_id)


def document_source(path: Path | str, source_id: str | None = None) -> DataSource:
    return DataSource(kind=DataSourceKind.DOCUMENT, path=Path(path), source_id=source_id)


def mbox_source(
    path: Path | str,
    user_emails: list[str],
    source_id: str | None = None,
) -> DataSource:
    return DataSource(
        kind=DataSourceKind.EMAIL_MBOX,
        path=Path(path),
        user_emails=user_emails,
        source_id=source_id,
    )


def imessage_source(path: Path | str, source_id: str | None = None) -> DataSource:
    return DataSource(kind=DataSourceKind.IMESSAGE, path=Path(path), source_id=source_id)


def whatsapp_source(
    path: Path | str,
    user_names: list[str],
    source_id: str | None = None,
) -> DataSource:
    return DataSource(
        kind=DataSourceKind.WHATSAPP,
        path=Path(path),
        user_names=user_names,
        source_id=source_id,
    )


def raw_source(items: list[RawItem], source_id: str = "raw") -> DataSource:
    return DataSource(kind=DataSourceKind.RAW, items=items, source_id=source_id)


__all__ = [
    "DataSource",
    "DataSourceKind",
    "document_source",
    "imessage_source",
    "mbox_source",
    "raw_source",
    "text_source",
    "whatsapp_source",
]
