"""Schemas for the local laptop-world index."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WorldFileKind(StrEnum):
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    DATA = "data"
    OTHER = "other"


class WorldScanConfig(BaseModel):
    """Configuration for a laptop-world scan."""

    roots: list[str] = Field(default_factory=list)
    full_disk: bool = True
    max_files: int = 5_000
    max_file_bytes: int = 1_048_576
    include_text_preview: bool = True
    follow_symlinks: bool = False


class WorldFile(BaseModel):
    """One file-like object visible to the local model."""

    id: str = Field(default_factory=lambda: f"world-{uuid.uuid4().hex[:12]}")
    user_id: str
    path: str
    name: str
    extension: str = ""
    kind: WorldFileKind = WorldFileKind.OTHER
    size_bytes: int = 0
    modified_at: datetime | None = None
    indexed_at: datetime = Field(default_factory=datetime.now)
    content_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorldScanReport(BaseModel):
    """Summary of a laptop-world indexing pass."""

    id: str = Field(default_factory=lambda: f"scan-{uuid.uuid4().hex[:12]}")
    user_id: str
    roots: list[str]
    full_disk_requested: bool = True
    files_seen: int = 0
    files_indexed: int = 0
    dirs_skipped: int = 0
    bytes_indexed: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime = Field(default_factory=datetime.now)
