"""User profile, style profile, and data manifest."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class StyleProfile(BaseModel):
    """Extracted writing style characteristics for a user."""

    formality: float = 0.5
    verbosity: float = 0.5
    tone_tags: list[str] = Field(default_factory=list)
    vocabulary_markers: list[str] = Field(default_factory=list)
    sentence_length_avg: float | None = None
    common_phrases: list[str] = Field(default_factory=list)
    description: str = ""


class DataManifest(BaseModel):
    """Tracks what data was used for which training run."""

    training_run_id: uuid.UUID
    dataset_version: str
    num_examples: int
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    checksum: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class User(BaseModel):
    """A PMC user and their associated state."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    email: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    style_profile: StyleProfile | None = None
    data_manifests: list[DataManifest] = Field(default_factory=list)
    active_adapter_path: str | None = None
    total_training_examples: int = 0
