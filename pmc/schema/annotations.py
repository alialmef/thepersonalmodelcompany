"""Extensible metadata attached to messages, conversations, and candidates.

Annotations carry metadata that doesn't affect rendering — source provenance,
quality scores, style tags, PII flags, preference signals. The core data model
stays clean; annotations carry the training/eval context.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PIIType(StrEnum):
    EMAIL_ADDRESS = "email_address"
    PHONE_NUMBER = "phone_number"
    PHYSICAL_ADDRESS = "physical_address"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    OTHER = "other"


class Annotation(BaseModel):
    """Base for all annotation types."""

    type: str


class SourceAnnotation(Annotation):
    """Provenance — where this data came from."""

    type: str = "source"
    source_type: str
    source_id: str
    timestamp: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class QualityAnnotation(Annotation):
    """Quality scores assigned during curation."""

    type: str = "quality"
    style_signal: float = 0.0
    coherence: float = 0.0
    sufficient_context: float = 0.0
    duplicate_risk: float = 0.0
    boilerplate_score: float = 0.0
    overall: float = 0.0


class StyleAnnotation(Annotation):
    """Writing style characteristics extracted by the curation agent."""

    type: str = "style"
    formality: float | None = None
    verbosity: float | None = None
    tone_tags: list[str] = Field(default_factory=list)
    vocabulary_markers: list[str] = Field(default_factory=list)


class PreferenceAnnotation(Annotation):
    """User preference signal on a candidate."""

    type: str = "preference"
    chosen: bool
    dimension: str = "overall"
    score: float | None = None


class PIIAnnotation(Annotation):
    """PII detection result for a span of text."""

    type: str = "pii"
    pii_type: PIIType
    start: int
    end: int
    redacted: bool = False
    sensitivity: float = 1.0
