"""PII detection and optional redaction.

We detect with regex (V0) and emit PIIAnnotations so downstream code can decide
what to redact. Some PII is high-severity (SSN, credit card) and gets redacted
by default; some is the user's own identity and they may want to keep it.
"""

from __future__ import annotations

import re

from pmc.schema.annotations import PIIAnnotation, PIIType

# Conservative patterns — we err toward false negatives over false positives.
PATTERNS: list[tuple[PIIType, re.Pattern[str], float]] = [
    (
        PIIType.EMAIL_ADDRESS,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        0.5,
    ),
    (
        PIIType.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        1.0,
    ),
    (
        PIIType.CREDIT_CARD,
        re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
        1.0,
    ),
    (
        PIIType.PHONE_NUMBER,
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s\-.]?)?"
            r"(?:\(\d{3}\)|\d{3})[\s\-.]\d{3}[\s\-.]\d{4}"
            r"(?!\d)"
        ),
        0.6,
    ),
]

SEVERE_PII: set[PIIType] = {PIIType.SSN, PIIType.CREDIT_CARD}


def detect_pii(text: str) -> list[PIIAnnotation]:
    """Find PII spans in text. Returns annotations sorted by start position."""
    found: list[PIIAnnotation] = []
    for pii_type, pattern, sensitivity in PATTERNS:
        for m in pattern.finditer(text):
            found.append(
                PIIAnnotation(
                    pii_type=pii_type,
                    start=m.start(),
                    end=m.end(),
                    sensitivity=sensitivity,
                )
            )
    found.sort(key=lambda a: (a.start, a.end))
    return _dedupe_overlapping(found)


def redact_text(
    text: str,
    annotations: list[PIIAnnotation],
    *,
    only_types: set[PIIType] | None = None,
    placeholder: str = "[REDACTED:{type}]",
) -> tuple[str, list[PIIAnnotation]]:
    """Replace PII spans with placeholders. Returns the redacted text and the
    annotations marked `redacted=True`.

    Spans are replaced in reverse order so indices stay valid.
    """
    annotations_sorted = sorted(annotations, key=lambda a: a.start, reverse=True)
    result = text
    out_annotations: list[PIIAnnotation] = []
    for ann in annotations_sorted:
        if only_types is not None and ann.pii_type not in only_types:
            out_annotations.append(ann)
            continue
        replacement = placeholder.format(type=ann.pii_type.value)
        result = result[: ann.start] + replacement + result[ann.end :]
        out_annotations.append(
            PIIAnnotation(
                pii_type=ann.pii_type,
                start=ann.start,
                end=ann.start + len(replacement),
                redacted=True,
                sensitivity=ann.sensitivity,
            )
        )
    out_annotations.sort(key=lambda a: a.start)
    return result, out_annotations


def _dedupe_overlapping(annotations: list[PIIAnnotation]) -> list[PIIAnnotation]:
    """If two patterns matched overlapping spans, keep the more severe / longer one."""
    if not annotations:
        return annotations
    kept: list[PIIAnnotation] = [annotations[0]]
    for ann in annotations[1:]:
        last = kept[-1]
        if ann.start < last.end:
            # overlap — keep whichever has higher sensitivity, then longer span
            ann_len = ann.end - ann.start
            last_len = last.end - last.start
            if (ann.sensitivity, ann_len) > (last.sensitivity, last_len):
                kept[-1] = ann
        else:
            kept.append(ann)
    return kept
