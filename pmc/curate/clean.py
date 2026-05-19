"""Strip boilerplate from message content before training.

Email signatures, quoted reply chains, "Sent from my iPhone" footers — these
add noise that hurts training. We remove them with conservative regex rules
that prefer leaving content untouched over over-trimming.
"""

from __future__ import annotations

import re

SIGNATURE_SEPARATORS = [
    re.compile(r"^\s*--\s*$", re.MULTILINE),
    re.compile(r"^\s*-{3,}\s*$", re.MULTILINE),
    re.compile(r"^\s*_{3,}\s*$", re.MULTILINE),
]

DEVICE_FOOTER_RE = re.compile(
    r"\n*\s*Sent (?:from|via) my (?:iPhone|iPad|Mac|Android|BlackBerry|Galaxy|"
    r"Phone|mobile device|smartphone)[^\n]*",
    re.IGNORECASE,
)

QUOTED_BLOCK_RE = re.compile(r"^(?:>\s?.*(?:\n|$))+", re.MULTILINE)

REPLY_PREAMBLE_RE = re.compile(
    r"\n*\s*On\s+.{0,80}?,?\s+.{0,80}?\s+wrote:\s*$",
    re.IGNORECASE | re.MULTILINE,
)

FORWARD_HEADER_RE = re.compile(
    r"\n*\s*-{2,}\s*Forwarded message\s*-{2,}.*",
    re.IGNORECASE | re.DOTALL,
)

CONFIDENTIALITY_RE = re.compile(
    r"\n*\s*(?:CONFIDENTIALITY NOTICE|This (?:e-?mail|message) (?:is|may be) (?:confidential|privileged))"
    r".*",
    re.IGNORECASE | re.DOTALL,
)


def strip_quoted_replies(text: str) -> str:
    text = REPLY_PREAMBLE_RE.split(text, maxsplit=1)[0]
    text = QUOTED_BLOCK_RE.sub("", text)
    return text


def strip_signature(text: str) -> str:
    earliest = len(text)
    for pattern in SIGNATURE_SEPARATORS:
        m = pattern.search(text)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest] if earliest < len(text) else text


def strip_device_footer(text: str) -> str:
    return DEVICE_FOOTER_RE.sub("", text)


def strip_forwarded(text: str) -> str:
    return FORWARD_HEADER_RE.split(text, maxsplit=1)[0]


def strip_confidentiality(text: str) -> str:
    return CONFIDENTIALITY_RE.split(text, maxsplit=1)[0]


def clean(text: str) -> str:
    """Apply all boilerplate strippers in a sensible order."""
    text = strip_quoted_replies(text)
    text = strip_forwarded(text)
    text = strip_signature(text)
    text = strip_device_footer(text)
    text = strip_confidentiality(text)
    return text.strip()
