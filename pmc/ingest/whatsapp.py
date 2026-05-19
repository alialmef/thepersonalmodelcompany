"""WhatsApp chat export ingest.

WhatsApp's "Export Chat" feature produces a .txt file with one message per line:

    [12/31/24, 11:59:00 PM] Alice: Happy New Year!
    [1/1/25, 12:01:15 AM] Bob: You too 🎉

Format varies slightly across locales and platforms — we accept a few common
variants and tolerate multi-line messages (continuation lines have no header).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pmc.ingest.base import Ingestor, RawItem, normalize_identifier
from pmc.schema.conversation import SourceType

# Matches a line starting with [date, time] sender: or date, time - sender:
_LINE_RE = re.compile(
    r"^[\[\(]?\s*"
    r"(?P<date>\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})"
    r"[,\s]+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?)"
    r"[\]\)]?"
    r"\s*[-–]?\s*"
    r"(?P<sender>[^:]+?):\s"
    r"(?P<message>.*)$"
)

_DATE_FORMATS = [
    "%m/%d/%y %I:%M:%S %p",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%y %H:%M",
    "%m/%d/%Y %H:%M",
]


class WhatsAppIngestor(Ingestor):
    """Read a WhatsApp .txt chat export.

    user_names: names that identify the user (matched against the sender field).
    """

    source_type = SourceType.IMESSAGE  # treat similarly for V0

    def __init__(self, user_names: list[str]) -> None:
        self.user_names = {normalize_identifier(n) for n in user_names}

    def ingest(self, source: Path | str) -> Iterator[RawItem]:
        path = Path(source)
        if not path.is_file():
            return
        thread_id = path.stem
        with path.open("r", encoding="utf-8", errors="replace") as f:
            yield from self._parse_lines(f, thread_id)

    def _parse_lines(self, lines: Iterator[str], thread_id: str) -> Iterator[RawItem]:
        current: dict[str, str | datetime | None] | None = None
        msg_index = 0
        for raw_line in lines:
            line = raw_line.rstrip("\n")
            match = _LINE_RE.match(line)
            if match:
                if current is not None:
                    yield self._build_item(current, thread_id, msg_index)
                    msg_index += 1
                sender = match.group("sender").strip()
                sender_norm = normalize_identifier(sender)
                current = {
                    "sender": sender,
                    "sender_norm": sender_norm,
                    "timestamp": _parse_dt(match.group("date"), match.group("time")),
                    "content": match.group("message"),
                }
            elif current is not None and line.strip():
                current["content"] = f"{current['content']}\n{line}"
        if current is not None:
            yield self._build_item(current, thread_id, msg_index)

    def _build_item(
        self, current: dict[str, str | datetime | None], thread_id: str, index: int
    ) -> RawItem:
        sender_norm = str(current["sender_norm"])
        content = str(current["content"]).strip()
        ts = current["timestamp"] if isinstance(current["timestamp"], datetime) else None
        return RawItem(
            source_type=self.source_type,
            source_id=f"whatsapp:{thread_id}:{index}",
            content=content,
            timestamp=ts,
            thread_id=f"whatsapp:{thread_id}",
            author_identifier=sender_norm,
            is_user=sender_norm in self.user_names,
            metadata={"sender_raw": str(current["sender"])},
        )


def _parse_dt(date_str: str, time_str: str) -> datetime | None:
    combined = f"{date_str} {time_str}".strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None
