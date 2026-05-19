"""iMessage ingest from the macOS chat.db SQLite database.

The default location is ~/Library/Messages/chat.db. Each row in `message` joined
with `handle` (sender) and `chat_message_join`/`chat` (conversation) gives us a
RawItem with thread_id = chat_identifier and is_user = message.is_from_me.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from pmc.ingest.base import Ingestor, RawItem
from pmc.schema.conversation import SourceType

# iMessage timestamps are nanoseconds since 2001-01-01 (Apple "Mac absolute time")
_APPLE_EPOCH = datetime(2001, 1, 1)


class IMessageIngestor(Ingestor):
    """Read messages from an iMessage chat.db file (macOS Messages app)."""

    source_type = SourceType.IMESSAGE

    def ingest(self, source: Path | str) -> Iterator[RawItem]:
        path = Path(source)
        if not path.is_file():
            return
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield from self._read_messages(conn)
        finally:
            conn.close()

    def _read_messages(self, conn: sqlite3.Connection) -> Iterator[RawItem]:
        query = """
        SELECT
            m.ROWID            AS msg_id,
            m.text             AS text,
            m.is_from_me       AS is_from_me,
            m.date             AS date_ns,
            h.id               AS handle_id,
            c.chat_identifier  AS chat_id,
            c.display_name     AS chat_name
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN chat c ON c.ROWID = cmj.chat_id
        WHERE m.text IS NOT NULL AND length(m.text) > 0
        ORDER BY m.date ASC
        """
        for row in conn.execute(query):
            text = (row["text"] or "").strip()
            if not text:
                continue
            yield RawItem(
                source_type=self.source_type,
                source_id=f"imessage:{row['msg_id']}",
                content=text,
                timestamp=_apple_time_to_datetime(row["date_ns"]),
                thread_id=row["chat_id"] or row["handle_id"] or "unknown",
                author_identifier=row["handle_id"] or None,
                is_user=bool(row["is_from_me"]),
                metadata={
                    "chat_name": row["chat_name"] or "",
                },
            )


def _apple_time_to_datetime(date_value: int | None) -> datetime | None:
    """Mac absolute time is in nanoseconds since 2001-01-01 (modern iOS/macOS).

    Older databases stored seconds. We detect by magnitude.
    """
    if date_value is None or date_value == 0:
        return None
    # Modern iMessage: nanoseconds → > 10^15 for any post-2001 date
    if date_value > 10**12:
        seconds = date_value / 1_000_000_000
    else:
        seconds = date_value
    try:
        return _APPLE_EPOCH + timedelta(seconds=seconds)
    except (OverflowError, ValueError):
        return None
