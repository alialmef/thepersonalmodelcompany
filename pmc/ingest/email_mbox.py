"""Email ingest from mbox files.

Mbox is the standard format Gmail/Outlook/Apple Mail use for export. We use
stdlib `mailbox` to parse, extract plain-text bodies, and thread messages by
their RFC 5322 References/In-Reply-To headers.
"""

from __future__ import annotations

import email.utils
import mailbox
from collections.abc import Iterator
from datetime import datetime
from email.message import Message as EmailMessage
from pathlib import Path

from pmc.ingest.base import Ingestor, RawItem, normalize_identifier
from pmc.schema.conversation import SourceType


class MboxIngestor(Ingestor):
    """Read an mbox file (Gmail export, Apple Mail export, etc.).

    user_emails: addresses that identify the user — used to set `is_user` on
    each message so the normalizer knows which side of the thread to train on.
    """

    source_type = SourceType.EMAIL

    def __init__(self, user_emails: list[str]) -> None:
        self.user_emails = {normalize_identifier(e) for e in user_emails}

    def ingest(self, source: Path | str) -> Iterator[RawItem]:
        path = Path(source)
        if not path.is_file():
            return
        box = mailbox.mbox(str(path), create=False)
        try:
            for key in box.keys():
                msg = box.get_message(key)
                item = self._to_raw_item(msg)
                if item is not None:
                    yield item
        finally:
            box.close()

    def _to_raw_item(self, msg: EmailMessage) -> RawItem | None:
        body = _extract_plain_body(msg)
        if not body.strip():
            return None

        message_id = (msg.get("Message-ID") or "").strip("<>").strip()
        if not message_id:
            return None

        from_header = msg.get("From", "")
        _, from_addr = email.utils.parseaddr(from_header)
        from_addr_norm = normalize_identifier(from_addr) if from_addr else ""

        in_reply_to = (msg.get("In-Reply-To") or "").strip("<>").strip()
        references = msg.get("References", "")
        thread_id = _thread_root(message_id, in_reply_to, references)

        timestamp = _parse_date(msg.get("Date"))
        subject = (msg.get("Subject") or "").strip()

        return RawItem(
            source_type=self.source_type,
            source_id=message_id,
            content=body.strip(),
            timestamp=timestamp,
            thread_id=thread_id,
            author_identifier=from_addr_norm or None,
            is_user=from_addr_norm in self.user_emails if from_addr_norm else None,
            subject=subject or None,
            metadata={
                "from": from_header,
                "to": msg.get("To", ""),
                "cc": msg.get("Cc", ""),
            },
        )


def _extract_plain_body(msg: EmailMessage) -> str:
    """Pull the text/plain body out of an email, falling back to text/html stripped."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition", "").startswith("attachment"):
                return _decode_payload(part)
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                return _strip_html(_decode_payload(part))
        return ""
    ctype = msg.get_content_type()
    if ctype == "text/plain":
        return _decode_payload(msg)
    if ctype == "text/html":
        return _strip_html(_decode_payload(msg))
    return ""


def _decode_payload(part: EmailMessage) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return str(payload or "")
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        import re
        return re.sub(r"<[^>]+>", " ", html)
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _thread_root(message_id: str, in_reply_to: str, references: str) -> str:
    """The thread_id is the root of the reply chain — first reference, or self."""
    if references:
        refs = [r.strip("<>").strip() for r in references.split() if r.strip()]
        if refs:
            return refs[0]
    if in_reply_to:
        return in_reply_to
    return message_id


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None
