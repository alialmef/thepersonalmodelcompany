"""Tests for the ingest layer: ingestors + normalizer."""

from __future__ import annotations

import mailbox
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import pytest

from pmc.ingest import (
    IMessageIngestor,
    MboxIngestor,
    Normalizer,
    RawItem,
    TextFileIngestor,
    WhatsAppIngestor,
)
from pmc.schema.annotations import SourceAnnotation
from pmc.schema.conversation import Conversation, Role, SourceType


# ---------- Text file ingestor ----------


def test_text_file_ingestor_single_file(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("# My Note\n\nThis is a thought I had.")
    items = list(TextFileIngestor().ingest(f))
    assert len(items) == 1
    assert items[0].source_type == SourceType.NOTES
    assert items[0].content.startswith("# My Note")
    assert items[0].is_user is True
    assert items[0].subject == "note"


def test_text_file_ingestor_directory_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Hello A")
    (tmp_path / "b.md").write_text("Hello B")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("Hello C")
    (tmp_path / "skip.png").write_text("binary stuff")
    (tmp_path / "empty.txt").write_text("   \n  ")

    items = list(TextFileIngestor().ingest(tmp_path))
    contents = {item.content for item in items}
    assert contents == {"Hello A", "Hello B", "Hello C"}


def test_text_file_ingestor_missing_path(tmp_path: Path) -> None:
    assert list(TextFileIngestor().ingest(tmp_path / "nope")) == []


# ---------- Email mbox ingestor ----------


def _write_mbox(path: Path, messages: list[tuple[str, str, str, str, str]]) -> None:
    """Helper: write an mbox with (from, to, subject, message_id, body) tuples."""
    box = mailbox.mbox(str(path))
    try:
        for from_, to, subject, msg_id, body in messages:
            msg = EmailMessage()
            msg["From"] = from_
            msg["To"] = to
            msg["Subject"] = subject
            msg["Message-ID"] = f"<{msg_id}>"
            msg["Date"] = "Mon, 15 Jan 2025 12:00:00 +0000"
            msg.set_content(body)
            box.add(msg)
    finally:
        box.close()


def test_mbox_ingestor_basic(tmp_path: Path) -> None:
    mbox_path = tmp_path / "inbox.mbox"
    _write_mbox(
        mbox_path,
        [
            ("alice@example.com", "user@example.com", "Hello", "id1", "Hi there"),
            ("user@example.com", "alice@example.com", "Re: Hello", "id2", "Hey Alice"),
        ],
    )
    ingestor = MboxIngestor(user_emails=["user@example.com"])
    items = list(ingestor.ingest(mbox_path))
    assert len(items) == 2
    by_id = {i.source_id: i for i in items}
    assert by_id["id1"].is_user is False
    assert by_id["id2"].is_user is True
    assert by_id["id1"].author_identifier == "alice@example.com"


def test_mbox_ingestor_threading(tmp_path: Path) -> None:
    mbox_path = tmp_path / "thread.mbox"
    box = mailbox.mbox(str(mbox_path))
    try:
        msg1 = EmailMessage()
        msg1["From"] = "alice@example.com"
        msg1["Subject"] = "Q"
        msg1["Message-ID"] = "<root@x>"
        msg1["Date"] = "Mon, 15 Jan 2025 12:00:00 +0000"
        msg1.set_content("Question")
        box.add(msg1)

        msg2 = EmailMessage()
        msg2["From"] = "user@example.com"
        msg2["Subject"] = "Re: Q"
        msg2["Message-ID"] = "<reply@x>"
        msg2["In-Reply-To"] = "<root@x>"
        msg2["References"] = "<root@x>"
        msg2["Date"] = "Mon, 15 Jan 2025 13:00:00 +0000"
        msg2.set_content("Answer")
        box.add(msg2)
    finally:
        box.close()

    items = list(MboxIngestor(user_emails=["user@example.com"]).ingest(mbox_path))
    threads = {i.thread_id for i in items}
    assert threads == {"root@x"}


# ---------- iMessage ingestor ----------


def _create_imessage_db(path: Path, rows: list[dict]) -> None:
    """Create a minimal chat.db-shaped SQLite file for testing."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            is_from_me INTEGER,
            date INTEGER,
            handle_id INTEGER
        );
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        """
    )
    for row in rows:
        conn.execute(
            "INSERT OR IGNORE INTO handle (ROWID, id) VALUES (?, ?)",
            (row["handle_id"], row["handle"]),
        )
        conn.execute(
            "INSERT OR IGNORE INTO chat (ROWID, chat_identifier, display_name) VALUES (?, ?, ?)",
            (row["chat_id"], row["chat_identifier"], row.get("chat_name", "")),
        )
        conn.execute(
            "INSERT INTO message (ROWID, text, is_from_me, date, handle_id) VALUES (?, ?, ?, ?, ?)",
            (
                row["msg_id"],
                row["text"],
                row["is_from_me"],
                row["date"],
                row["handle_id"],
            ),
        )
        conn.execute(
            "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
            (row["chat_id"], row["msg_id"]),
        )
    conn.commit()
    conn.close()


def test_imessage_ingestor(tmp_path: Path) -> None:
    db = tmp_path / "chat.db"
    # Use nanosecond Apple time (post-2001 dates produce values > 10^15)
    base_ns = 7 * 365 * 24 * 3600 * 1_000_000_000  # ~2008
    _create_imessage_db(
        db,
        [
            {
                "msg_id": 1, "text": "Hey", "is_from_me": 0, "date": base_ns,
                "handle_id": 1, "handle": "+15551234567",
                "chat_id": 1, "chat_identifier": "chat-abc",
            },
            {
                "msg_id": 2, "text": "What's up", "is_from_me": 1, "date": base_ns + 60_000_000_000,
                "handle_id": 1, "handle": "+15551234567",
                "chat_id": 1, "chat_identifier": "chat-abc",
            },
        ],
    )
    items = list(IMessageIngestor().ingest(db))
    assert len(items) == 2
    assert items[0].is_user is False
    assert items[1].is_user is True
    assert items[0].thread_id == "chat-abc"
    assert items[0].timestamp is not None
    assert items[0].timestamp.year >= 2007


# ---------- WhatsApp ingestor ----------


def test_whatsapp_ingestor(tmp_path: Path) -> None:
    chat = tmp_path / "chat.txt"
    chat.write_text(
        "[12/31/24, 11:59:00 PM] Alice: Happy New Year!\n"
        "[1/1/25, 12:01:15 AM] Me: You too\n"
        "continuation line\n"
        "[1/1/25, 12:02:00 AM] Alice: 🎉\n",
        encoding="utf-8",
    )
    items = list(WhatsAppIngestor(user_names=["Me"]).ingest(chat))
    assert len(items) == 3
    assert items[0].author_identifier == "alice"
    assert items[1].is_user is True
    assert "continuation line" in items[1].content
    assert items[2].content == "🎉"


# ---------- Normalizer ----------


def test_normalizer_standalone_items() -> None:
    items = [
        RawItem(
            source_type=SourceType.NOTES,
            source_id="note-1",
            content="A thought",
            is_user=True,
        )
    ]
    conversations = list(Normalizer().normalize(items))
    assert len(conversations) == 1
    conv = conversations[0]
    assert len(conv.messages) == 1
    assert conv.messages[0].role == Role.ASSISTANT
    assert conv.messages[0].content == "A thought"
    assert isinstance(conv.messages[0].annotations[0], SourceAnnotation)


def test_normalizer_threaded_groups_by_thread_id() -> None:
    base = datetime(2025, 1, 1, 12, 0, 0)
    items = [
        RawItem(
            source_type=SourceType.EMAIL,
            source_id="msg-1",
            content="Hi, can you help?",
            timestamp=base,
            thread_id="thread-A",
            author_identifier="alice@example.com",
            is_user=False,
        ),
        RawItem(
            source_type=SourceType.EMAIL,
            source_id="msg-2",
            content="Sure, here's the plan.",
            timestamp=base + timedelta(hours=1),
            thread_id="thread-A",
            author_identifier="user@example.com",
            is_user=True,
        ),
        RawItem(
            source_type=SourceType.EMAIL,
            source_id="msg-3",
            content="Unrelated email",
            timestamp=base,
            thread_id="thread-B",
            author_identifier="bob@example.com",
            is_user=False,
        ),
    ]
    conversations = list(Normalizer().normalize(items))
    # thread-B has no user messages, should be dropped (default min_user_messages=1)
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.source_type == SourceType.EMAIL
    assert [m.role for m in conv.messages] == [Role.USER, Role.ASSISTANT]
    assert conv.messages[0].content.startswith("Hi")
    assert conv.messages[1].content.startswith("Sure")


def test_normalizer_merges_consecutive_same_role() -> None:
    base = datetime(2025, 1, 1, 12, 0, 0)
    items = [
        RawItem(
            source_type=SourceType.IMESSAGE,
            source_id="m1",
            content="hey",
            timestamp=base,
            thread_id="t1",
            is_user=False,
        ),
        RawItem(
            source_type=SourceType.IMESSAGE,
            source_id="m2",
            content="you there?",
            timestamp=base + timedelta(seconds=10),
            thread_id="t1",
            is_user=False,
        ),
        RawItem(
            source_type=SourceType.IMESSAGE,
            source_id="m3",
            content="yeah sup",
            timestamp=base + timedelta(seconds=20),
            thread_id="t1",
            is_user=True,
        ),
    ]
    conversations = list(Normalizer().normalize(items))
    assert len(conversations) == 1
    conv = conversations[0]
    assert len(conv.messages) == 2
    assert conv.messages[0].role == Role.USER
    assert "hey" in conv.messages[0].content
    assert "you there?" in conv.messages[0].content


def test_normalizer_drops_threads_without_user_messages() -> None:
    items = [
        RawItem(
            source_type=SourceType.EMAIL,
            source_id="m1",
            content="newsletter content",
            thread_id="newsletter-x",
            is_user=False,
        ),
    ]
    convs = list(Normalizer(min_user_messages=1).normalize(items))
    assert convs == []


def test_normalizer_end_to_end_text_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("Personal essay one")
    (tmp_path / "b.txt").write_text("Personal essay two")
    items = TextFileIngestor().ingest(tmp_path)
    convs = list(Normalizer().normalize(items))
    assert len(convs) == 2
    assert all(c.messages[0].role == Role.ASSISTANT for c in convs)


def test_raw_item_content_hash_stable() -> None:
    item1 = RawItem(source_type=SourceType.NOTES, source_id="a", content="same content")
    item2 = RawItem(source_type=SourceType.NOTES, source_id="b", content="same content")
    assert item1.content_hash() == item2.content_hash()


def test_conversation_is_pydantic_serializable() -> None:
    items = [
        RawItem(
            source_type=SourceType.EMAIL,
            source_id="m1",
            content="Hi",
            timestamp=datetime(2025, 1, 1),
            thread_id="t1",
            is_user=False,
        ),
        RawItem(
            source_type=SourceType.EMAIL,
            source_id="m2",
            content="Hello",
            timestamp=datetime(2025, 1, 1, 13, 0),
            thread_id="t1",
            is_user=True,
        ),
    ]
    conv = next(Normalizer().normalize(items))
    json_str = conv.model_dump_json()
    restored = Conversation.model_validate_json(json_str)
    assert len(restored.messages) == 2
    assert restored.messages[1].role == Role.ASSISTANT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
