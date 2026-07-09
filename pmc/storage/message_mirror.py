"""Local mirror of the user's iMessages, owned by the user.

Why this exists: macOS gates ~/Library/Messages/chat.db behind Full
Disk Access, and TCC attributes an MCP server's file reads to the
*interpreter binary* when the host app (Claude Desktop, Cursor)
disclaims responsibility for its subprocesses — so live chat.db reads
from inside an MCP server fail even when the host app has FDA.

The fix is architectural, not a permissions dance: `pmc connect`
(which already requires an FDA-granted terminal) copies message text
into the user's own store, and search reads the mirror. No agent host
ever needs FDA.

The mirror also decodes `attributedBody` blobs (typedstream) — most
modern messages have NULL `text`, so a text-only search misses them.
Format notes: NSString marker, then `+` (0x2b, the typedstream char
type code) anchors a length-prefixed UTF-8 field. Length encodings:
single byte 0x01..0x7f, or 0x81 + u16 LE, or 0x82 + u32 LE.

Sender/chat identity is resolved to contact display names at build
time using the graph's people.jsonl, so "messages with <person>"
queries work — a conversation with someone rarely contains their name
in the text.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
from pathlib import Path

_APPLE_EPOCH_OFFSET = 978307200  # 2001-01-01 in unix seconds


def mirror_path(storage_root: Path, user_id: str) -> Path:
    return storage_root / "users" / user_id / "mirror" / "messages.db"


def decode_attributed_body(blob: bytes) -> str | None:
    """Extract the message string from a typedstream attributedBody."""
    if not blob:
        return None
    i = blob.find(b"NSString")
    if i < 0:
        return None
    # The `+` char-type code appears within a few bytes of the class
    # reference; anchor on it explicitly (scanning too wide risks
    # matching a `+` inside the message itself).
    j = blob.find(b"\x2b", i + 8, i + 24)
    if j < 0:
        return None
    k = j + 1
    if k >= len(blob):
        return None
    first = blob[k]
    if 0x01 <= first <= 0x7F:
        length, start = first, k + 1
    elif first == 0x81:
        if k + 3 > len(blob):
            return None
        length, start = struct.unpack_from("<H", blob, k + 1)[0], k + 3
    elif first == 0x82:
        if k + 5 > len(blob):
            return None
        length, start = struct.unpack_from("<I", blob, k + 1)[0], k + 5
    else:
        return None
    if length <= 0 or start + length > len(blob):
        return None
    return blob[start:start + length].decode("utf-8", errors="replace")


def _normalize_handle(handle: str | None) -> str | None:
    """Normalize a phone/email handle for name matching."""
    if not handle:
        return None
    h = handle.strip().lower()
    if "@" in h:
        return h
    digits = re.sub(r"\D", "", h)
    return digits[-10:] if len(digits) >= 10 else (digits or None)


def load_handle_names(storage_root: Path, user_id: str) -> dict[str, str]:
    """Map normalized phone/email handles → display names, from the
    graph's people.jsonl (populated by the contacts extractor)."""
    people = storage_root / "users" / user_id / "graph" / "people.jsonl"
    names: dict[str, str] = {}
    if not people.is_file():
        return names
    for line in people.read_text(errors="replace").splitlines():
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = (p.get("display_name") or "").strip()
        if not name:
            continue
        for handle in (p.get("phones") or []) + (p.get("emails") or []):
            key = _normalize_handle(handle)
            if key:
                names.setdefault(key, name)
    return names


def build_mirror(chat_db: Path, out_db: Path,
                 handle_names: dict[str, str] | None = None) -> int:
    """Rebuild the mirror from chat.db. Returns message count.

    Raises sqlite3.Error if chat.db is unreadable (no FDA). The caller
    runs in an FDA-granted context (`pmc connect` preflights this).
    """
    handle_names = handle_names or {}

    def name_for(handle: str | None) -> str | None:
        key = _normalize_handle(handle)
        return handle_names.get(key) if key else None

    src = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True, timeout=5.0)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_db.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    dst = sqlite3.connect(tmp)
    n = 0
    try:
        dst.execute(
            "CREATE TABLE messages ("
            " ts REAL, chat TEXT, chat_name TEXT,"
            " sender TEXT, sender_name TEXT, is_from_me INTEGER,"
            " text TEXT)"
        )
        rows = src.execute("""
            SELECT m.text, m.attributedBody, m.is_from_me, m.date,
                   h.id, c.chat_identifier
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN chat c ON c.ROWID = cmj.chat_id
        """)
        batch = []
        for text, body, is_from_me, date, handle_id, chat_id in rows:
            if not text and body:
                text = decode_attributed_body(body)
            if not text or not text.strip():
                continue
            ts = (date / 1e9 if date and date > 1e12 else date or 0)
            ts += _APPLE_EPOCH_OFFSET
            batch.append((
                ts, chat_id, name_for(chat_id),
                handle_id, name_for(handle_id),
                int(is_from_me or 0), text.strip(),
            ))
            n += 1
            if len(batch) >= 5000:
                dst.executemany(
                    "INSERT INTO messages VALUES (?,?,?,?,?,?,?)", batch)
                batch = []
        if batch:
            dst.executemany(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)", batch)
        dst.execute("CREATE INDEX idx_messages_ts ON messages(ts)")
        dst.commit()
    finally:
        dst.close()
        src.close()
    tmp.replace(out_db)
    return n


def search_mirror(db_path: Path, query: str, limit: int,
                  person: str | None = None,
                  since_ts: float | None = None) -> list[tuple]:
    """Search by text substring and/or person and/or time, newest first.

    `person` matches contact name or raw handle, and includes messages
    the user sent in that person's conversation (via the chat columns).
    Returns (ts, sender, sender_name, is_from_me, text) tuples.
    """
    clauses, params = [], []
    if query and query.strip():
        clauses.append("text LIKE ?")
        params.append(f"%{query.strip()}%")
    if person and person.strip():
        p = f"%{person.strip()}%"
        clauses.append(
            "(sender_name LIKE ? OR chat_name LIKE ?"
            " OR sender LIKE ? OR chat LIKE ?)")
        params.extend([p, p, p, p])
    if since_ts is not None:
        clauses.append("ts >= ?")
        params.append(since_ts)
    if not clauses:
        return []
    params.append(limit)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    try:
        return conn.execute(
            "SELECT ts, sender, sender_name, is_from_me, text FROM messages"
            f" WHERE {' AND '.join(clauses)} ORDER BY ts DESC LIMIT ?",
            params,
        ).fetchall()
    finally:
        conn.close()


__all__ = [
    "mirror_path", "build_mirror", "search_mirror",
    "load_handle_names", "decode_attributed_body",
]
