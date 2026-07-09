"""PMC's MCP server — the bridge between your personal substrate and
whatever agent you're using.

`pmc serve --mcp` starts this. The agent (Claude Desktop, Cursor,
Continue, anything that speaks MCP) launches it as a subprocess,
opens a stdio JSON-RPC channel, and gets a fixed tool surface that
maps to agent QUESTIONS, not graph queries:

  whoami()              → 500-token bootstrap packet (auto-call on connect)
  self()                → full self.md
  whats_active()        → currently in motion (projects, threads)
  time_today()          → today's reconstructed activity
  time_typical_day()    → the 24h shape
  time_week()           → weekly rhythm
  time_month()          → monthly aggregate
  recent_reading()      → topic clusters from browser content
  recent_activity()     → last 7 days reconstructed
  search_messages(q)    → full-text iMessage search
  find_thread_with(p)   → message thread with a specific person

Every tool returns interpretation + evidence. The agent doesn't have
to do synthesis work — PMC already did it.

This server is READ-ONLY. It does not write to the graph. It cannot
take actions on the user's behalf. The host agent decides what to do
with the substrate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

from pmc.cli.local_config import LocalConfig, load


log = logging.getLogger("pmc.mcp")


# ---------------------------------------------------------------------------
# Server boot
# ---------------------------------------------------------------------------


def serve_blocking() -> None:
    """Synchronous wrapper that runs the asyncio MCP server until the
    connecting agent disconnects."""
    cfg = load()
    if cfg is None:
        # MCP servers shouldn't print to stdout (it's the protocol channel).
        # Use stderr.
        import sys
        sys.stderr.write("pmc-mcp: no config (run `pmc configure` first)\n")
        sys.exit(1)
    asyncio.run(_serve(cfg))


async def _serve(cfg: LocalConfig) -> None:
    server = Server("pmc")
    _register_tools(server, cfg)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="pmc",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def _register_tools(server: Server, cfg: LocalConfig) -> None:
    """Define every tool the connecting agent will see."""

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="whoami",
                description=(
                    "Get a concise (~500 token) packet describing who the "
                    "user is right now: what they're building, where their "
                    "time goes, what's currently in motion. ALWAYS call "
                    "this first when starting a conversation. For the "
                    "deeper structured portrait, call `picture` after this."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="self",
                description=(
                    "Get the user's full self portrait (self.md) — "
                    "characterized prose covering who they are at the level "
                    "of work, creation, attention, and rhythm. Longer than "
                    "whoami; use when you need a deeper picture."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="whats_active",
                description=(
                    "What the user is currently working on: active "
                    "projects/repos with status, language, recent commits."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="time_today",
                description=(
                    "What the user has done TODAY so far: commits, messages, "
                    "photos, with active-hours window. Compared to typical."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="time_typical_day",
                description=(
                    "The user's typical 24-hour day shape — when they "
                    "code, when they read, when they message, when they "
                    "sleep — aggregated from the last 30 days."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="time_week",
                description=(
                    "Weekly rhythm: hours per day of week, weekday vs "
                    "weekend averages."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="time_month",
                description=(
                    "Last 30 days: total Mac time, breakdown by category "
                    "(coding/reading/comms/etc.), top apps, web subcategories."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="picture",
                description=(
                    "The canonical portrait of the user. Top-level: "
                    "'here is how the user seems to spend their time' + "
                    "drilled-down sections (work, reading, attention) + "
                    "what's in motion + patterns an agent could learn. "
                    "This is the richest single source — read this when "
                    "you need a deep, structured picture of the user."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="agenda",
                description=(
                    "Concrete pieces of work the user has on their plate "
                    "right now (actionable) + repeating patterns an agent "
                    "could learn to do on their behalf (learnable). Use "
                    "to know what to help with proactively."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="recent_reading",
                description=(
                    "What the user has been reading and thinking about, "
                    "based on the actual content of pages they visited in "
                    "the last 7 days. Returns topic clusters with 1-2 "
                    "sentence summaries — use this to know what they're "
                    "currently learning or working through."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="recent_activity",
                description=(
                    "Last 7 days reconstructed from precise timestamps: "
                    "commits per day per repo, messages per day, photos. "
                    "Use to know 'what was yesterday like' / 'what was the "
                    "big day this week'."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="search_messages",
                description=(
                    "Search the user's iMessages. Filter by text "
                    "substring (query), by conversation partner "
                    "(person — contact name or phone/email; matches "
                    "the whole conversation including messages the "
                    "user sent), and/or by date (since, YYYY-MM-DD). "
                    "For 'what did X and I talk about', use person "
                    "alone — the text rarely contains their name. "
                    "Use sparingly — this reads the user's private "
                    "comms."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "text substring to match "
                                           "(optional if person given)"
                        },
                        "person": {
                            "type": "string",
                            "description": "conversation partner — "
                                           "contact name or handle"
                        },
                        "since": {
                            "type": "string",
                            "description": "only messages on/after "
                                           "this date (YYYY-MM-DD)"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "max results (1-50)"
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, args: dict[str, Any]) -> list[types.TextContent]:
        try:
            text = await _dispatch(cfg, name, args)
        except Exception as e:  # noqa: BLE001
            text = f"(error in {name}: {e})"
        return [types.TextContent(type="text", text=text)]


# ---------------------------------------------------------------------------
# Tool dispatch — each tool reads from the portrait / graph
# ---------------------------------------------------------------------------


async def _dispatch(cfg: LocalConfig, name: str, args: dict[str, Any]) -> str:
    if name == "whoami":
        return _read_portrait(cfg, "whoami.txt")
    if name == "self":
        return _strip_md_header(_read_portrait(cfg, "self.md"))
    if name == "whats_active":
        return _whats_active(cfg)
    if name == "time_today":
        return _time_today(cfg)
    if name == "time_typical_day":
        return _strip_md_header(_read_portrait(cfg, "time_day.md"))
    if name == "time_week":
        return _strip_md_header(_read_portrait(cfg, "time_week.md"))
    if name == "time_month":
        return _strip_md_header(_read_portrait(cfg, "time_month.md"))
    if name == "picture":
        return _strip_md_header(_read_portrait(cfg, "picture.md"))
    if name == "agenda":
        return _strip_md_header(_read_portrait(cfg, "agenda.md"))
    if name == "recent_reading":
        return _strip_md_header(_read_portrait(cfg, "reading.md"))
    if name == "recent_activity":
        return _strip_md_header(_read_portrait(cfg, "time_recent.md"))
    if name == "search_messages":
        return _search_messages(
            args.get("query") or "",
            int(args.get("limit") or 10),
            person=args.get("person") or None,
            since=args.get("since") or None,
        )
    return f"(unknown tool: {name})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _portrait_dir(cfg: LocalConfig) -> Path:
    root = cfg.effective_storage_root()
    return (
        root / "users" / (cfg.user_id or "local")
        / "graph" / "synth" / "portrait"
    )


def _read_portrait(cfg: LocalConfig, name: str) -> str:
    p = _portrait_dir(cfg) / name
    if not p.is_file():
        return (
            f"(no {name} yet — run `pmc consolidate` to build the "
            "portrait before using this tool)"
        )
    return p.read_text()


def _strip_md_header(text: str) -> str:
    """Remove our own `<!-- generated by pmc consolidate -->` marker."""
    out_lines: list[str] = []
    skipping = True
    for line in text.splitlines():
        if skipping and (line.strip().startswith("<!--") or line.strip() == ""):
            continue
        skipping = False
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _whats_active(cfg: LocalConfig) -> str:
    """Read work.jsonl and return active projects."""
    work_path = _portrait_dir(cfg) / "work.jsonl"
    if not work_path.is_file():
        return "(no work portrait yet — run `pmc consolidate`)"
    lines: list[str] = []
    for ln in work_path.open():
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        status = d.get("status") or ""
        lines.append(
            f"## {d.get('name','?')}  ({status})\n"
            f"{d.get('one_liner','')}\n\n"
            f"{d.get('body','')}"
        )
    return "\n\n".join(lines) if lines else "(no active work)"


def _time_today(cfg: LocalConfig) -> str:
    """Read time_recent.md and isolate today + provide context."""
    recent = _read_portrait(cfg, "time_recent.md")
    today_key = datetime.now(timezone.utc).astimezone().date().isoformat()
    today_lines = [
        ln for ln in recent.splitlines() if today_key in ln
    ]
    if not today_lines:
        return (
            "no precise activity recorded for today yet.\n\n"
            "context — last 7 days:\n"
            + _strip_md_header(recent)
        )
    return (
        f"today ({today_key}):\n"
        + "\n".join(today_lines)
        + "\n\nlast 7 days for context:\n"
        + _strip_md_header(recent)
    )


def _search_messages(query: str, limit: int,
                     person: str | None = None,
                     since: str | None = None) -> str:
    """Search the user-owned message mirror (built by `pmc connect`).

    The mirror avoids needing Full Disk Access in the MCP host app —
    TCC attributes this process's file reads to the interpreter, not
    the host, so live chat.db reads fail even when the host has FDA.
    Falls back to live chat.db for setups that predate the mirror.
    """
    if not query.strip() and not (person or "").strip():
        return "(give a text query, a person, or both)"
    limit = max(1, min(50, limit))

    since_ts: float | None = None
    if since:
        try:
            since_ts = datetime.strptime(since, "%Y-%m-%d").astimezone().timestamp()
        except ValueError:
            return f"(bad since date {since!r} — use YYYY-MM-DD)"

    from pmc.cli.local_config import load
    from pmc.storage.message_mirror import mirror_path, search_mirror

    cfg = load()
    if cfg is not None:
        mirror = mirror_path(cfg.effective_storage_root(),
                             cfg.user_id or "local")
        if mirror.is_file():
            try:
                rows = search_mirror(mirror, query, limit,
                                     person=person, since_ts=since_ts)
            except sqlite3.Error as e:
                return f"(message mirror unreadable: {e})"
            hint = ""
            if not rows and (person or since):
                mtime = datetime.fromtimestamp(
                    mirror.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                hint = (f" — note: the mirror was last refreshed "
                        f"{mtime}; newer messages need `pmc connect`")
            return _format_message_rows(rows, query or person or "", hint)

    # No mirror yet — try live chat.db (works only if this process's
    # TCC identity has FDA, which is rare inside MCP hosts).
    db = Path("~/Library/Messages/chat.db").expanduser()
    rebuild_fix = (
        "No message mirror exists yet and chat.db isn't readable from "
        "this process. Tell the user to run `pmc connect` in an "
        "FDA-granted terminal — it builds a local, user-owned message "
        "mirror that this tool searches without needing Full Disk "
        "Access in the host app."
    )
    if not db.is_file():
        return f"({rebuild_fix})"
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return f"({rebuild_fix})"
    try:
        epoch_offset = 978307200
        rows = [
            (
                (date / 1e9 if date and date > 1e12 else date or 0)
                + epoch_offset,
                handle_id, None, is_from_me, text,
            )
            for text, is_from_me, date, handle_id in conn.execute("""
                SELECT m.text, m.is_from_me, m.date, h.id
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.text LIKE ?
                  AND m.text IS NOT NULL
                ORDER BY m.date DESC
                LIMIT ?
            """, (f"%{query.strip()}%", limit))
            if text is not None
        ]
    finally:
        conn.close()
    note = (" — note: person/since filters need the message mirror; "
            "run `pmc connect`") if (person or since) else ""
    return _format_message_rows(rows, query, note)


def _format_message_rows(rows: list[tuple], query: str,
                         hint: str = "") -> str:
    epoch_offset = 978307200
    out_lines: list[str] = []
    for ts, sender, sender_name, is_from_me, text in rows:
        dt = (datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
              .isoformat(timespec="seconds")) if ts > epoch_offset else "?"
        if is_from_me:
            who = "you"
        else:
            who = sender_name or sender or "(unknown)"
        text_clean = text.replace("\n", " ").strip()[:240]
        out_lines.append(f"[{dt}] {who}: {text_clean}")
    if not out_lines:
        return f"(no messages matching {query!r}{hint})"
    return "\n\n".join(out_lines)


__all__ = ["serve_blocking"]
