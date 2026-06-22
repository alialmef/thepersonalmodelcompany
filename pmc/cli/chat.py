"""`pmc chat` — terminal REPL with your agent.

Visual surface designed in pmc/cli/ui.py: reading column, speaker
labels above messages, soft turn breaks, animated "reading" pause
before each reply. The terminal is the canvas; conversation lives
inside a fixed 72-char column with a 3-char left margin regardless
of terminal width.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.console import Console

from pmc.agent.prompts import TaskKind, compose
from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
)
from pmc.agent.providers.registry import get_provider
from pmc.cli import ui
from pmc.cli.context import build_context
from pmc.cli.local_config import CONFIG_DIR, LocalConfig, auto_pick_user_id
from pmc.storage.graph_store import GraphStore


HISTORY_TURNS_KEPT = 24
HISTORY_FILE = CONFIG_DIR / "chat_history"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    cfg: LocalConfig
    storage_root: Path
    user_id: str
    user_email: str
    history: list[Message]
    context_block: str
    console: Console

    @classmethod
    def open(cls, cfg: LocalConfig, console: Console) -> "Session":
        storage_root = cfg.effective_storage_root()
        configured = cfg.user_id or "local"
        if _user_graph_empty(storage_root, configured):
            discovered = auto_pick_user_id(storage_root)
            if discovered and discovered != configured:
                console.print()
                console.print(
                    f"{ui.margin()}[{ui.DIM}](auto-selected {discovered} — "
                    f"populated graph found on this machine)[/]"
                )
                user_id = discovered
            else:
                user_id = configured
        else:
            user_id = configured
        return cls(
            cfg=cfg,
            storage_root=storage_root,
            user_id=user_id,
            user_email=cfg.user_email or f"{user_id}@local",
            history=[],
            context_block=build_context(storage_root, user_id),
            console=console,
        )

    def reload_context(self) -> None:
        self.context_block = build_context(self.storage_root, self.user_id)

    def system_prompt(self, task: TaskKind = TaskKind.CHAT) -> str:
        base = compose(self.user_email, task)
        return base + "\n\n" + self.context_block


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(cfg: LocalConfig, *, skip_opener: bool = False, clear: bool = True) -> int:
    console = Console()
    if clear:
        console.clear()
    session = Session.open(cfg, console)
    _print_banner(session)
    try:
        asyncio.run(_loop(session, skip_opener=skip_opener))
    except KeyboardInterrupt:
        console.print()
        return 0
    return 0


async def _loop(session: Session, *, skip_opener: bool = False) -> None:
    provider = get_provider(session.cfg.provider)
    if provider is None:
        ui.say(session.console,
               f"unknown provider: {session.cfg.provider!r}", style=ui.WARN)
        return
    pcfg = ProviderConfig(
        provider=session.cfg.provider,
        model=session.cfg.model,
        api_key=session.cfg.api_key,
    )

    if not skip_opener:
        await _fire_opener(session, provider, pcfg)

    prompt_session = _build_prompt_session()
    prompt_prefix = ANSI(f"\x1b[2m{ui.margin()}\x1b[0m")

    while True:
        # Speaker label above the input line.
        ui.speaker(session.console, "you")
        try:
            line = await prompt_session.prompt_async(prompt_prefix)
        except (EOFError, KeyboardInterrupt):
            session.console.print()
            return
        if line is None:
            return
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.split()[0].lower()
            if cmd == "/intro":
                await _fire_opener(session, provider, pcfg)
                continue
            if _handle_slash(line, session):
                return
            continue

        session.history.append(Message(role="user", content=line))
        await _stream_reply(session, provider, pcfg)
        _trim_history(session)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def _stream_reply(
    session: Session,
    provider,
    pcfg: ProviderConfig,
    *,
    task: TaskKind = TaskKind.CHAT,
) -> Optional[str]:
    """One assistant turn. Pre-stream `⋯ reading` beat, then text streams
    into the reading column with word-boundary wrapping + left margin."""
    ui.speaker(session.console, "agent")
    await ui.pause(session.console, label="reading", duration_s=0.5)

    col = ui.StreamColumn(session.console)
    accumulator: list[str] = []
    try:
        async for chunk in provider.stream_chat(
            messages=list(session.history),
            config=pcfg,
            system=session.system_prompt(task),
            max_tokens=2048,
        ):
            accumulator.append(chunk)
            col.feed(chunk)
        col.close()
    except ProviderError as e:
        col.close()
        session.console.print()
        ui.say(session.console, f"[{e.kind}] {e}", style=ui.WARN)
        if session.history and session.history[-1].role == "user":
            session.history.pop()
        return None
    except Exception as e:  # noqa: BLE001
        col.close()
        session.console.print()
        ui.say(session.console, f"[error] {e}", style=ui.WARN)
        if session.history and session.history[-1].role == "user":
            session.history.pop()
        return None

    reply = "".join(accumulator).strip()
    if reply:
        session.history.append(Message(role="assistant", content=reply))
    return reply


async def _fire_opener(session: Session, provider, pcfg: ProviderConfig) -> None:
    seed = Message(
        role="user",
        content="(session opened — make your opening per the OPENER task spec)",
    )
    ui.speaker(session.console, "agent")
    await ui.pause(session.console, label="reading", duration_s=0.7)

    col = ui.StreamColumn(session.console)
    accumulator: list[str] = []
    try:
        async for chunk in provider.stream_chat(
            messages=[seed],
            config=pcfg,
            system=session.system_prompt(TaskKind.OPENER),
            max_tokens=512,
        ):
            accumulator.append(chunk)
            col.feed(chunk)
        col.close()
    except ProviderError as e:
        col.close()
        ui.say(session.console, f"[{e.kind}] {e}", style=ui.WARN)
        return
    except Exception as e:  # noqa: BLE001
        col.close()
        ui.say(session.console, f"[error] {e}", style=ui.WARN)
        return
    reply = "".join(accumulator).strip()
    if reply:
        session.history.append(Message(role="assistant", content=reply))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_graph_empty(storage_root: Path, user_id: str) -> bool:
    graph_dir = storage_root / "users" / user_id / "graph"
    if not graph_dir.is_dir():
        return True
    for fp in graph_dir.glob("*.jsonl"):
        try:
            if fp.stat().st_size > 0:
                return False
        except OSError:
            continue
    return True


def _print_banner(session: Session) -> None:
    counts = GraphStore(session.storage_root).counts(session.user_id)
    total = sum(counts.values())
    if total == 0:
        subtitle = "graph empty · run pmc connect"
    else:
        subtitle = f"{total:,} entities · {session.cfg.model}"
    ui.banner_top(session.console, title="pmc", subtitle=subtitle)


def _build_prompt_session() -> PromptSession:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(history=FileHistory(str(HISTORY_FILE)))


def _trim_history(session: Session) -> None:
    if len(session.history) <= HISTORY_TURNS_KEPT:
        return
    session.history = session.history[-HISTORY_TURNS_KEPT:]


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _handle_slash(line: str, session: Session) -> bool:
    cmd = line.split()[0].lower()
    c = session.console
    if cmd in ("/quit", "/exit", "/q"):
        return True
    if cmd == "/reset":
        session.history.clear()
        c.print(f"{ui.margin()}[{ui.DIM}](history cleared)[/]")
        return False
    if cmd == "/reload":
        session.reload_context()
        c.print(f"{ui.margin()}[{ui.DIM}](context reloaded)[/]")
        return False
    if cmd == "/context":
        _print_context(session)
        return False
    if cmd == "/threads":
        _print_threads(session)
        return False
    if cmd == "/patterns":
        _print_patterns(session)
        return False
    if cmd == "/drift":
        _print_drift(session)
        return False
    if cmd in ("/help", "/?"):
        _print_help(session)
        return False
    c.print(f"{ui.margin()}[{ui.WARN}]unknown command {cmd!r} — try /help.[/]")
    return False


def _print_help(session: Session) -> None:
    c = session.console
    ui.card_title(c, "commands")
    items = [
        ("/threads",  "live threads the agent thinks are in motion"),
        ("/patterns", "steady-state patterns of how you spend time"),
        ("/drift",    "what's different about now vs the recent past"),
        ("/context",  "the full block the agent sees as system context"),
        ("/intro",    "re-fire the opener turn"),
        ("/reload",   "re-read the graph from disk"),
        ("/reset",    "clear conversation history (keep context)"),
        ("/quit",     "exit"),
    ]
    for tag, body in items:
        ui.card_item(c, tag, body)


def _print_threads(session: Session) -> None:
    from pmc.synthesis import load_threads
    items = load_threads(session.storage_root, session.user_id)
    c = session.console
    ui.card_title(c, "live threads")
    if not items:
        ui.say_dim(c, "no threads synthesized yet")
        c.print()
        return
    for thr in items[:20]:
        urgency = (getattr(thr, "urgency", "") or "").replace("_", " ")
        kind = getattr(thr, "kind", "") or ""
        head = getattr(thr, "headline", "") or ""
        tag = "  ·  ".join(b for b in [urgency, kind] if b)
        ui.card_item(c, tag, head)


def _print_patterns(session: Session) -> None:
    from pmc.synthesis import load_patterns
    items = load_patterns(session.storage_root, session.user_id)
    c = session.console
    ui.card_title(c, "patterns — what your life keeps doing")
    if not items:
        ui.say_dim(c, "no patterns synthesized yet")
        c.print()
        return
    for p in items[:20]:
        cat = getattr(p, "category", "") or ""
        head = getattr(p, "headline", "") or ""
        ui.card_item(c, cat, head)


def _print_drift(session: Session) -> None:
    from pmc.synthesis import load_drift
    items = load_drift(session.storage_root, session.user_id)
    c = session.console
    ui.card_title(c, "drift — what's different about now")
    if not items:
        ui.say_dim(c, "no drift detected yet")
        c.print()
        return
    for d in items[:20]:
        cat = getattr(d, "category", "") or ""
        head = getattr(d, "headline", "") or ""
        ui.card_item(c, cat, head)


def _print_context(session: Session) -> None:
    c = session.console
    ui.card_title(c, "context")
    for line in session.context_block.splitlines():
        c.print(f"{ui.margin()}[{ui.DIM}]{line}[/]")
    c.print()


__all__ = ["run", "Session"]
