"""`pmc reading` — what you've been reading lately.

Shows the topic clusters extracted from your recent browser history,
read from `portrait/reading.md` (regenerated on each `pmc consolidate`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from pmc.cli import ui
from pmc.cli.local_config import load


def cmd_reading(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load()
    if cfg is None:
        ui.say(console, "no pmc config — run `pmc configure` first.",
               style=ui.WARN)
        return 1
    root = cfg.effective_storage_root()
    reading_md = (
        root / "users" / (cfg.user_id or "local")
        / "graph" / "synth" / "portrait" / "reading.md"
    )
    if not reading_md.is_file():
        ui.say(console,
               "no reading layer yet — run `pmc consolidate` to build it.",
               style=ui.WARN)
        return 1
    if args.clear:
        console.clear()
    text = reading_md.read_text()
    _render_reading(console, text)
    return 0


def _render_reading(console: Console, text: str) -> None:
    lines = text.splitlines()
    in_pages_block = False
    current_topic: str | None = None
    summary_lines: list[str] = []

    # First pass: parse into structured shape, then render nicely.
    topics: list[dict] = []
    cur: dict | None = None
    intro_lines: list[str] = []
    seen_first_topic = False
    for line in lines:
        if line.startswith("# "):
            continue  # title — we render our own banner
        if line.startswith("## "):
            if cur is not None:
                topics.append(cur)
            cur = {"label": line[3:].strip(), "summary": [], "pages": []}
            seen_first_topic = True
            in_pages_block = False
            continue
        if line.strip().lower().startswith("pages:"):
            in_pages_block = True
            continue
        if in_pages_block and line.strip().startswith("-"):
            cur["pages"].append(line.strip()[1:].strip())
            continue
        if not seen_first_topic:
            if line.strip():
                intro_lines.append(line.strip())
            continue
        if cur is not None and line.strip():
            cur["summary"].append(line.strip())
    if cur is not None:
        topics.append(cur)

    intro = " ".join(intro_lines).strip()
    ui.banner_top(
        console,
        title="reading",
        subtitle=intro or "what's been on your screen",
    )

    if not topics:
        ui.say_dim(console, "(no topics extracted yet)")
        return

    for i, t in enumerate(topics):
        ui.card_title(console, t["label"])
        if t["summary"]:
            ui.say(console, " ".join(t["summary"]))
        if t["pages"]:
            console.print()
            for u in t["pages"][:5]:
                ui.say_dim(console, "  → " + u)
            if len(t["pages"]) > 5:
                ui.say_dim(console, f"  → and {len(t['pages']) - 5} more")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "reading",
        help="What you've been reading — topic clusters from browser history",
    )
    p.add_argument("--no-clear", dest="clear", action="store_false",
                   default=True, help="don't clear the screen")
    p.set_defaults(func=cmd_reading)


__all__ = ["cmd_reading", "register"]
