"""`pmc onboard` — chained first-time setup.

Walks a fresh user through every step in the right order, with the
right confirmations:

   doctor       → preflight checks
   configure    → provider + API key (interactive)
   connect      → extract data (warns about FDA)
   sandbox      → build portrait (warns about ~$0.20 API cost)
   install-mcp  → plug into Claude / Cursor / Continue

Each step is gated by a confirm so the user knows what's about to
happen and can bail. Steps that already succeeded are skipped on
re-runs.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from pmc.cli import ui
from pmc.cli.local_config import CONFIG_FILE, load


def cmd_onboard(args: argparse.Namespace) -> int:
    console = Console()
    if args.clear:
        console.clear()
    ui.banner_top(
        console,
        title="welcome to pmc",
        subtitle="setup is five steps. each one will tell you what it's doing.",
    )

    # 1. preflight diagnostic
    if not _step_doctor(console, args.yes):
        return 1

    # 2. provider + API key
    if not _step_configure(console, args.yes):
        return 2

    # 3. extract data
    if not _step_connect(console, args.yes):
        return 3

    # 4. build the portrait
    if not _step_sandbox(console, args.yes):
        return 4

    # 5. plug into an agent
    if not _step_install_mcp(console, args.yes):
        return 5

    # done
    console.print()
    ui.banner_top(
        console,
        title="you're done.",
        subtitle="restart your agent and ask: 'what do you know about me?'",
    )
    return 0


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _step_doctor(console: Console, yes: bool) -> bool:
    ui.card_title(console, "1 of 5  ·  preflight checks")
    ui.say_dim(console,
               "we'll verify uv, the Rust extractor binary, your API key, "
               "Full Disk Access, and disk space.")
    if not yes and not Confirm.ask("\n[dim]run pmc doctor?[/]", default=True):
        ui.say_dim(console, "(skipped)")
        return True
    rc = subprocess.call(["pmc", "doctor", "--no-clear"])
    if rc != 0:
        console.print()
        ui.say(console,
               "doctor reported errors. fix the ones marked with × above, "
               "then re-run `pmc onboard`.",
               style=ui.WARN)
        return False
    return True


def _step_configure(console: Console, yes: bool) -> bool:
    console.print()
    ui.card_title(console, "2 of 5  ·  provider + API key")
    cfg = load()
    if cfg and cfg.api_key:
        ui.say(console,
               f"already configured: {cfg.provider}/{cfg.model}",
               style=ui.OK)
        if not yes and not Confirm.ask(
            "[dim]reconfigure (pick a different provider or replace the key)?[/]",
            default=False,
        ):
            return True
    ui.say_dim(console,
               "you'll need an API key from anthropic, openai, google, or "
               "openrouter. paste it when asked — input is hidden.")
    if not yes and not Confirm.ask("\n[dim]run pmc configure?[/]", default=True):
        ui.say_dim(console, "(skipped — but pmc chat / sandbox will need a key)")
        return True
    rc = subprocess.call(["pmc", "configure"])
    return rc == 0


def _step_connect(console: Console, yes: bool) -> bool:
    console.print()
    ui.card_title(console, "3 of 5  ·  extract data from your Mac")
    ui.say_dim(console,
               "pmc will read your messages, mail, photos metadata, calendar, "
               "voice memos, browser history, code repos, and shell history. "
               "everything stays on your machine.")
    console.print()
    ui.say(console,
           "this needs Full Disk Access (FDA) on whichever terminal you're "
           "running pmc from.",
           style=ui.ACCENT)
    ui.say_dim(console,
               "fix: System Settings → Privacy & Security → Full Disk Access "
               "→ add Terminal.app (or iTerm, Cursor, etc.)")
    console.print()
    if not yes and not Confirm.ask(
        "[dim]have you granted FDA and are ready to extract? (~3 minutes)[/]",
        default=True,
    ):
        ui.say_dim(console, "(skipped — run `pmc connect` later when ready)")
        return True
    rc = subprocess.call(["pmc", "connect"])
    return rc in (0, None)


def _step_sandbox(console: Console, yes: bool) -> bool:
    console.print()
    ui.card_title(console, "4 of 5  ·  build the portrait")
    ui.say_dim(console,
               "pmc will run the consolidator: characterize your projects, "
               "fetch + read pages you visited, cluster topics, compose your "
               "self portrait, and produce an actionable agenda.")
    console.print()
    ui.say(console,
           "cost: about $0.10–0.30 in API calls (one-time per pass).",
           style=ui.ACCENT)
    ui.say_dim(console, "takes about 90 seconds.")
    console.print()
    if not yes and not Confirm.ask(
        "[dim]run pmc sandbox?[/]", default=True,
    ):
        ui.say_dim(console,
                   "(skipped — run `pmc sandbox` later. without it, the agent "
                   "won't have a structured portrait to read.)")
        return True
    rc = subprocess.call(["pmc", "sandbox", "--no-clear", "--keep-pages"])
    return rc == 0


def _step_install_mcp(console: Console, yes: bool) -> bool:
    console.print()
    ui.card_title(console, "5 of 5  ·  plug into your agent")
    ui.say_dim(console,
               "pmc exposes your portrait via MCP. any agent that speaks MCP "
               "(claude desktop, cursor, continue) can subscribe and "
               "instantly know you.")
    console.print()
    choices = ["claude", "cursor", "continue", "skip"]
    if yes:
        choice = "claude"
    else:
        choice = Prompt.ask(
            "[dim]which agent do you use most?[/]",
            choices=choices,
            default="claude",
        )
    if choice == "skip":
        ui.say_dim(console, "(skipped — run `pmc install-mcp <agent>` any time)")
        return True
    rc = subprocess.call(["pmc", "install-mcp", choice])
    return rc == 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "onboard",
        help="First-time setup — chained doctor → configure → connect → "
             "sandbox → install-mcp with confirmations at each step",
    )
    p.add_argument("-y", "--yes", action="store_true",
                   help="auto-confirm every step (no interactive prompts)")
    p.add_argument("--no-clear", dest="clear", action="store_false",
                   default=True, help="don't clear the screen")
    p.set_defaults(func=cmd_onboard)


__all__ = ["cmd_onboard", "register"]
