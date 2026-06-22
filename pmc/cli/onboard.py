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

    # Probe whether FDA is already granted (read-only chat.db / History.db).
    fda_ok, parent_app = _check_fda_with_parent()

    if fda_ok:
        ui.say(console, f"{ui.GLYPH_DONE} Full Disk Access already granted",
               style=ui.OK)
    else:
        ui.say(console,
               f"Full Disk Access is not granted to {parent_app}.",
               style=ui.WARN)
        ui.say_dim(console,
                   f"pmc reads system databases that require this permission. "
                   f"without it, most extractors will return empty.")
        console.print()
        if yes or Confirm.ask(
            f"[dim]open System Settings → Full Disk Access for you? "
            f"(you'll need to drag {parent_app} into the list)[/]",
            default=True,
        ):
            _open_fda_pane()
            console.print()
            ui.say(console,
                   f"In the window that just opened:",
                   style=ui.ACCENT)
            ui.say_dim(console,
                       f"  1. click the + button at the bottom of the FDA list")
            ui.say_dim(console,
                       f"  2. add {parent_app}  (often under Applications "
                       f"or Applications/Utilities)")
            ui.say_dim(console,
                       f"  3. toggle the switch next to it to ON")
            ui.say_dim(console,
                       f"  4. you may be prompted to quit & relaunch "
                       f"{parent_app} — do that, then re-run "
                       f"`pmc onboard`")
            console.print()
            if not yes and not Confirm.ask(
                "[dim]done? proceed with extraction now?[/]",
                default=True,
            ):
                ui.say_dim(console, "(paused — run `pmc onboard` again when ready)")
                return True
        else:
            ui.say_dim(console, "(skipped — run `pmc connect` later)")
            return True

    console.print()
    if not yes and not Confirm.ask(
        "[dim]ready to extract? (~3 minutes)[/]",
        default=True,
    ):
        ui.say_dim(console, "(skipped — run `pmc connect` later)")
        return True
    rc = subprocess.call(["pmc", "connect"])
    return rc in (0, None)


def _open_fda_pane() -> None:
    """Pop open System Settings → Privacy & Security → Full Disk Access.
    Uses the macOS x-apple.systempreferences: URL scheme."""
    try:
        subprocess.run(
            ["open",
             "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"],
            check=False, timeout=3,
        )
    except (subprocess.SubprocessError, OSError):
        pass


def _check_fda_with_parent() -> tuple[bool, str]:
    """Return (granted, parent_app_name). parent_app_name is the
    .app the user needs to drag into FDA if it's not granted yet."""
    import os
    import sqlite3
    from pathlib import Path

    home = Path.home()
    chat_db = home / "Library/Messages/chat.db"
    history_db = home / "Library/Safari/History.db"
    target = chat_db if chat_db.is_file() else (
        history_db if history_db.is_file() else None
    )
    granted = True
    if target is not None:
        try:
            conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=2.0)
            try:
                conn.execute("SELECT 1").fetchone()
            finally:
                conn.close()
        except (sqlite3.OperationalError, PermissionError):
            granted = False

    # Walk up the process tree until we find an `.app` ancestor.
    # Direct parent is usually the shell (zsh/bash), which isn't what
    # the user needs to add to FDA — Terminal.app / iTerm.app / Cursor.app
    # etc. is what needs the grant.
    parent = "your terminal app"
    try:
        pid = os.getppid()
        for _ in range(10):  # cap the walk
            # `comm` truncates on macOS; `args` (full command) doesn't.
            # First field is the executable, then the parent PID we
            # get from a second `ps` call to avoid parsing whitespace
            # inside args.
            args_res = subprocess.run(
                ["ps", "-ww", "-o", "args=", "-p", str(pid)],
                capture_output=True, text=True, timeout=2,
            )
            ppid_res = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                capture_output=True, text=True, timeout=2,
            )
            if args_res.returncode != 0 or ppid_res.returncode != 0:
                break
            cmdline = args_res.stdout.strip()
            ppid_str = ppid_res.stdout.strip()
            if not cmdline:
                break
            if ".app/" in cmdline:
                # e.g. /Applications/iTerm.app/Contents/MacOS/iTerm2 -psn_...
                app = cmdline.split(".app/")[0].split("/")[-1] + ".app"
                parent = app
                break
            try:
                pid = int(ppid_str)
            except ValueError:
                break
            if pid <= 1:
                break
    except (subprocess.SubprocessError, OSError):
        pass
    return granted, parent


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
