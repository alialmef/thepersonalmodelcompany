"""`pmc connect` — populate the personal graph from the CLI.

Mac app parity: calls the same Rust engine the Mac app does, with a
live-rendered checklist as the visible surface. Each extractor lights
up as it starts (◐, animated dots), then flips to ● with a count when
it finishes. A running total ticks at the bottom.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from pmc.cli import ui
from pmc.cli.local_config import CONFIG_FILE, load


# Binary discovery roots (in priority order).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEV_REL = _REPO_ROOT / "desktop/target/release/examples/pmc_ingest"
_DEV_DBG = _REPO_ROOT / "desktop/target/debug/examples/pmc_ingest"


# Human-readable labels for each source name the binary emits. Kept
# short so the checklist column stays narrow.
SOURCE_LABEL = {
    "contacts":         "contacts",
    "imessage_enrich":  "messages",
    "calendar":         "calendar",
    "photos":           "photos",
    "safari":           "safari",
    "call_history":     "calls",
    "music":            "music",
    "files":            "files",
    "mail_enrich":      "mail",
    "notes_enrich":     "notes",
    "reminders":        "reminders",
    "chrome":           "chrome",
    "screen_time":      "screen time",
    "shell":            "shell",
    "locations":        "locations",
    "editor_state":     "editor state",
    "notifications":    "notifications",
    "voice_memos":      "voice memos",
    "slack":            "slack",
    "bookmarks":        "bookmarks",
    "wallet":           "wallet",
    "photo_concepts":   "photo concepts",
    "icloud_drive":     "icloud drive",
    "synthesis":        "synthesis",
}


def cmd_ingest(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load()
    if cfg is None:
        console.print()
        ui.say(console, f"no config at {CONFIG_FILE}. run pmc configure first.",
               style=ui.WARN)
        return 1

    # Preflight FDA check — fail fast with a clear message instead of
    # silently failing inside the extractor.
    fda_ok, fda_msg = _check_fda()
    if not fda_ok:
        console.print()
        ui.say(console, "Full Disk Access is not granted to this terminal.",
               style=ui.WARN)
        console.print()
        ui.say(console, fda_msg)
        console.print()
        ui.say_dim(console, "without FDA, pmc connect can only read a small "
                            "fraction of your data (no iMessage, no Mail, no "
                            "Safari history, no Photos metadata).")
        ui.say_dim(console, "pass --skip-fda-check to continue anyway.")
        if not getattr(args, "skip_fda_check", False):
            return 3

    binary = _find_or_build(console, allow_build=not args.no_build)
    if binary is None:
        return 2

    user_id = (args.user or cfg.user_id or "local")
    storage = cfg.effective_storage_root()

    if args.json:
        # Passthrough mode for scripts.
        cmd = [str(binary), "--user", user_id, "--root", str(storage), "--json"]
        return subprocess.call(cmd)

    if args.clear:
        console.clear()

    ui.banner_top(console, title="reading your mac",
                  subtitle=f"user {user_id[:8]}…")

    cmd = [str(binary), "--user", user_id, "--root", str(storage), "--json"]
    try:
        rc = _run_with_checklist(console, cmd)
    except KeyboardInterrupt:
        console.print()
        ui.say_dim(console, "(cancelled)")
        return 130

    if rc == 0:
        console.print()
        ui.say(console, "you're in the graph.", style=ui.ACCENT)
        ui.say_dim(console, "run pmc chat to talk to your agent.")
        console.print()
    return rc


# ---------------------------------------------------------------------------
# Live checklist renderer
# ---------------------------------------------------------------------------


def _run_with_checklist(console: Console, cmd: list[str]) -> int:
    """Spawn pmc-ingest --json, parse phase events line-by-line, drive a
    live-updating checklist of source rows."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # line-buffered
    )

    state: dict[str, dict] = {}   # source -> {status, count, ms, error}
    order: list[str] = []         # display order, first-seen
    started_at = time.time()
    spinner_idx = [0]
    done = threading.Event()

    def render() -> Group:
        rows: list[Text] = []
        total_entities = 0
        for src in order:
            entry = state[src]
            status = entry["status"]
            label = SOURCE_LABEL.get(src, src)

            if status == "running":
                glyph = ui.SPIN_FRAMES[spinner_idx[0] % len(ui.SPIN_FRAMES)]
                glyph_style = ui.ACCENT
                suffix = entry.get("activity", "")
            elif status == "done":
                glyph = ui.GLYPH_DONE
                glyph_style = ui.ACCENT
                if entry.get("skipped"):
                    glyph = ui.GLYPH_SKIPPED
                    glyph_style = ui.DIM
                    suffix = f"skipped — {entry.get('skip_reason', '')}"
                else:
                    n = entry.get("entities", 0)
                    ms = entry.get("ms", 0)
                    total_entities += n
                    suffix = f"{n:,} entities  ·  {_fmt_ms(ms)}"
            elif status == "error":
                glyph = ui.GLYPH_ERROR
                glyph_style = ui.WARN
                suffix = entry.get("error", "")
            else:
                glyph = ui.GLYPH_PENDING
                glyph_style = ui.DIM
                suffix = ""

            t = Text(ui.margin(), style="")
            t.append(glyph, style=glyph_style)
            t.append("  ")
            t.append(f"{label:<14}", style=ui.WHITE if status == "done"
                     else ui.DIM)
            if suffix:
                t.append("  ")
                t.append(suffix, style=ui.DIM)
            rows.append(t)

        # Footer
        elapsed = time.time() - started_at
        footer = Text()
        footer.append(ui.margin(), style="")
        footer.append("─" * 4, style=ui.DIM)
        footer.append("\n", style="")
        footer.append(ui.margin(), style="")
        footer.append(
            f"so far: {total_entities:,} entities  ·  "
            f"{_fmt_elapsed(elapsed)} elapsed",
            style=ui.DIM,
        )

        return Group(*rows, Text(""), footer)

    # Spinner animator — keeps the running rows ticking even when no
    # events arrive (binary can stall on a slow extractor).
    def animator(live: Live) -> None:
        while not done.is_set():
            spinner_idx[0] += 1
            live.update(render())
            time.sleep(0.1)

    with Live(render(), console=console, refresh_per_second=10,
              transient=False) as live:
        animator_thread = threading.Thread(target=animator, args=(live,),
                                           daemon=True)
        animator_thread.start()

        assert proc.stdout is not None
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue
            src = evt.get("source", "")
            phase = evt.get("phase", "")
            if not src:
                continue
            if src not in state:
                state[src] = {"status": "pending"}
                order.append(src)
            if phase == "start":
                state[src]["status"] = "running"
            elif phase == "done":
                state[src]["status"] = "done"
                state[src]["entities"] = evt.get("entities_written", 0)
                state[src]["ms"] = evt.get("duration_ms", 0)
                state[src]["skipped"] = evt.get("skipped", False)
                state[src]["skip_reason"] = evt.get("skip_reason") or ""
            elif phase == "error":
                state[src]["status"] = "error"
                state[src]["error"] = evt.get("error", "")
            live.update(render())

        done.set()
        animator_thread.join(timeout=0.5)
        live.update(render())

    rc = proc.wait()
    return rc


def _fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m = s / 60
    return f"{m:.1f}m"


def _check_fda() -> tuple[bool, str]:
    """Returns (ok, remediation_message). True if FDA appears to be
    granted to the current process (we can read chat.db read-only)."""
    import sqlite3
    home = Path.home()
    chat_db = home / "Library/Messages/chat.db"
    history_db = home / "Library/Safari/History.db"
    target = None
    if chat_db.is_file():
        target = chat_db
    elif history_db.is_file():
        target = history_db
    if target is None:
        # Nothing to probe — treat as OK (the user might be on a fresh
        # account that doesn't have either set up).
        return True, ""
    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=2.0)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        return True, ""
    except (sqlite3.OperationalError, PermissionError):
        return False, _fda_remediation()


def _fda_remediation() -> str:
    """Try to identify which terminal/parent process is running, so the
    fix message can name the exact app the user needs to add."""
    from pmc.cli.onboard import _check_fda_with_parent
    _, parent_app = _check_fda_with_parent()
    return (
        f"Fix: open System Settings → Privacy & Security → "
        f"Full Disk Access → click `+` → add {parent_app}, toggle it ON, "
        f"then re-run `pmc connect`. (run `pmc onboard` to have pmc open "
        f"that settings page for you.)"
    )


def _fmt_elapsed(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Binary discovery / build
# ---------------------------------------------------------------------------


def _find_or_build(console: Console, *, allow_build: bool) -> Optional[Path]:
    explicit = os.environ.get("PMC_INGEST_BIN")
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        ui.say(console, f"PMC_INGEST_BIN={explicit} but no file there.",
               style=ui.WARN)

    on_path = shutil.which("pmc-ingest")
    if on_path:
        return Path(on_path)

    if _DEV_REL.is_file():
        return _DEV_REL
    if _DEV_DBG.is_file():
        return _DEV_DBG

    if not allow_build:
        ui.say(console, "pmc-ingest not found and --no-build is set.",
               style=ui.WARN)
        return None

    return _build(console)


def _build(console: Console) -> Optional[Path]:
    if not shutil.which("cargo"):
        ui.say(console,
               "cargo not found — install Rust to build pmc-ingest, "
               "or set PMC_INGEST_BIN to a prebuilt binary.",
               style=ui.WARN)
        ui.say_dim(console,
                   "install rust:  curl --proto '=https' --tlsv1.2 -sSf "
                   "https://sh.rustup.rs | sh")
        return None
    crate_dir = _REPO_ROOT / "desktop"
    if not crate_dir.is_dir():
        ui.say(console, f"desktop/ crate not found at {crate_dir}.",
               style=ui.WARN)
        return None
    console.print()
    ui.say_dim(console, "(first run — building the engine. ~30s.)")
    with console.status(f"[{ui.DIM}]cargo build --example pmc_ingest --release…[/]"):
        rc = subprocess.call(
            ["cargo", "build", "--example", "pmc_ingest", "--release"],
            cwd=str(crate_dir),
        )
    if rc != 0:
        ui.say(console, f"build failed (exit {rc}).", style=ui.WARN)
        return None
    if _DEV_REL.is_file():
        ui.say(console, "engine built.", style=ui.OK)
        return _DEV_REL
    return None


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "connect",
        help="Connect your data sources — populate the graph "
        "(same engine the Mac app uses)",
    )
    p.add_argument("--user", help="user_id (default: from ~/.pmc/agent.json)")
    p.add_argument("--json", action="store_true",
                   help="pass through machine-readable JSON output")
    p.add_argument("--no-build", action="store_true",
                   help="don't try to build the engine if it isn't found")
    p.add_argument("--no-clear", dest="clear", action="store_false",
                   default=True,
                   help="don't clear the screen on start")
    p.add_argument("--skip-fda-check", action="store_true",
                   help="ignore the Full Disk Access preflight and try "
                        "to ingest anyway (most extractors will silently skip)")
    p.set_defaults(func=cmd_ingest)


__all__ = ["cmd_ingest", "register"]
