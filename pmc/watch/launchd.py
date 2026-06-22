"""launchd integration — install/uninstall the user-level LaunchAgent
that keeps `pmc watch` running across logins and crashes.

LaunchAgent plist lives at:
    ~/Library/LaunchAgents/com.thepersonalmodelcompany.watch.plist

Logs go to:
    ~/.pmc/watch.log         (stdout)
    ~/.pmc/watch.err         (stderr)

The plist runs `<pmc-binary> watch --log-file` so logs are written to
the file path the daemon already supports.

Install:    pmc watch install
            (writes plist, runs `launchctl load -w` on it)
Uninstall:  pmc watch uninstall
            (runs `launchctl unload -w` and removes the plist)
Status:     pmc watch status
            (queries `launchctl list` for our label)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from pmc.cli import ui
from pmc.cli.local_config import CONFIG_DIR


LABEL = "com.thepersonalmodelcompany.watch"
PLIST_DIR = Path(os.path.expanduser("~/Library/LaunchAgents"))
PLIST_FILE = PLIST_DIR / f"{LABEL}.plist"

STDOUT_LOG = CONFIG_DIR / "watch.log"
STDERR_LOG = CONFIG_DIR / "watch.err"


def _pmc_binary() -> str:
    """Find the pmc CLI on PATH or fall back to the venv binary in
    the repo. The plist needs an absolute path."""
    found = shutil.which("pmc")
    if found:
        return found
    # Dev mode: the venv binary inside the repo.
    repo = Path(__file__).resolve().parents[2]
    venv = repo / ".venv/bin/pmc"
    if venv.is_file():
        return str(venv)
    # Last resort — let the user see the error when they run it.
    return "pmc"


def _plist_contents(pmc: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>{pmc}</string>
    <string>watch</string>
    <string>--log-file</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>ProcessType</key>
  <string>Background</string>

  <key>Nice</key>
  <integer>5</integer>

  <key>StandardOutPath</key>
  <string>{STDOUT_LOG}</string>

  <key>StandardErrorPath</key>
  <string>{STDERR_LOG}</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
"""


def cmd_install(args: argparse.Namespace) -> int:
    console = Console()
    PLIST_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    pmc = _pmc_binary()
    plist = _plist_contents(pmc)
    PLIST_FILE.write_text(plist)
    ui.say(console, f"{ui.GLYPH_DONE} wrote {PLIST_FILE}", style=ui.OK)

    # Unload first in case we're upgrading; then load.
    subprocess.run(["launchctl", "unload", str(PLIST_FILE)],
                   capture_output=True)
    rc = subprocess.call(["launchctl", "load", "-w", str(PLIST_FILE)])
    if rc != 0:
        ui.say(console, f"launchctl load failed (rc={rc})", style=ui.WARN)
        return rc
    ui.say(console, f"{ui.GLYPH_DONE} launchd loaded the agent", style=ui.OK)
    ui.say_dim(console, f"logs:  {STDOUT_LOG}")
    ui.say_dim(console, "pmc watch is now running and will restart on crash + login.")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    console = Console()
    if not PLIST_FILE.is_file():
        ui.say_dim(console, "(no plist installed; nothing to do)")
        return 0
    subprocess.run(["launchctl", "unload", "-w", str(PLIST_FILE)],
                   capture_output=True)
    PLIST_FILE.unlink(missing_ok=True)
    ui.say(console, f"{ui.GLYPH_DONE} uninstalled {PLIST_FILE}", style=ui.OK)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    if not PLIST_FILE.is_file():
        ui.say_dim(console, "(not installed — run pmc watch install)")
        return 1
    res = subprocess.run(["launchctl", "list", LABEL], capture_output=True, text=True)
    if res.returncode != 0:
        ui.say(console, "agent not loaded — try pmc watch install again",
               style=ui.WARN)
        return 1
    # launchctl list prints a plist-y dict; just show the PID line.
    pid_line = next((l for l in res.stdout.splitlines() if "PID" in l), "").strip()
    ui.say(console, f"{ui.GLYPH_DONE} loaded:  {pid_line or '(running)'}",
           style=ui.OK)
    ui.say_dim(console, f"logs:  {STDOUT_LOG}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Mount install/uninstall/status as subcommands under `pmc watch ...`.
    `pmc watch` alone keeps running the daemon in the foreground."""
    # Note: these are *separate* top-level commands rather than nested
    # under `pmc watch` because argparse subparser-on-subparser is
    # painful. We use `pmc watch-install` / `pmc watch-status` / etc.
    install = sub.add_parser(
        "watch-install",
        help="Install the launchd agent — pmc watch runs at login and "
        "restarts on crash",
    )
    install.set_defaults(func=cmd_install)

    uninstall = sub.add_parser(
        "watch-uninstall",
        help="Remove the launchd agent",
    )
    uninstall.set_defaults(func=cmd_uninstall)

    status = sub.add_parser(
        "watch-status",
        help="Show whether the launchd agent is loaded",
    )
    status.set_defaults(func=cmd_status)


__all__ = ["cmd_install", "cmd_uninstall", "cmd_status", "register"]
