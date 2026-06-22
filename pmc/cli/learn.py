"""`pmc remember` / `pmc forget` — teach the gate what matters.

Both commands record one feedback row in `~/.pmc/gate-feedback.jsonl`
and refresh the in-memory overlay so the next event sees the new rule.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from rich.console import Console

from pmc.cli import ui
from pmc.watch.classifier import learn, rules


def cmd_remember(args: argparse.Namespace) -> int:
    return _record(args, kind="remember", weight=+1.0)


def cmd_forget(args: argparse.Namespace) -> int:
    return _record(args, kind="forget", weight=-1.0)


def _record(args: argparse.Namespace, *, kind: str, weight: float) -> int:
    console = Console()
    target = args.target.strip()
    if not target:
        ui.say(console, "no target — usage: pmc remember <path-or-pattern>",
               style=ui.WARN)
        return 1
    # Expand ~ and resolve absolute path for paths.
    if "/" in target or target.startswith("."):
        expanded = str(Path(os.path.expanduser(target)).resolve())
        target_type = "path"
        target = expanded
    else:
        target_type = "person" if " " in target else "theme"
    fb = learn.Feedback(
        ts=time.time(),
        kind=kind,
        target_type=target_type,
        target=target,
        weight=weight,
        note=args.note or "",
    )
    learn.record(fb)
    rules.refresh_overlay()
    console.print()
    ui.say(console,
           f"{ui.GLYPH_DONE} {kind}ed  ({target_type}: {target})",
           style=ui.OK)
    ui.say_dim(console,
               "saved to ~/.pmc/gate-feedback.jsonl — pmc watch will respect "
               "this on its next event.")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    r = sub.add_parser("remember", help="Teach the gate this matters")
    r.add_argument("target", help="path, pattern, or name")
    r.add_argument("--note", help="optional note explaining why")
    r.set_defaults(func=cmd_remember)

    f = sub.add_parser("forget", help="Teach the gate this doesn't matter")
    f.add_argument("target", help="path, pattern, or name")
    f.add_argument("--note", help="optional note")
    f.set_defaults(func=cmd_forget)


__all__ = ["cmd_remember", "cmd_forget", "register"]
