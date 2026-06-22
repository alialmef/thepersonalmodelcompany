"""`pmc doctor` — preflight diagnostic.

Single command that tells a fresh user exactly what's set up correctly
and what needs fixing, with copy-paste-able remediation steps. Run
this first; nothing else should be debug-by-traceback.

Checks, in order of importance:
  1. uv               (Python runtime)
  2. cargo or prebuilt pmc-ingest binary
  3. API key configured + provider validates
  4. Full Disk Access (try reading chat.db / Safari History.db)
  5. Disk space at the storage root
  6. Graph populated (informational — not having one is fine for new users)
  7. Portrait built (informational)

Each check returns:
  ok / warning / error  →  with a one-line fix the user can run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text

from pmc.cli import ui
from pmc.cli.local_config import load


@dataclass
class CheckResult:
    label: str
    state: str             # "ok" | "warning" | "error"
    detail: str = ""
    fix: str = ""          # copy-paste-able remediation


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_uv() -> CheckResult:
    if shutil.which("uv"):
        return CheckResult("uv (python runtime)", "ok")
    return CheckResult(
        "uv (python runtime)", "error",
        detail="uv is not on PATH",
        fix="curl -LsSf https://astral.sh/uv/install.sh | sh",
    )


def check_pmc_ingest() -> CheckResult:
    """Look for the Rust extractor binary or the toolchain to build it."""
    # Explicit override
    env_bin = os.environ.get("PMC_INGEST_BIN")
    if env_bin and Path(env_bin).expanduser().is_file():
        return CheckResult(
            "pmc-ingest (Rust extractor binary)", "ok",
            detail=f"explicit: {env_bin}",
        )
    # On PATH
    on_path = shutil.which("pmc-ingest")
    if on_path:
        return CheckResult(
            "pmc-ingest (Rust extractor binary)", "ok",
            detail=f"on PATH: {on_path}",
        )
    # Dev mode — built in the repo's target/release/examples/
    repo = Path(__file__).resolve().parents[2]
    dev = repo / "desktop/target/release/examples/pmc_ingest"
    if dev.is_file():
        return CheckResult(
            "pmc-ingest (Rust extractor binary)", "ok",
            detail=f"dev build: {dev}",
        )
    # Not built yet. Can we build it?
    if shutil.which("cargo"):
        return CheckResult(
            "pmc-ingest (Rust extractor binary)", "warning",
            detail="not yet built (cargo is available)",
            fix=f"cd {repo}/desktop && cargo build --example pmc_ingest --release",
        )
    return CheckResult(
        "pmc-ingest (Rust extractor binary)", "error",
        detail="not built and cargo isn't installed",
        fix=("install Rust:  curl --proto '=https' --tlsv1.2 -sSf "
             "https://sh.rustup.rs | sh"),
    )


def check_api_key() -> CheckResult:
    cfg = load()
    if cfg is None:
        return CheckResult(
            "API key (provider config)", "error",
            detail="no ~/.pmc/agent.json — run pmc configure",
            fix="pmc configure",
        )
    if not cfg.api_key:
        return CheckResult(
            "API key (provider config)", "error",
            detail=f"provider={cfg.provider} but key is empty",
            fix="pmc configure",
        )
    # Validate against the provider (live probe).
    from pmc.agent.providers.base import ProviderError
    from pmc.agent.providers.registry import get_provider
    p = get_provider(cfg.provider)
    if p is None:
        return CheckResult(
            "API key (provider config)", "error",
            detail=f"unknown provider {cfg.provider!r}",
            fix="pmc configure",
        )
    try:
        ok = asyncio.run(p.validate_key(api_key=cfg.api_key))
    except ProviderError as e:
        return CheckResult(
            "API key (provider config)", "error",
            detail=f"provider rejected key ({e.kind})",
            fix="pmc configure",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "API key (provider config)", "warning",
            detail=f"couldn't probe ({e})",
            fix="check network; pmc configure --no-validate to skip probe",
        )
    if not ok:
        return CheckResult(
            "API key (provider config)", "error",
            detail="provider rejected key",
            fix="pmc configure",
        )
    return CheckResult(
        "API key (provider config)", "ok",
        detail=f"{cfg.provider}/{cfg.model}",
    )


def check_fda() -> CheckResult:
    """Try to read chat.db. If FDA isn't granted, this fails with
    SQLITE_AUTH or PermissionError. We translate that into the
    System Settings instruction."""
    home = Path.home()
    chat_db = home / "Library/Messages/chat.db"
    history_db = home / "Library/Safari/History.db"

    if not chat_db.is_file() and not history_db.is_file():
        return CheckResult(
            "Full Disk Access (FDA)", "warning",
            detail="neither chat.db nor History.db exists — can't verify",
        )

    # Try chat.db first.
    target = chat_db if chat_db.is_file() else history_db
    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=2.0)
        try:
            # Trivial probe.
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "authoriz" in msg or "permission" in msg or "denied" in msg:
            return CheckResult(
                "Full Disk Access (FDA)", "error",
                detail=f"can't read {target.name} ({e})",
                fix=(
                    "open System Settings → Privacy & Security → "
                    "Full Disk Access → click `+` → add the terminal app "
                    "you're running pmc from (Terminal.app, iTerm.app, "
                    "Cursor.app, etc.)"
                ),
            )
        # Other SQLite error
        return CheckResult(
            "Full Disk Access (FDA)", "warning",
            detail=f"{target.name}: {e}",
        )
    except PermissionError as e:
        return CheckResult(
            "Full Disk Access (FDA)", "error",
            detail=f"permission denied on {target}",
            fix=(
                "open System Settings → Privacy & Security → "
                "Full Disk Access → click `+` → add your terminal app"
            ),
        )
    return CheckResult(
        "Full Disk Access (FDA)", "ok",
        detail=f"read access confirmed via {target.name}",
    )


def check_disk_space() -> CheckResult:
    """Make sure the storage root has at least 500MB free for the graph
    + page-content cache."""
    cfg = load()
    root = cfg.effective_storage_root() if cfg else Path.home() / ".pmc-dev/storage"
    root.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(root)
    except OSError as e:
        return CheckResult(
            "disk space at storage root", "warning",
            detail=str(e),
        )
    free_gb = usage.free / 1e9
    if free_gb < 0.5:
        return CheckResult(
            "disk space at storage root", "error",
            detail=f"only {free_gb:.2f} GB free at {root}",
            fix=f"free up space at {root}",
        )
    if free_gb < 5:
        return CheckResult(
            "disk space at storage root", "warning",
            detail=f"{free_gb:.1f} GB free at {root} (low)",
        )
    return CheckResult(
        "disk space at storage root", "ok",
        detail=f"{free_gb:.0f} GB free at {root}",
    )


def check_graph_populated() -> CheckResult:
    """Informational — having no graph is fine for a new user."""
    cfg = load()
    if cfg is None:
        return CheckResult(
            "graph populated", "warning",
            detail="(skipped — no config)",
        )
    from pmc.cli.local_config import discover_user_ids
    found = discover_user_ids(cfg.effective_storage_root())
    if not found:
        return CheckResult(
            "graph populated", "warning",
            detail="no graph data yet",
            fix="pmc connect",
        )
    uid, n = found[0]
    return CheckResult(
        "graph populated", "ok",
        detail=f"{n:,} entities for {uid[:8]}…",
    )


def check_portrait_built() -> CheckResult:
    """Informational — no portrait is fine; it's the next step."""
    cfg = load()
    if cfg is None:
        return CheckResult(
            "portrait built", "warning",
            detail="(skipped — no config)",
        )
    portrait = (
        cfg.effective_storage_root() / "users" / (cfg.user_id or "local")
        / "graph" / "synth" / "portrait" / "picture.md"
    )
    if portrait.is_file():
        return CheckResult(
            "portrait built", "ok",
            detail=f"picture.md present ({portrait.stat().st_size} bytes)",
        )
    return CheckResult(
        "portrait built", "warning",
        detail="no picture.md yet",
        fix="pmc sandbox",
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


CHECKS = [
    check_uv,
    check_pmc_ingest,
    check_api_key,
    check_fda,
    check_disk_space,
    check_graph_populated,
    check_portrait_built,
]


def cmd_doctor(args: argparse.Namespace) -> int:
    console = Console()
    if args.clear:
        console.clear()
    ui.banner_top(console, title="pmc doctor",
                  subtitle="preflight checks for a fresh install")

    results: list[CheckResult] = []
    for fn in CHECKS:
        try:
            results.append(fn())
        except Exception as e:  # noqa: BLE001
            results.append(CheckResult(
                fn.__name__, "error",
                detail=f"check itself errored: {e}",
            ))

    errors = sum(1 for r in results if r.state == "error")
    warnings = sum(1 for r in results if r.state == "warning")

    for r in results:
        line = Text(ui.margin(), style="")
        if r.state == "ok":
            line.append(f"{ui.GLYPH_DONE} ", style=ui.OK)
        elif r.state == "warning":
            line.append("△ ", style=ui.WARN)
        else:
            line.append(f"{ui.GLYPH_ERROR} ", style=ui.WARN)
        line.append(f"{r.label:<35}", style=ui.WHITE)
        if r.detail:
            line.append(f"  {r.detail}", style=ui.DIM)
        console.print(line, highlight=False)
        if r.fix and r.state != "ok":
            console.print(f"{ui.margin()}    [{ui.ACCENT}]fix:[/]  "
                          f"[{ui.DIM}]{r.fix}[/]", highlight=False)
        console.print()

    # Summary
    if errors == 0 and warnings == 0:
        ui.say(console, f"{ui.GLYPH_DONE} all checks pass — you're ready to go.",
               style=ui.OK)
        ui.say_dim(console, "next: pmc connect → pmc sandbox → pmc install-mcp claude")
    elif errors == 0:
        ui.say(console, f"{ui.GLYPH_DONE} no errors. {warnings} warning(s).",
               style=ui.OK)
        ui.say_dim(console, "you can proceed; warnings are usually 'next step' hints.")
    else:
        ui.say(console,
               f"× {errors} error(s) — fix these before running pmc connect.",
               style=ui.WARN)

    return 0 if errors == 0 else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Preflight checks — run before pmc connect to see what's missing",
    )
    p.add_argument("--no-clear", dest="clear", action="store_false",
                   default=True, help="don't clear the screen")
    p.set_defaults(func=cmd_doctor)


__all__ = ["cmd_doctor", "register"]
