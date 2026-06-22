"""`pmc serve` — expose PMC's substrate via MCP.

Two related surfaces under one command:
   pmc serve --mcp            run the MCP server (stdio, for agent subprocess)
   pmc install-mcp <agent>    install the MCP config into the agent
   pmc uninstall-mcp <agent>  remove it

Supported agents (for install-mcp):
   claude   — Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json)
   cursor   — Cursor (~/.cursor/mcp.json)
   continue — Continue (~/.continue/config.json)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

from pmc.cli import ui


# ---------------------------------------------------------------------------
# `pmc serve --mcp`
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    if not args.mcp:
        # No transport selected → print help.
        print("usage: pmc serve --mcp")
        return 2
    from pmc.serve.mcp_server import serve_blocking
    serve_blocking()
    return 0


# ---------------------------------------------------------------------------
# `pmc install-mcp <agent>`
# ---------------------------------------------------------------------------


HOME = Path.home()

CLAUDE_DESKTOP_CONFIG = HOME / "Library/Application Support/Claude/claude_desktop_config.json"
CURSOR_CONFIG = HOME / ".cursor/mcp.json"
CONTINUE_CONFIG = HOME / ".continue/config.json"


def _pmc_binary() -> str:
    found = shutil.which("pmc")
    if found:
        return found
    repo = Path(__file__).resolve().parents[2]
    venv = repo / ".venv/bin/pmc"
    if venv.is_file():
        return str(venv)
    return "pmc"


def _server_entry(label: str = "pmc") -> dict[str, Any]:
    return {
        "command": _pmc_binary(),
        "args": ["serve", "--mcp"],
    }


def cmd_install_mcp(args: argparse.Namespace) -> int:
    console = Console()
    agent = (args.agent or "").lower()
    if agent == "claude":
        return _install_claude(console)
    if agent == "cursor":
        return _install_cursor(console)
    if agent == "continue":
        return _install_continue(console)
    ui.say(console, f"unknown agent {agent!r}. supported: claude, cursor, continue",
           style=ui.WARN)
    return 1


def cmd_uninstall_mcp(args: argparse.Namespace) -> int:
    console = Console()
    agent = (args.agent or "").lower()
    if agent == "claude":
        return _remove_from_json(console, CLAUDE_DESKTOP_CONFIG, ["mcpServers", "pmc"])
    if agent == "cursor":
        return _remove_from_json(console, CURSOR_CONFIG, ["mcpServers", "pmc"])
    if agent == "continue":
        return _remove_from_json(console, CONTINUE_CONFIG, ["mcpServers", "pmc"])
    ui.say(console, f"unknown agent {agent!r}", style=ui.WARN)
    return 1


# ---------------------------------------------------------------------------
# Per-agent installers
# ---------------------------------------------------------------------------


def _install_claude(console: Console) -> int:
    return _install_into_json(
        console,
        path=CLAUDE_DESKTOP_CONFIG,
        keys=["mcpServers", "pmc"],
        value=_server_entry(),
        agent_label="Claude Desktop",
        restart_hint="restart Claude Desktop to pick it up.",
    )


def _install_cursor(console: Console) -> int:
    return _install_into_json(
        console,
        path=CURSOR_CONFIG,
        keys=["mcpServers", "pmc"],
        value=_server_entry(),
        agent_label="Cursor",
        restart_hint="restart Cursor (Cmd+Q then reopen) to pick it up.",
    )


def _install_continue(console: Console) -> int:
    # Continue uses a different format — its config.json has an
    # `mcpServers` array (sometimes) or settings.json. We use the
    # standard map shape for simplicity.
    return _install_into_json(
        console,
        path=CONTINUE_CONFIG,
        keys=["mcpServers", "pmc"],
        value=_server_entry(),
        agent_label="Continue",
        restart_hint="restart your editor to pick it up.",
    )


def _install_into_json(
    console: Console,
    *,
    path: Path,
    keys: list[str],
    value: dict[str, Any],
    agent_label: str,
    restart_hint: str,
) -> int:
    """Merge `value` into `path` at the nested key sequence, writing
    the file back. Creates the file + parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            ui.say(console,
                   f"warning: {path} exists but couldn't be parsed; "
                   "will overwrite the pmc section only.",
                   style=ui.WARN)
            data = {}
    cursor = data
    for k in keys[:-1]:
        if not isinstance(cursor.get(k), dict):
            cursor[k] = {}
        cursor = cursor[k]
    cursor[keys[-1]] = value
    path.write_text(json.dumps(data, indent=2))
    ui.say(console, f"{ui.GLYPH_DONE} wrote {path}", style=ui.OK)
    ui.say_dim(console, restart_hint)
    ui.say_dim(console, f"after restart, ask {agent_label}: 'what do you know about me?'")
    return 0


def _remove_from_json(
    console: Console, path: Path, keys: list[str],
) -> int:
    if not path.is_file():
        ui.say_dim(console, f"(no config at {path}; nothing to remove)")
        return 0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        ui.say(console, f"{path} not valid JSON; not touching", style=ui.WARN)
        return 1
    cursor = data
    for k in keys[:-1]:
        if not isinstance(cursor.get(k), dict):
            ui.say_dim(console, "(pmc not in config)")
            return 0
        cursor = cursor[k]
    if keys[-1] in cursor:
        del cursor[keys[-1]]
        path.write_text(json.dumps(data, indent=2))
        ui.say(console, f"{ui.GLYPH_DONE} removed pmc from {path}", style=ui.OK)
    else:
        ui.say_dim(console, "(pmc not in config)")
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser(
        "serve",
        help="Run PMC as an MCP server — agents subscribe over stdio",
    )
    s.add_argument("--mcp", action="store_true",
                   help="run the MCP stdio server")
    s.set_defaults(func=cmd_serve)

    inst = sub.add_parser(
        "install-mcp",
        help="Install PMC into your agent (claude / cursor / continue)",
    )
    inst.add_argument("agent", help="claude | cursor | continue")
    inst.set_defaults(func=cmd_install_mcp)

    uninst = sub.add_parser(
        "uninstall-mcp",
        help="Remove PMC from your agent's MCP config",
    )
    uninst.add_argument("agent")
    uninst.set_defaults(func=cmd_uninstall_mcp)


__all__ = ["register"]
