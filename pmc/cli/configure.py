"""`pmc configure` — set or inspect local provider/model/api_key.

Visual flow on rich.prompt. Interactive when called with no args;
flag-driven when scripted. Ctrl-C aborts cleanly with no traceback.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from pmc.agent.providers.base import ProviderError
from pmc.agent.providers.registry import (
    KNOWN_PROVIDERS,
    get_provider,
    is_known_provider,
)
from pmc.cli.local_config import (
    CONFIG_FILE,
    DEFAULT_STORAGE_ROOT,
    LocalConfig,
    auto_pick_user_id,
    discover_user_ids,
    load,
    mask_key,
    save,
)


# Shared theme with chat.py
STYLE_AGENT = "cyan"
STYLE_DIM = "dim"
STYLE_TITLE = "bold cyan"
STYLE_OK = "green"
STYLE_WARN = "yellow"


def cmd_show(args) -> int:
    console = Console()
    cfg = load()
    if cfg is None:
        console.print(f"[{STYLE_DIM}](no config at {CONFIG_FILE})[/]")
        console.print(f"run [{STYLE_AGENT}]pmc configure[/] to set up your provider.")
        return 1
    rows = [
        ("config",       str(CONFIG_FILE)),
        ("provider",     cfg.provider),
        ("model",        cfg.model),
        ("api_key",      mask_key(cfg.api_key)),
        ("user_id",      cfg.user_id),
        ("user_email",   cfg.user_email or "(unset)"),
        ("storage_root", str(cfg.effective_storage_root())),
    ]
    body = Text()
    for k, v in rows:
        body.append(f"{k:>14}  ", style=STYLE_DIM)
        body.append(f"{v}\n")
    console.print()
    console.print(Panel(body, title="pmc config", border_style=STYLE_AGENT,
                        padding=(1, 2)))
    console.print()
    return 0


def cmd_configure(args) -> int:
    console = Console()
    try:
        return _cmd_configure_inner(args, console)
    except (KeyboardInterrupt, EOFError):
        console.print()
        console.print(f"[{STYLE_DIM}](cancelled — nothing saved)[/]")
        return 130


def _cmd_configure_inner(args, console: Console) -> int:
    existing = load()
    _print_intro(console, existing)

    provider = (args.provider or "").strip().lower() or _ask_provider(console, existing)
    if not is_known_provider(provider):
        console.print(f"[{STYLE_WARN}]unknown provider {provider!r}. options: "
                      f"anthropic, openai, google, openrouter[/]")
        return 1
    model = (args.model or "").strip() or _ask_model(console, provider, existing)
    api_key = (args.api_key or "").strip() or _ask_key(console, provider, existing)
    if not api_key:
        console.print(f"[{STYLE_WARN}]no key provided. nothing saved.[/]")
        return 1

    storage_root = (args.storage_root or "").strip() or (existing.storage_root if existing else "")
    effective_root = Path(storage_root).expanduser() if storage_root else DEFAULT_STORAGE_ROOT

    user_id = (args.user_id or "").strip()
    if not user_id and existing and existing.user_id and existing.user_id != "local":
        user_id = existing.user_id
    if not user_id:
        # Auto-discover the populated graph on this machine. If found,
        # use it; this is what the user almost always wants.
        discovered = auto_pick_user_id(effective_root)
        if discovered:
            console.print()
            console.print(f"[{STYLE_DIM}]found a populated graph at[/] "
                          f"[{STYLE_AGENT}]{effective_root}/users/{discovered}[/]")
            console.print(f"[{STYLE_DIM}]using it as your user_id (override with "
                          f"--user-id).[/]")
            user_id = discovered
        else:
            user_id = "local"

    user_email = (args.user_email or "").strip() or (existing.user_email if existing else "")

    cfg = LocalConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        user_id=user_id,
        user_email=user_email,
        storage_root=storage_root,
    )
    save(cfg)
    console.print()
    console.print(f"[{STYLE_OK}]✓[/] saved → [{STYLE_DIM}]{CONFIG_FILE}[/]")

    if not args.no_validate:
        ok = _validate(console, provider, api_key)
        if not ok:
            console.print()
            console.print(f"[{STYLE_DIM}](the key was saved but the test call didn't succeed. "
                          f"could be a stale model id, network, or wrong key.[/]")
            console.print(f"[{STYLE_DIM}] run [{STYLE_AGENT}]pmc chat[/{STYLE_DIM}] to try it for real, "
                          f"or [{STYLE_AGENT}]pmc configure[/{STYLE_DIM}] again to replace it.)[/]")

    # First-run hook: if the user has no graph yet, offer to ingest.
    # Calls the same Rust engine the Mac app uses (pmc-ingest binary).
    if _graph_is_empty(cfg) and not args.no_ingest:
        console.print()
        if Confirm.ask("ingest your data now? (reads your mac so the agent has "
                       "something to talk about)", default=True):
            _run_ingest(console, cfg)

    console.print()
    console.print(f"[{STYLE_AGENT}]→[/] run [{STYLE_AGENT}]pmc chat[/] to talk to your agent.")
    return 0


def _graph_is_empty(cfg: LocalConfig) -> bool:
    """Cheap check — does the user have any graph files yet?"""
    try:
        from pmc.storage.graph_store import GraphStore
        store = GraphStore(cfg.effective_storage_root())
        return sum(store.counts(cfg.user_id or "local").values()) == 0
    except Exception:  # noqa: BLE001
        return True


def _run_ingest(console: Console, cfg: LocalConfig) -> None:
    """Delegate to pmc.cli.ingest, but pass an args-like shim instead
    of a real argparse.Namespace so we keep the import local."""
    from types import SimpleNamespace
    from pmc.cli.ingest import cmd_ingest
    cmd_ingest(SimpleNamespace(user=cfg.user_id, json=False, no_build=False))


# ---------------------------------------------------------------------------
# Intro panel
# ---------------------------------------------------------------------------


def _print_intro(console: Console, existing: Optional[LocalConfig]) -> None:
    if existing is None:
        body = Text("set up your model provider and api key. "
                    "everything is stored locally at ~/.pmc/agent.json.",
                    style="")
    else:
        body = Text()
        body.append("updating existing config at ", style=STYLE_DIM)
        body.append(f"~/.pmc/agent.json\n\n", style=STYLE_DIM)
        body.append("current: ", style=STYLE_DIM)
        body.append(f"{existing.provider}/{existing.model}", style=STYLE_AGENT)
        body.append("\npress enter to keep current values where shown in brackets.",
                    style=STYLE_DIM)
    console.print()
    console.print(Panel(body, title="pmc configure",
                        border_style=STYLE_AGENT, padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _ask_provider(console: Console, existing: Optional[LocalConfig]) -> str:
    console.print(f"[{STYLE_TITLE}]which model provider?[/]")
    for i, p in enumerate(KNOWN_PROVIDERS, 1):
        marker = f" [{STYLE_DIM}](current)[/]" if existing and existing.provider == p["id"] else ""
        console.print(f"  [{STYLE_AGENT}][{i}][/] {p['label']}{marker}")
    while True:
        raw = Prompt.ask("choose").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(KNOWN_PROVIDERS):
            return KNOWN_PROVIDERS[int(raw) - 1]["id"]
        if is_known_provider(raw.lower()):
            return raw.lower()
        console.print(f"[{STYLE_WARN}]invalid choice — pick a number 1-4 or a provider name.[/]")


def _ask_model(console: Console, provider: str, existing: Optional[LocalConfig]) -> str:
    info = next((p for p in KNOWN_PROVIDERS if p["id"] == provider), None)
    defaults = (info or {}).get("default_models") or []
    if defaults:
        console.print()
        console.print(f"[{STYLE_TITLE}]model[/]")
        for i, m in enumerate(defaults, 1):
            console.print(f"  [{STYLE_AGENT}][{i}][/] {m}")
        console.print(f"  [{STYLE_DIM}]or paste a custom model id[/]")
    fallback = existing.model if existing and existing.provider == provider else (
        defaults[0] if defaults else ""
    )
    raw = Prompt.ask("choose", default=fallback).strip()
    if not raw:
        return fallback
    if raw.isdigit() and defaults and 1 <= int(raw) <= len(defaults):
        return defaults[int(raw) - 1]
    if defaults and raw not in defaults:
        console.print(f"[{STYLE_DIM}]'{raw}' isn't in the default list — assuming "
                      f"you mean a custom model id.[/]")
        if not Confirm.ask("use it anyway?", default=True):
            return _ask_model(console, provider, existing)
    return raw


def _ask_key(console: Console, provider: str, existing: Optional[LocalConfig]) -> str:
    info = next((p for p in KNOWN_PROVIDERS if p["id"] == provider), None)
    hint = (info or {}).get("key_prefix_hint")
    console_url = (info or {}).get("console_url")
    console.print()
    console.print(f"[{STYLE_TITLE}]api key[/]")
    if hint:
        console.print(f"[{STYLE_DIM}]looks like {hint}…  get one at {console_url}[/]")
    console.print(f"[{STYLE_DIM}](your input is hidden — paste your key and press enter)[/]")
    if existing and existing.provider == provider:
        console.print(f"[{STYLE_DIM}](or press enter to keep the existing key: "
                      f"{mask_key(existing.api_key)})[/]")
    # `password=True` on rich.Prompt masks input the same way getpass does.
    raw = Prompt.ask("key", password=True).strip()
    if not raw and existing and existing.provider == provider:
        return existing.api_key
    return raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(console: Console, provider: str, api_key: str) -> bool:
    p = get_provider(provider)
    if p is None:
        return False
    with console.status(f"[{STYLE_DIM}]checking key with {provider}…[/]"):
        try:
            ok = asyncio.run(p.validate_key(api_key=api_key))
        except ProviderError as e:
            console.print(f"[{STYLE_WARN}]failed [{e.kind}] {e}[/]")
            return False
        except Exception as e:  # noqa: BLE001
            console.print(f"[{STYLE_WARN}]failed: {e}[/]")
            return False
    if ok:
        console.print(f"[{STYLE_OK}]✓[/] key works.")
    else:
        console.print(f"[{STYLE_WARN}]× key was not accepted.[/]")
    return bool(ok)


__all__ = ["cmd_configure", "cmd_show"]
