"""Local config for the terminal CLI.

The web backend stores per-account provider/model/api_key in Postgres
behind the auth layer. The terminal CLI runs without an account —
it's a single-user local tool — so config lives in a small JSON file
at `~/.pmc/agent.json`.

The API key is stored in plaintext. This matches what every other
CLI tool that takes a provider key does (gh, openai, anthropic, etc.)
and is consistent with the local-only threat model: anyone with read
access to your home directory already has access to the graph itself,
which is the more sensitive artifact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path(os.path.expanduser("~/.pmc"))
CONFIG_FILE = CONFIG_DIR / "agent.json"


# Default storage root the Rust extractors write to (see
# desktop/src/graph/store.rs). Override with `PMC_STORAGE_ROOT`.
DEFAULT_STORAGE_ROOT = Path(os.path.expanduser("~/.pmc-dev/storage"))


@dataclass
class LocalConfig:
    provider: str          # "anthropic" | "openai" | "google" | "openrouter"
    model: str
    api_key: str
    user_id: str = "local"
    user_email: str = ""
    storage_root: str = ""

    def effective_storage_root(self) -> Path:
        env = os.environ.get("PMC_STORAGE_ROOT")
        if env:
            return Path(os.path.expanduser(env))
        if self.storage_root:
            return Path(os.path.expanduser(self.storage_root))
        return DEFAULT_STORAGE_ROOT


def load() -> Optional[LocalConfig]:
    if not CONFIG_FILE.is_file():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return LocalConfig(
            provider=data["provider"],
            model=data["model"],
            api_key=data["api_key"],
            user_id=data.get("user_id", "local"),
            user_email=data.get("user_email", ""),
            storage_root=data.get("storage_root", ""),
        )
    except KeyError:
        return None


def save(cfg: LocalConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": cfg.provider,
        "model": cfg.model,
        "api_key": cfg.api_key,
        "user_id": cfg.user_id,
        "user_email": cfg.user_email,
        "storage_root": cfg.storage_root,
    }
    # Best-effort 0600 perms.
    CONFIG_FILE.write_text(json.dumps(payload, indent=2))
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return key[:4] + "•" * 6 + key[-4:]


def discover_user_ids(storage_root: Path) -> list[tuple[str, int]]:
    """Return `(user_id, total_entity_count)` for every user with a graph
    directory under `storage_root`, sorted by count descending. Empty
    list if the root doesn't exist.

    Used by `pmc configure` and `pmc chat` to auto-pick the populated
    graph on this machine when the user hasn't specified one — so we
    don't make them paste a UUID just to talk to their agent."""
    users_dir = storage_root / "users"
    if not users_dir.is_dir():
        return []
    found: list[tuple[str, int]] = []
    for d in users_dir.iterdir():
        if not d.is_dir():
            continue
        graph_dir = d / "graph"
        if not graph_dir.is_dir():
            continue
        total = 0
        for fp in graph_dir.glob("*.jsonl"):
            try:
                # Cheap line count — no JSON parse.
                with fp.open("rb") as f:
                    total += sum(1 for _ in f)
            except OSError:
                continue
        if total > 0:
            found.append((d.name, total))
    found.sort(key=lambda t: t[1], reverse=True)
    return found


def auto_pick_user_id(storage_root: Path) -> str | None:
    """Return the user_id with the most entities, or None if no
    populated graph exists. Heuristic: take the top entry if it has
    >100 entities (the threshold rules out empty stubs like `demo`)."""
    candidates = discover_user_ids(storage_root)
    if not candidates:
        return None
    top_id, top_count = candidates[0]
    if top_count < 100:
        return None
    return top_id
