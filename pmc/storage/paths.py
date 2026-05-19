"""Canonical disk layout for PMC storage.

All paths live under a single root. Per-user data is fully isolated under
`{root}/users/{user_id}/` — no shared directories across users. That makes
deletion clean: removing one user's directory removes everything we know about
them.

Layout:

    {root}/
      users/
        {user_id}/
          user.json                          # User profile snapshot
          raw/{source_id}.jsonl              # raw RawItems, partitioned per source
          curated/{version}.jsonl            # curated Completions
          curated/{version}_manifest.json    # DataManifest for this dataset
          curated/holdout_{version}.jsonl    # held-out completions
          bundles/{run_id}/                  # ArtifactBundle per training run
          active.json                        # pointer: which run_id is currently active
          audit.jsonl                        # append-only event log
          tombstones.json                    # deleted source IDs + timestamps
"""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9._\-]")


def safe_id(value: str, *, max_len: int = 128) -> str:
    """Make a string safe as a directory or filename component.

    Replaces unsafe chars with `_`, trims to `max_len`. Empty input becomes "_".
    """
    if not value:
        return "_"
    cleaned = _SAFE_ID_RE.sub("_", value.strip())
    return cleaned[:max_len] or "_"


class StoragePaths:
    """Resolves paths under a storage root.

    Path helpers are PURE — they compute paths without creating directories.
    This matters for deletion semantics: after a user is fully deleted, simply
    listing or checking their state must not silently recreate their directory.

    Use `ensure(path)` (or the parent's `path.parent.mkdir`) at write sites.
    """

    def __init__(self, root: Path | str, *, create_root: bool = True) -> None:
        self.root = Path(root)
        if create_root:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def ensure(path: Path) -> Path:
        """Create the directory if missing. Use at write sites."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- per-user roots ---------------------------------------------------

    def user_root(self, user_id: str) -> Path:
        return self.root / "users" / safe_id(user_id)

    def user_file(self, user_id: str) -> Path:
        return self.user_root(user_id) / "user.json"

    # -- raw data (partitioned per source) --------------------------------

    def raw_dir(self, user_id: str) -> Path:
        return self.user_root(user_id) / "raw"

    def raw_file(self, user_id: str, source_id: str) -> Path:
        return self.raw_dir(user_id) / f"{safe_id(source_id)}.jsonl"

    # -- curated datasets -------------------------------------------------

    def curated_dir(self, user_id: str) -> Path:
        return self.user_root(user_id) / "curated"

    def curated_file(self, user_id: str, version: str) -> Path:
        return self.curated_dir(user_id) / f"{safe_id(version)}.jsonl"

    def holdout_file(self, user_id: str, version: str) -> Path:
        return self.curated_dir(user_id) / f"holdout_{safe_id(version)}.jsonl"

    def manifest_file(self, user_id: str, version: str) -> Path:
        return self.curated_dir(user_id) / f"{safe_id(version)}_manifest.json"

    # -- artifact bundles -------------------------------------------------

    def bundles_dir(self, user_id: str) -> Path:
        return self.user_root(user_id) / "bundles"

    def bundle_dir(self, user_id: str, run_id: str) -> Path:
        return self.bundles_dir(user_id) / safe_id(run_id)

    def active_pointer_file(self, user_id: str) -> Path:
        return self.user_root(user_id) / "active.json"

    # -- audit + tombstones ----------------------------------------------

    def audit_file(self, user_id: str) -> Path:
        return self.user_root(user_id) / "audit.jsonl"

    def tombstones_file(self, user_id: str) -> Path:
        return self.user_root(user_id) / "tombstones.json"
