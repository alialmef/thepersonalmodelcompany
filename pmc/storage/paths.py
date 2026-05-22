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
          verification/probes.jsonl           # private user-specific eval probes
          verification/judgments.jsonl        # user judgments over probes
          verification/action_proposals.jsonl # dry-run action proposals
          verification/action_traces.jsonl    # supervised tool/action decisions
          verification/action_receipts.jsonl  # simulate/stage/execute/undo receipts
          actions/undo/                       # local undo snapshots for adapters
          world/files.jsonl                   # latest laptop-world file index
          world/scans.jsonl                   # scan reports
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

    # -- memory layer (recall) -------------------------------------------

    def memory_dir(self, user_id: str) -> Path:
        return self.user_root(user_id) / "memory"

    def memory_store_file(self, user_id: str) -> Path:
        """Per-user SQLite vector store for semantic recall."""
        return self.memory_dir(user_id) / "store.db"

    def identity_file(self, user_id: str) -> Path:
        """Per-user IdentityProfile (system-prompt facts for inference)."""
        return self.user_root(user_id) / "identity.json"

    def runs_ledger_file(self, user_id: str) -> Path:
        """Append-only ledger of every training run + its eval scalar.
        Powers the per-user "your model over time" view."""
        return self.user_root(user_id) / "runs.jsonl"

    # -- verification + personal evals -----------------------------------

    def verification_dir(self, user_id: str) -> Path:
        """Durable private eval and correction artifacts."""
        return self.user_root(user_id) / "verification"

    def verification_probes_file(self, user_id: str) -> Path:
        return self.verification_dir(user_id) / "probes.jsonl"

    def verification_judgments_file(self, user_id: str) -> Path:
        return self.verification_dir(user_id) / "judgments.jsonl"

    def action_proposals_file(self, user_id: str) -> Path:
        return self.verification_dir(user_id) / "action_proposals.jsonl"

    def action_traces_file(self, user_id: str) -> Path:
        return self.verification_dir(user_id) / "action_traces.jsonl"

    def action_receipts_file(self, user_id: str) -> Path:
        return self.verification_dir(user_id) / "action_receipts.jsonl"

    def action_runtime_dir(self, user_id: str) -> Path:
        return self.user_root(user_id) / "actions"

    def action_undo_dir(self, user_id: str) -> Path:
        return self.action_runtime_dir(user_id) / "undo"

    # -- laptop world index ----------------------------------------------

    def world_dir(self, user_id: str) -> Path:
        return self.user_root(user_id) / "world"

    def world_files_file(self, user_id: str) -> Path:
        return self.world_dir(user_id) / "files.jsonl"

    def world_scans_file(self, user_id: str) -> Path:
        return self.world_dir(user_id) / "scans.jsonl"
