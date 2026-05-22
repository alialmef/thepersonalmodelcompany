"""Per-user data store: raw items + curated datasets.

Why partition raw data by source_id: when a user says "delete all my Gmail
data," we just delete `raw/{source_id}.jsonl` for that source. No scanning,
no row-level surgery. The curated dataset is then stale and must be rebuilt
(deletion → retrain is the model).

Datasets are versioned by an opaque string (a timestamp, a UUID, whatever the
caller chooses). The store persists each version so prior runs remain
reproducible until explicitly cleaned up.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

# Pre-fix native ingesters generated source_ids like
# "imessage-20260521-181322" — a new timestamped file on every Connect
# click, so the raw/ dir accumulated duplicates and the curate stage
# re-processed the same content N times. Native ingesters now send
# stable per-kind ids ("imessage", "notes", "email") that overwrite a
# single canonical file. When a stable id writes, any legacy timestamped
# sibling for the same kind is removed — quietly migrating pre-fix
# users on their next ingest.
_LEGACY_TIMESTAMP_SUFFIX = re.compile(r"^-\d{8}-\d{6}$")

from pmc.ingest.base import RawItem
from pmc.schema.conversation import Completion
from pmc.schema.user import DataManifest, User
from pmc.storage.paths import StoragePaths


class UserStore:
    """Per-user data persistence with full isolation."""

    def __init__(self, root: Path | str) -> None:
        self.paths = StoragePaths(root)

    # -- user profile -----------------------------------------------------

    def save_user(self, user: User, *, user_id: str | None = None) -> None:
        """Persist a user profile. Keyed by `user_id` if given, else `str(user.id)`."""
        key = user_id if user_id is not None else str(user.id)
        path = self.paths.user_file(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(user.model_dump_json(indent=2))

    def load_user(self, user_id: str) -> User | None:
        path = self.paths.user_file(user_id)
        if not path.is_file():
            return None
        return User.model_validate_json(path.read_text())

    # -- raw items (per-source partitioning) ------------------------------

    def save_raw_items(
        self,
        user_id: str,
        source_id: str,
        items: Iterable[RawItem],
        *,
        append: bool = False,
    ) -> int:
        """Save items for one source. By default overwrites; set append=True to
        add to an existing source file."""
        path = self.paths.raw_file(user_id, source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self._drop_legacy_timestamped_siblings(user_id, source_id)
        mode = "a" if append and path.is_file() else "w"
        count = 0
        with path.open(mode, encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")
                count += 1
        return count

    def _drop_legacy_timestamped_siblings(self, user_id: str, source_id: str) -> None:
        # Only fires when the caller is using a modern stable id (no
        # embedded timestamp). For a legacy id like "imessage-2026...",
        # the suffix check would match itself and self-delete — guard
        # against that by skipping ids that already look legacy.
        if _LEGACY_TIMESTAMP_SUFFIX.search(source_id):
            return
        raw_dir = self.paths.raw_dir(user_id)
        if not raw_dir.is_dir():
            return
        prefix = f"{source_id}-"
        for sibling in raw_dir.glob(f"{source_id}-*.jsonl"):
            suffix = sibling.stem[len(prefix) - 1 :]  # keep leading dash
            if _LEGACY_TIMESTAMP_SUFFIX.match(suffix):
                sibling.unlink(missing_ok=True)

    def load_raw_items(
        self,
        user_id: str,
        source_id: str | None = None,
    ) -> Iterator[RawItem]:
        """Stream raw items. If source_id is None, yields all sources."""
        if source_id is not None:
            yield from self._read_jsonl(self.paths.raw_file(user_id, source_id), RawItem)
            return
        raw_dir = self.paths.raw_dir(user_id)
        if not raw_dir.is_dir():
            return
        for path in sorted(raw_dir.glob("*.jsonl")):
            yield from self._read_jsonl(path, RawItem)

    def list_sources(self, user_id: str) -> list[str]:
        """All source IDs that have raw data stored for this user."""
        raw_dir = self.paths.raw_dir(user_id)
        if not raw_dir.is_dir():
            return []
        return sorted(p.stem for p in raw_dir.glob("*.jsonl"))

    def delete_source(self, user_id: str, source_id: str) -> bool:
        """Remove all raw data for one source. Returns True if a file existed."""
        path = self.paths.raw_file(user_id, source_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def count_raw_items(self, user_id: str, source_id: str | None = None) -> int:
        if source_id is not None:
            return self._count_lines(self.paths.raw_file(user_id, source_id))
        raw_dir = self.paths.raw_dir(user_id)
        if not raw_dir.is_dir():
            return 0
        return sum(self._count_lines(p) for p in raw_dir.glob("*.jsonl"))

    # -- curated datasets (versioned) -------------------------------------

    def save_curated_dataset(
        self,
        user_id: str,
        version: str,
        train: list[Completion],
        holdout: list[Completion] | None = None,
        *,
        manifest: DataManifest | None = None,
    ) -> DataManifest:
        """Persist a curated dataset. Returns the (possibly synthesized) manifest."""
        train_path = self.paths.curated_file(user_id, version)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(train_path, train)

        if holdout is not None:
            holdout_path = self.paths.holdout_file(user_id, version)
            self._write_jsonl(holdout_path, holdout)

        if manifest is None:
            sources = self._summarize_sources(train + (holdout or []))
            manifest = DataManifest(
                training_run_id=uuid.uuid4(),
                dataset_version=version,
                num_examples=len(train),
                source_breakdown=sources,
                checksum=_dataset_checksum(train),
            )
        else:
            manifest.checksum = manifest.checksum or _dataset_checksum(train)
            manifest.num_examples = manifest.num_examples or len(train)

        manifest_path = self.paths.manifest_file(user_id, version)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.model_dump_json(indent=2))
        return manifest

    def load_curated_dataset(self, user_id: str, version: str) -> list[Completion]:
        return list(self._read_jsonl(self.paths.curated_file(user_id, version), Completion))

    def load_holdout(self, user_id: str, version: str) -> list[Completion]:
        return list(self._read_jsonl(self.paths.holdout_file(user_id, version), Completion))

    def load_manifest(self, user_id: str, version: str) -> DataManifest | None:
        path = self.paths.manifest_file(user_id, version)
        if not path.is_file():
            return None
        return DataManifest.model_validate_json(path.read_text())

    def list_dataset_versions(self, user_id: str) -> list[str]:
        curated = self.paths.curated_dir(user_id)
        if not curated.is_dir():
            return []
        return sorted(
            p.stem
            for p in curated.glob("*.jsonl")
            if not p.stem.startswith("holdout_")
        )

    def delete_dataset(self, user_id: str, version: str) -> bool:
        deleted = False
        for path in [
            self.paths.curated_file(user_id, version),
            self.paths.holdout_file(user_id, version),
            self.paths.manifest_file(user_id, version),
        ]:
            if path.is_file():
                path.unlink()
                deleted = True
        return deleted

    # -- hard delete ------------------------------------------------------

    def delete_user(self, user_id: str) -> bool:
        """Remove everything for this user. Irreversible."""
        import shutil
        user_root = self.paths.user_root(user_id)
        if not user_root.is_dir() or not any(user_root.iterdir()):
            shutil.rmtree(user_root, ignore_errors=True)
            return False
        shutil.rmtree(user_root, ignore_errors=True)
        return True

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _read_jsonl(path: Path, model_cls):
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield model_cls.model_validate_json(line)

    @staticmethod
    def _write_jsonl(path: Path, items: Iterable) -> None:
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    @staticmethod
    def _summarize_sources(completions: list[Completion]) -> dict[str, int]:
        """Count completions by source_type for the manifest."""
        counts: dict[str, int] = {}
        for c in completions:
            source = c.conversation.source_type
            key = source.value if source else "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts


def _dataset_checksum(completions: list[Completion]) -> str:
    """Stable SHA-256 over the IDs and candidate text of the completions."""
    h = hashlib.sha256()
    for c in completions:
        h.update(str(c.id).encode())
        for cand in c.candidates:
            for msg in cand.messages:
                h.update(msg.content.encode("utf-8"))
                h.update(b"\x00")
    return h.hexdigest()[:32]


__all__ = ["UserStore"]
