"""AdapterRegistry — persistent map of user_id → adapter / bundle on disk.

Each registered user has an `AdapterRecord` pointing at their adapter directory
(and optionally the parent ArtifactBundle). The registry persists to a JSON
file at `root/registry.json` so a restarted server picks up where it left off.

For serving warmth (which adapters are loaded into GPU memory right now), see
the engine — the registry only tracks what *exists*.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pmc.train.bundle import ArtifactBundle
from pmc.train.checkpoint import adapter_info, is_valid_adapter

REGISTRY_FILE = "registry.json"


class AdapterRecord(BaseModel):
    """One entry in the registry — everything we need to serve this user's model."""

    user_id: str
    adapter_dir: str  # absolute path on disk
    base_model: str
    bundle_dir: str | None = None
    adapter_size_mb: float = 0.0
    rank: int | None = None
    registered_at: datetime = Field(default_factory=datetime.now)
    last_served_at: datetime | None = None
    request_count: int = 0
    # Engine-specific identifiers (e.g. together_adapter_id, modal_adapter_url).
    # Engines look here for the references they need to route to a user's adapter.
    metadata: dict[str, str] = Field(default_factory=dict)


class AdapterRegistry:
    """File-backed registry of user_id → AdapterRecord."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, AdapterRecord] = {}
        self._load()

    # -- registration ------------------------------------------------------

    def register(
        self,
        user_id: str,
        adapter_dir: Path | str,
        base_model: str,
        *,
        bundle_dir: Path | str | None = None,
    ) -> AdapterRecord:
        """Add or update an adapter for a user."""
        adapter_path = Path(adapter_dir).resolve()
        if not is_valid_adapter(adapter_path):
            raise ValueError(f"Not a valid LoRA adapter directory: {adapter_path}")

        info = adapter_info(adapter_path)
        record = AdapterRecord(
            user_id=user_id,
            adapter_dir=str(adapter_path),
            base_model=base_model,
            bundle_dir=str(Path(bundle_dir).resolve()) if bundle_dir else None,
            adapter_size_mb=round(info.size_bytes / (1024 * 1024), 3),
            rank=info.rank,
            metadata=_remote_metadata(adapter_path),
        )
        self._records[user_id] = record
        self._save()
        return record

    def register_bundle(self, bundle_dir: Path | str) -> AdapterRecord:
        """Register straight from an ArtifactBundle directory."""
        bundle = ArtifactBundle.load(bundle_dir)
        return self.register(
            user_id=bundle.metadata.user_id,
            adapter_dir=bundle.adapter_dir,
            base_model=bundle.metadata.base_model,
            bundle_dir=bundle_dir,
        )

    def unregister(self, user_id: str, *, delete_files: bool = False) -> bool:
        """Remove a user from the registry. If `delete_files`, also delete the
        adapter and bundle directories — used for hard-delete on user request."""
        record = self._records.pop(user_id, None)
        if record is None:
            return False
        if delete_files:
            adapter_path = Path(record.adapter_dir)
            if adapter_path.is_dir():
                shutil.rmtree(adapter_path, ignore_errors=True)
            if record.bundle_dir:
                bundle_path = Path(record.bundle_dir)
                if bundle_path.is_dir():
                    shutil.rmtree(bundle_path, ignore_errors=True)
        self._save()
        return True

    # -- lookup ------------------------------------------------------------

    def get(self, user_id: str) -> AdapterRecord | None:
        return self._records.get(user_id)

    def require(self, user_id: str) -> AdapterRecord:
        record = self.get(user_id)
        if record is None:
            raise KeyError(f"No adapter registered for user_id={user_id!r}")
        return record

    def list_users(self) -> list[str]:
        return sorted(self._records.keys())

    def list_records(self) -> list[AdapterRecord]:
        return [self._records[uid] for uid in self.list_users()]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, user_id: object) -> bool:
        return user_id in self._records

    # -- usage tracking ----------------------------------------------------

    def mark_served(self, user_id: str) -> None:
        record = self._records.get(user_id)
        if record is None:
            return
        record.last_served_at = datetime.now()
        record.request_count += 1
        self._save()

    # -- persistence -------------------------------------------------------

    def _registry_path(self) -> Path:
        return self.root / REGISTRY_FILE

    def _load(self) -> None:
        path = self._registry_path()
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for entry in raw.get("records", []):
            try:
                record = AdapterRecord.model_validate(entry)
            except Exception:
                continue
            self._records[record.user_id] = record

    def _save(self) -> None:
        data = {
            "records": [r.model_dump(mode="json") for r in self._records.values()],
            "saved_at": datetime.now().isoformat(),
        }
        self._registry_path().write_text(json.dumps(data, indent=2, default=str))


def _remote_metadata(adapter_path: Path) -> dict[str, str]:
    remote_path = adapter_path / "remote.json"
    if not remote_path.is_file():
        return {}
    try:
        remote = json.loads(remote_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    provider = str(remote.get("provider") or "")
    if provider != "together":
        return {"provider": provider} if provider else {}
    metadata: dict[str, str] = {"provider": "together"}
    if remote.get("job_id"):
        metadata["together_job_id"] = str(remote["job_id"])
    if remote.get("base_model"):
        metadata["together_base_model"] = str(remote["base_model"])
    if remote.get("output_model"):
        metadata["together_output_model"] = str(remote["output_model"])
    return metadata
