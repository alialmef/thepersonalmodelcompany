"""Per-user ArtifactBundle history + active-bundle pointer.

A user accumulates one ArtifactBundle per training run. The "active" pointer
records which run is currently serving — used to wire up the serving registry
and to know which adapter to swap out when a new one is promoted.

This sits on top of `pmc/train/bundle.py` (which knows how to read/write a
single bundle). The store adds per-user listing, run-id management, and the
active pointer.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pmc.storage.paths import StoragePaths, safe_id
from pmc.train.bundle import ArtifactBundle


def new_run_id(prefix: str = "run") -> str:
    """Generate a sortable run_id: prefix-YYYYMMDDhhmmss-shortuuid."""
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{now}-{uuid.uuid4().hex[:8]}"


class ActivePointer(BaseModel):
    """The contents of active.json — which run_id is currently serving."""

    run_id: str
    promoted_at: datetime = Field(default_factory=datetime.now)
    notes: str = ""


class RunSummary(BaseModel):
    """Lightweight summary for listing — avoids loading the full bundle."""

    run_id: str
    user_id: str
    base_model: str
    job_type: str
    created_at: datetime
    bundle_dir: str


class ArtifactStore:
    """Per-user ArtifactBundle persistence and active-version management."""

    def __init__(self, root: Path | str) -> None:
        self.paths = StoragePaths(root)

    # -- write ------------------------------------------------------------

    def save_bundle(
        self,
        user_id: str,
        bundle: ArtifactBundle,
        *,
        run_id: str | None = None,
        promote_to_active: bool = False,
        copy_adapter: bool = True,
    ) -> str:
        """Persist a bundle under the user's directory. Returns the run_id.

        `copy_adapter=False` when the adapter has already been written into
        the target bundle directory (e.g. by a trainer writing in place).
        """
        run_id = run_id or new_run_id()
        target = self.paths.bundle_dir(user_id, run_id)
        target.mkdir(parents=True, exist_ok=True)
        bundle.write(target, copy_adapter=copy_adapter)

        if promote_to_active:
            self.set_active(user_id, run_id)
        return run_id

    # -- read -------------------------------------------------------------

    def load_bundle(self, user_id: str, run_id: str) -> ArtifactBundle:
        path = self.paths.bundle_dir(user_id, run_id)
        if not path.is_dir():
            raise FileNotFoundError(f"No bundle at {path}")
        return ArtifactBundle.load(path)

    def load_active(self, user_id: str) -> ArtifactBundle | None:
        pointer = self.get_active(user_id)
        if pointer is None:
            return None
        try:
            return self.load_bundle(user_id, pointer.run_id)
        except FileNotFoundError:
            return None

    def list_runs(self, user_id: str) -> list[RunSummary]:
        """All training runs for this user, newest first."""
        bundles_dir = self.paths.bundles_dir(user_id)
        if not bundles_dir.is_dir():
            return []
        summaries: list[RunSummary] = []
        for d in sorted(bundles_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "bundle.json"
            if not meta_path.is_file():
                continue
            try:
                data = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            summaries.append(
                RunSummary(
                    run_id=d.name,
                    user_id=data.get("user_id", user_id),
                    base_model=data.get("base_model", "unknown"),
                    job_type=data.get("job_type", "unknown"),
                    created_at=datetime.fromisoformat(data["created_at"])
                    if "created_at" in data
                    else datetime.fromtimestamp(d.stat().st_mtime),
                    bundle_dir=str(d),
                )
            )
        return summaries

    # -- active pointer ---------------------------------------------------

    def set_active(self, user_id: str, run_id: str, *, notes: str = "") -> ActivePointer:
        """Promote a run to be the active model for this user."""
        if not self.paths.bundle_dir(user_id, run_id).is_dir():
            raise FileNotFoundError(f"Cannot promote unknown run_id={run_id!r}")
        pointer = ActivePointer(run_id=safe_id(run_id), notes=notes)
        ptr_path = self.paths.active_pointer_file(user_id)
        ptr_path.parent.mkdir(parents=True, exist_ok=True)
        ptr_path.write_text(pointer.model_dump_json(indent=2))
        return pointer

    def get_active(self, user_id: str) -> ActivePointer | None:
        path = self.paths.active_pointer_file(user_id)
        if not path.is_file():
            return None
        try:
            return ActivePointer.model_validate_json(path.read_text())
        except Exception:
            return None

    def clear_active(self, user_id: str) -> bool:
        """Used when a model is invalidated (e.g. data deleted, retrain pending)."""
        path = self.paths.active_pointer_file(user_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # -- delete -----------------------------------------------------------

    def delete_run(self, user_id: str, run_id: str) -> bool:
        """Remove one training run. If it was active, clear the active pointer."""
        path = self.paths.bundle_dir(user_id, run_id)
        if not path.is_dir():
            return False
        active = self.get_active(user_id)
        if active and active.run_id == safe_id(run_id):
            self.clear_active(user_id)
        shutil.rmtree(path, ignore_errors=True)
        return True


__all__ = ["ActivePointer", "ArtifactStore", "RunSummary", "new_run_id"]
