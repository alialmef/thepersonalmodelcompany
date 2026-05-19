"""Data deletion + retrain orchestration.

LoRA adapters can't unlearn. From the analysis doc:

    "Deletion = retraining: If a user deletes data, you must retrain from
    scratch (no practical machine unlearning for LoRA)."

So this module's job is:
1. Delete the requested data from `UserStore`.
2. Record a tombstone so we know what was removed and when.
3. Invalidate the active model (clear the `ArtifactStore` active pointer)
   so the orchestrator knows a retrain is needed before this user is served
   the stale adapter again.
4. Write an audit event.

The actual retrain is scheduled by the orchestrator, which queries
`is_retrain_needed()` to discover users with pending tombstones.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from pmc.storage.artifact_store import ArtifactStore
from pmc.storage.audit import AuditLog
from pmc.storage.paths import StoragePaths
from pmc.storage.user_store import UserStore


class DeletionScope(StrEnum):
    """How much to delete in a single request."""

    SOURCES = "sources"        # delete listed source IDs, keep everything else
    ALL_DATA = "all_data"      # delete every raw source + curated dataset; keep bundles
    FULL = "full"              # nuke everything (raw + curated + bundles + audit)


class Tombstone(BaseModel):
    """A record of what was deleted, when, and whether retrain is pending."""

    requested_at: datetime = Field(default_factory=datetime.now)
    applied_at: datetime | None = None
    scope: DeletionScope
    sources: list[str] = Field(default_factory=list)
    raw_items_removed: int = 0
    datasets_removed: list[str] = Field(default_factory=list)
    notes: str = ""
    retrain_needed: bool = True


class DeletionResult(BaseModel):
    user_id: str
    scope: DeletionScope
    sources_deleted: list[str] = Field(default_factory=list)
    raw_items_removed: int = 0
    datasets_removed: list[str] = Field(default_factory=list)
    bundles_removed: int = 0
    active_cleared: bool = False
    retrain_needed: bool = True
    tombstone: Tombstone


class DeletionManager:
    """Carries out deletion requests and tracks pending retrains.

    Composes the three persistence stores so the orchestrator gets atomic-ish
    deletion (best-effort: we sequence the steps and audit each one).
    """

    def __init__(
        self,
        user_store: UserStore,
        artifact_store: ArtifactStore,
        audit_log: AuditLog,
    ) -> None:
        self.user_store = user_store
        self.artifact_store = artifact_store
        self.audit_log = audit_log
        self.paths = StoragePaths(user_store.paths.root)

    # -- main API ---------------------------------------------------------

    def delete(
        self,
        user_id: str,
        *,
        scope: DeletionScope = DeletionScope.ALL_DATA,
        sources: list[str] | None = None,
        notes: str = "",
    ) -> DeletionResult:
        """Apply a deletion. Returns a DeletionResult and writes a tombstone."""
        sources_to_delete: list[str]
        if scope == DeletionScope.SOURCES:
            sources_to_delete = sources or []
        else:
            sources_to_delete = self.user_store.list_sources(user_id)

        raw_removed = 0
        sources_deleted: list[str] = []
        for source_id in sources_to_delete:
            count = self.user_store.count_raw_items(user_id, source_id)
            if self.user_store.delete_source(user_id, source_id):
                sources_deleted.append(source_id)
                raw_removed += count

        datasets_removed: list[str] = []
        if scope in (DeletionScope.ALL_DATA, DeletionScope.FULL):
            for version in self.user_store.list_dataset_versions(user_id):
                if self.user_store.delete_dataset(user_id, version):
                    datasets_removed.append(version)

        active_cleared = self.artifact_store.clear_active(user_id)

        bundles_removed = 0
        if scope == DeletionScope.FULL:
            for run in self.artifact_store.list_runs(user_id):
                if self.artifact_store.delete_run(user_id, run.run_id):
                    bundles_removed += 1
            self.user_store.delete_user(user_id)
            self.audit_log.clear(user_id)
            self._delete_tombstone_file(user_id)

        tombstone = Tombstone(
            applied_at=datetime.now(),
            scope=scope,
            sources=sources_deleted,
            raw_items_removed=raw_removed,
            datasets_removed=datasets_removed,
            notes=notes,
            retrain_needed=scope != DeletionScope.FULL and (raw_removed > 0 or active_cleared),
        )
        if scope != DeletionScope.FULL:
            self._append_tombstone(user_id, tombstone)
            self.audit_log.log(
                user_id,
                stage="delete",
                event="deletion_applied",
                data={
                    "scope": scope.value,
                    "sources_deleted": sources_deleted,
                    "raw_items_removed": raw_removed,
                    "datasets_removed": datasets_removed,
                    "active_cleared": active_cleared,
                    "retrain_needed": tombstone.retrain_needed,
                    "notes": notes,
                },
            )

        return DeletionResult(
            user_id=user_id,
            scope=scope,
            sources_deleted=sources_deleted,
            raw_items_removed=raw_removed,
            datasets_removed=datasets_removed,
            bundles_removed=bundles_removed,
            active_cleared=active_cleared,
            retrain_needed=tombstone.retrain_needed,
            tombstone=tombstone,
        )

    # -- retrain status ---------------------------------------------------

    def is_retrain_needed(self, user_id: str) -> bool:
        """True if there's a pending tombstone marking retrain needed."""
        tombstones = self.list_tombstones(user_id)
        return any(t.retrain_needed for t in tombstones)

    def mark_retrained(self, user_id: str, *, run_id: str | None = None) -> int:
        """Clear retrain_needed on all open tombstones. Returns count cleared."""
        path = self.paths.tombstones_file(user_id)
        if not path.is_file():
            return 0
        tombstones = self.list_tombstones(user_id)
        cleared = 0
        for t in tombstones:
            if t.retrain_needed:
                t.retrain_needed = False
                cleared += 1
        if cleared:
            self._write_tombstones(user_id, tombstones)
            self.audit_log.log(
                user_id,
                stage="train",
                event="retrain_completed_after_deletion",
                run_id=run_id,
                data={"tombstones_cleared": cleared},
            )
        return cleared

    # -- tombstone file ---------------------------------------------------

    def list_tombstones(self, user_id: str) -> list[Tombstone]:
        path = self.paths.tombstones_file(user_id)
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return [Tombstone.model_validate(t) for t in raw.get("tombstones", [])]

    def _append_tombstone(self, user_id: str, tombstone: Tombstone) -> None:
        existing = self.list_tombstones(user_id)
        existing.append(tombstone)
        self._write_tombstones(user_id, existing)

    def _write_tombstones(self, user_id: str, tombstones: list[Tombstone]) -> None:
        path = self.paths.tombstones_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tombstones": [t.model_dump(mode="json") for t in tombstones],
            "saved_at": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    def _delete_tombstone_file(self, user_id: str) -> None:
        path = self.paths.tombstones_file(user_id)
        if path.is_file():
            path.unlink()


__all__ = [
    "DeletionManager",
    "DeletionResult",
    "DeletionScope",
    "Tombstone",
]
