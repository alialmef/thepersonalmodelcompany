"""Monitor — per-user and system-wide status views.

Aggregates state from every layer (UserStore, ArtifactStore, AuditLog,
DeletionManager, AdapterRegistry) into a single read-only view the web app
or CLI can present.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from pmc.serve.registry import AdapterRegistry
from pmc.storage.artifact_store import ArtifactStore
from pmc.storage.audit import AuditEvent, AuditLog
from pmc.storage.deletion import DeletionManager
from pmc.storage.graph_store import GraphStore, NODE_KINDS
from pmc.storage.user_store import UserStore


class UserStatus(BaseModel):
    """Everything visible about one user at a moment in time."""

    user_id: str
    has_profile: bool = False
    total_runs: int = 0
    active_run_id: str | None = None
    last_training_at: datetime | None = None
    last_eval_scores: dict[str, float] = Field(default_factory=dict)
    retrain_needed: bool = False
    pending_tombstones: int = 0
    raw_sources: list[str] = Field(default_factory=list)
    raw_item_count: int = 0
    # Per-source item counts so the UI can render a live scoreboard
    # ("Messages 12,431 · Notes 47"). Each entry is {source_id, kind, item_count}.
    raw_source_breakdown: list[dict] = Field(default_factory=list)
    # Graph entity counts produced by the Rust extractors. Read from the
    # Tauri-local graph store (which the backend can also reach when the
    # storage_root is shared, e.g. local dev pointing at
    # ~/.pmc-dev/storage). Keys are entity kinds: person, place, theme, etc.
    graph_entity_counts: dict[str, int] = Field(default_factory=dict)
    graph_node_total: int = 0
    dataset_versions: list[str] = Field(default_factory=list)
    registered_for_serving: bool = False
    recent_events: list[AuditEvent] = Field(default_factory=list)


class SystemStatus(BaseModel):
    total_users: int = 0
    total_runs: int = 0
    deployed_users: int = 0
    users_needing_retrain: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class Monitor:
    """Read-only aggregator across all storage layers."""

    def __init__(
        self,
        user_store: UserStore,
        artifact_store: ArtifactStore,
        audit_log: AuditLog,
        *,
        deletion: DeletionManager | None = None,
        registry: AdapterRegistry | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.user_store = user_store
        self.artifact_store = artifact_store
        self.audit_log = audit_log
        self.deletion = deletion
        self.registry = registry
        # Optional: when the backend shares its storage_root with the
        # Tauri-local graph path, this Monitor surfaces graph counts in
        # /status so the UI tells the truth about what's been structured.
        self.graph_store = graph_store

    def user_status(self, user_id: str, *, recent_events: int = 10) -> UserStatus:
        status = UserStatus(user_id=user_id)
        status.has_profile = self.user_store.load_user(user_id) is not None
        status.raw_sources = self.user_store.list_sources(user_id)
        status.raw_item_count = self.user_store.count_raw_items(user_id)
        # Per-source counts. Infer kind from the source_id prefix; the native
        # ingesters all use a "<kind>-YYYYMMDD-HHMMSS" naming convention.
        breakdown = []
        for src in status.raw_sources:
            kind = src.split("-", 1)[0] if "-" in src else src
            try:
                count = self.user_store.count_raw_items(user_id, src)
            except Exception:
                count = 0
            breakdown.append({"source_id": src, "kind": kind, "item_count": count})
        status.raw_source_breakdown = breakdown
        status.dataset_versions = self.user_store.list_dataset_versions(user_id)

        runs = self.artifact_store.list_runs(user_id)
        status.total_runs = len(runs)
        if runs:
            status.last_training_at = runs[0].created_at
        active = self.artifact_store.get_active(user_id)
        if active is not None:
            status.active_run_id = active.run_id
            try:
                bundle = self.artifact_store.load_bundle(user_id, active.run_id)
                status.last_eval_scores = {
                    k: v for k, v in (bundle.eval_report or {}).items()
                    if isinstance(v, (int, float))
                }
            except FileNotFoundError:
                pass

        if self.deletion is not None:
            tombstones = self.deletion.list_tombstones(user_id)
            status.pending_tombstones = sum(1 for t in tombstones if t.retrain_needed)
            status.retrain_needed = self.deletion.is_retrain_needed(user_id)

        if self.registry is not None:
            status.registered_for_serving = user_id in self.registry

        # Graph counts (the load-bearing add — what the Rust extractors
        # actually produced, surfaced to the UI/agent for the first time).
        if self.graph_store is not None and self.graph_store.exists(user_id):
            counts = self.graph_store.counts(user_id)
            status.graph_entity_counts = counts
            status.graph_node_total = sum(counts.get(k, 0) for k in NODE_KINDS)

        status.recent_events = self.audit_log.latest(user_id, n=recent_events)
        return status

    def list_users(self) -> list[str]:
        """All user IDs discoverable from any layer (storage union registry)."""
        users: set[str] = set()
        users_root = self.user_store.paths.root / "users"
        if users_root.is_dir():
            for p in users_root.iterdir():
                if p.is_dir():
                    users.add(p.name)
        if self.registry is not None:
            users.update(self.registry.list_users())
        return sorted(users)

    def system_status(self) -> SystemStatus:
        all_users = self.list_users()
        total_runs = 0
        deployed = 0
        retrain = 0
        for uid in all_users:
            total_runs += len(self.artifact_store.list_runs(uid))
            if self.artifact_store.get_active(uid) is not None:
                deployed += 1
            if self.deletion is not None and self.deletion.is_retrain_needed(uid):
                retrain += 1
        return SystemStatus(
            total_users=len(all_users),
            total_runs=total_runs,
            deployed_users=deployed,
            users_needing_retrain=retrain,
        )

    def list_pending_retrains(self) -> list[str]:
        """Users with at least one open tombstone marking retrain needed."""
        if self.deletion is None:
            return []
        return [uid for uid in self.list_users() if self.deletion.is_retrain_needed(uid)]


__all__ = ["Monitor", "SystemStatus", "UserStatus"]
