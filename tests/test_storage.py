"""Tests for the storage layer: paths, user_store, artifact_store, audit, deletion."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pmc.ingest.base import RawItem
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
    SourceType,
)
from pmc.schema.user import StyleProfile, User
from pmc.storage import (
    ArtifactStore,
    AuditLog,
    DeletionManager,
    DeletionScope,
    StoragePaths,
    UserStore,
    new_run_id,
    safe_id,
)
from pmc.train.bundle import ArtifactBundle, BundleMetadata


# ---------- Helpers ----------


def _completion(prompt: str = "hi", response: str = "hello", source: SourceType | None = None) -> Completion:
    return Completion(
        conversation=Conversation(
            messages=[Message(role=Role.USER, content=prompt)],
            source_type=source,
        ),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content=response)])
        ],
    )


def _raw_item(source: SourceType, source_id: str, content: str = "x") -> RawItem:
    return RawItem(
        source_type=source,
        source_id=source_id,
        content=content,
        is_user=True,
    )


def _fake_adapter(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"], "base_model_name_or_path": "x"})
    )
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    return path


def _fake_bundle(tmp_path: Path, user_id: str = "u1") -> ArtifactBundle:
    adapter = _fake_adapter(tmp_path / "src_adapter")
    return ArtifactBundle(
        metadata=BundleMetadata(user_id=user_id, base_model="Qwen/Qwen3-8B", job_type="sft"),
        adapter_dir=adapter,
        style_profile=StyleProfile(tone_tags=["warm"]),
    )


# ---------- paths ----------


def test_safe_id_sanitizes():
    assert safe_id("hello world!") == "hello_world_"
    assert safe_id("../../etc/passwd") == ".._.._etc_passwd"
    assert "/" not in safe_id("../../etc/passwd")  # the property that matters
    assert safe_id("") == "_"
    assert safe_id("user@example.com") == "user_example.com"
    assert len(safe_id("x" * 500, max_len=10)) == 10


def test_storage_paths_isolated_per_user(tmp_path: Path):
    paths = StoragePaths(tmp_path)
    u1 = paths.user_root("alice")
    u2 = paths.user_root("bob")
    assert u1 != u2
    assert u1.parent == u2.parent


def test_storage_paths_are_pure(tmp_path: Path):
    """Path helpers compute paths but do NOT create them. Creation happens at
    write sites only — so a deleted user doesn't get silently recreated by
    a subsequent read or status check."""
    paths = StoragePaths(tmp_path)
    assert not paths.user_root("u").is_dir()
    assert not paths.raw_dir("u").is_dir()
    assert not paths.bundles_dir("u").is_dir()
    # The storage root itself is created so callers can write to subdirs
    assert paths.root.is_dir()


def test_storage_ensure_creates_dir(tmp_path: Path):
    paths = StoragePaths(tmp_path)
    target = paths.raw_dir("u")
    paths.ensure(target)
    assert target.is_dir()


def test_storage_paths_unsafe_ids_sanitized(tmp_path: Path):
    paths = StoragePaths(tmp_path)
    p1 = paths.raw_file("user", "../../etc/passwd")
    assert "../" not in str(p1)
    # path stays under user_root
    assert str(p1).startswith(str(paths.user_root("user")))


# ---------- UserStore: user profile ----------


def test_user_store_save_and_load_user(tmp_path: Path):
    store = UserStore(tmp_path)
    user = User(
        email="alex@example.com",
        name="Alex",
        style_profile=StyleProfile(tone_tags=["direct"]),
    )
    store.save_user(user)
    loaded = store.load_user(str(user.id))
    assert loaded is not None
    assert loaded.email == "alex@example.com"
    assert loaded.style_profile is not None
    assert "direct" in loaded.style_profile.tone_tags


def test_user_store_load_user_missing_returns_none(tmp_path: Path):
    store = UserStore(tmp_path)
    assert store.load_user("ghost") is None


# ---------- UserStore: raw items ----------


def test_user_store_save_and_load_raw_items(tmp_path: Path):
    store = UserStore(tmp_path)
    items = [_raw_item(SourceType.EMAIL, "email-1", f"body {i}") for i in range(3)]
    count = store.save_raw_items("alice", "gmail-export-jan", items)
    assert count == 3
    loaded = list(store.load_raw_items("alice", "gmail-export-jan"))
    assert len(loaded) == 3
    assert loaded[0].source_type == SourceType.EMAIL


def test_user_store_partitions_sources(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_raw_items("alice", "gmail", [_raw_item(SourceType.EMAIL, "g1")])
    store.save_raw_items("alice", "imessage", [_raw_item(SourceType.IMESSAGE, "i1")])
    sources = store.list_sources("alice")
    assert sources == ["gmail", "imessage"]
    assert store.count_raw_items("alice") == 2
    assert store.count_raw_items("alice", "gmail") == 1


def test_user_store_isolation_between_users(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_raw_items("alice", "gmail", [_raw_item(SourceType.EMAIL, "a")])
    store.save_raw_items("bob", "gmail", [_raw_item(SourceType.EMAIL, "b")])
    assert store.count_raw_items("alice") == 1
    assert store.count_raw_items("bob") == 1
    alice_items = list(store.load_raw_items("alice"))
    assert alice_items[0].source_id == "a"


def test_user_store_append_raw_items(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_raw_items("u", "src", [_raw_item(SourceType.EMAIL, "1")])
    store.save_raw_items("u", "src", [_raw_item(SourceType.EMAIL, "2")], append=True)
    assert store.count_raw_items("u", "src") == 2


def test_user_store_delete_source(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_raw_items("u", "gmail", [_raw_item(SourceType.EMAIL, "g")])
    store.save_raw_items("u", "imessage", [_raw_item(SourceType.IMESSAGE, "i")])
    assert store.delete_source("u", "gmail") is True
    assert store.delete_source("u", "gmail") is False
    assert store.list_sources("u") == ["imessage"]


# ---------- UserStore: curated datasets ----------


def test_user_store_save_and_load_curated_dataset(tmp_path: Path):
    store = UserStore(tmp_path)
    completions = [_completion(f"q{i}", f"r{i}", SourceType.EMAIL) for i in range(5)]
    holdout = [_completion("hq", "hr", SourceType.EMAIL)]

    manifest = store.save_curated_dataset("alice", "v1", completions, holdout)
    assert manifest.num_examples == 5
    assert manifest.source_breakdown.get("email") == 6
    assert manifest.checksum

    loaded_train = store.load_curated_dataset("alice", "v1")
    loaded_holdout = store.load_holdout("alice", "v1")
    assert len(loaded_train) == 5
    assert len(loaded_holdout) == 1

    loaded_manifest = store.load_manifest("alice", "v1")
    assert loaded_manifest is not None
    assert loaded_manifest.dataset_version == "v1"


def test_user_store_list_dataset_versions_excludes_holdouts(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_curated_dataset("u", "v1", [_completion()], [_completion()])
    store.save_curated_dataset("u", "v2", [_completion()])
    versions = store.list_dataset_versions("u")
    assert versions == ["v1", "v2"]


def test_user_store_delete_dataset(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_curated_dataset("u", "v1", [_completion()], [_completion()])
    assert store.delete_dataset("u", "v1") is True
    assert store.list_dataset_versions("u") == []
    assert store.load_manifest("u", "v1") is None


def test_user_store_delete_user(tmp_path: Path):
    store = UserStore(tmp_path)
    store.save_raw_items("u", "src", [_raw_item(SourceType.EMAIL, "x")])
    store.save_curated_dataset("u", "v1", [_completion()])
    assert store.delete_user("u") is True
    assert store.count_raw_items("u") == 0
    assert store.list_dataset_versions("u") == []


# ---------- ArtifactStore ----------


def test_new_run_id_format():
    rid = new_run_id()
    assert rid.startswith("run-")
    assert len(rid.split("-")) >= 4


def test_artifact_store_save_and_load(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    bundle = _fake_bundle(tmp_path)
    run_id = store.save_bundle("u1", bundle, run_id="run-test-1")

    assert run_id == "run-test-1"
    loaded = store.load_bundle("u1", run_id)
    assert loaded.metadata.user_id == "u1"
    assert loaded.style_profile is not None


def test_artifact_store_list_runs_newest_first(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.save_bundle("u", _fake_bundle(tmp_path / "b1"), run_id="run-20260101-aaaa")
    store.save_bundle("u", _fake_bundle(tmp_path / "b2"), run_id="run-20260201-bbbb")
    runs = store.list_runs("u")
    assert len(runs) == 2
    assert runs[0].run_id == "run-20260201-bbbb"


def test_artifact_store_active_pointer(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.save_bundle("u", _fake_bundle(tmp_path), run_id="run-A")
    assert store.get_active("u") is None

    pointer = store.set_active("u", "run-A", notes="first deploy")
    assert pointer.run_id == "run-A"
    assert store.get_active("u") is not None
    assert store.get_active("u").run_id == "run-A"


def test_artifact_store_load_active(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    bundle = _fake_bundle(tmp_path, user_id="u")
    store.save_bundle("u", bundle, run_id="run-active", promote_to_active=True)
    active = store.load_active("u")
    assert active is not None
    assert active.metadata.user_id == "u"


def test_artifact_store_promote_unknown_run_fails(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.set_active("u", "nonexistent-run")


def test_artifact_store_delete_run_clears_active_if_matched(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.save_bundle("u", _fake_bundle(tmp_path), run_id="run-X", promote_to_active=True)
    assert store.get_active("u") is not None

    store.delete_run("u", "run-X")
    assert store.get_active("u") is None
    assert store.list_runs("u") == []


def test_artifact_store_clear_active(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.save_bundle("u", _fake_bundle(tmp_path), run_id="r1", promote_to_active=True)
    assert store.clear_active("u") is True
    assert store.clear_active("u") is False
    assert store.get_active("u") is None


# ---------- AuditLog ----------


def test_audit_log_append_and_read(tmp_path: Path):
    log = AuditLog(tmp_path)
    log.log("u", "ingest", "loaded_items", data={"count": 100})
    log.log("u", "curate", "filtered", data={"kept": 80})

    events = log.events("u")
    assert len(events) == 2
    assert events[0].event == "loaded_items"
    assert events[1].data == {"kept": 80}


def test_audit_log_isolation_between_users(tmp_path: Path):
    log = AuditLog(tmp_path)
    log.log("alice", "ingest", "a-event")
    log.log("bob", "ingest", "b-event")

    assert len(log.events("alice")) == 1
    assert len(log.events("bob")) == 1
    assert log.events("alice")[0].event == "a-event"


def test_audit_log_filter_by_stage(tmp_path: Path):
    log = AuditLog(tmp_path)
    log.log("u", "ingest", "e1")
    log.log("u", "train", "e2")
    log.log("u", "ingest", "e3")
    events = log.events("u", stage="ingest")
    assert {e.event for e in events} == {"e1", "e3"}


def test_audit_log_filter_by_run_id(tmp_path: Path):
    log = AuditLog(tmp_path)
    log.log("u", "train", "started", run_id="run-1")
    log.log("u", "train", "started", run_id="run-2")
    log.log("u", "ingest", "loaded")

    by_run = log.events_for_run("u", "run-1")
    assert len(by_run) == 1
    assert by_run[0].run_id == "run-1"


def test_audit_log_filter_by_since(tmp_path: Path):
    log = AuditLog(tmp_path)
    log.log("u", "ingest", "old")
    time.sleep(0.01)
    cutoff = datetime.now()
    time.sleep(0.01)
    log.log("u", "ingest", "new")

    events = log.events("u", since=cutoff)
    assert len(events) == 1
    assert events[0].event == "new"


def test_audit_log_latest_returns_newest_first(tmp_path: Path):
    log = AuditLog(tmp_path)
    for i in range(5):
        log.log("u", "ingest", f"event-{i}")
    latest = log.latest("u", n=2)
    assert [e.event for e in latest] == ["event-4", "event-3"]


def test_audit_log_clear(tmp_path: Path):
    log = AuditLog(tmp_path)
    log.log("u", "ingest", "e1")
    assert log.clear("u") is True
    assert log.events("u") == []
    assert log.clear("u") is False


# ---------- DeletionManager ----------


def _make_deletion_setup(tmp_path: Path) -> tuple[DeletionManager, UserStore, ArtifactStore, AuditLog]:
    user_store = UserStore(tmp_path)
    artifact_store = ArtifactStore(tmp_path)
    audit_log = AuditLog(tmp_path)
    return DeletionManager(user_store, artifact_store, audit_log), user_store, artifact_store, audit_log


def test_deletion_specific_sources(tmp_path: Path):
    manager, store, _, _ = _make_deletion_setup(tmp_path)
    store.save_raw_items("u", "gmail", [_raw_item(SourceType.EMAIL, "g1"), _raw_item(SourceType.EMAIL, "g2")])
    store.save_raw_items("u", "imessage", [_raw_item(SourceType.IMESSAGE, "i1")])

    result = manager.delete("u", scope=DeletionScope.SOURCES, sources=["gmail"])

    assert result.sources_deleted == ["gmail"]
    assert result.raw_items_removed == 2
    assert "imessage" in store.list_sources("u")
    assert "gmail" not in store.list_sources("u")
    assert result.retrain_needed is True


def test_deletion_all_data_clears_datasets_and_active(tmp_path: Path):
    manager, store, artifacts, _ = _make_deletion_setup(tmp_path)
    store.save_raw_items("u", "gmail", [_raw_item(SourceType.EMAIL, "g")])
    store.save_curated_dataset("u", "v1", [_completion()], [_completion()])
    artifacts.save_bundle("u", _fake_bundle(tmp_path), run_id="r1", promote_to_active=True)

    result = manager.delete("u", scope=DeletionScope.ALL_DATA)

    assert "gmail" not in store.list_sources("u")
    assert store.list_dataset_versions("u") == []
    assert artifacts.get_active("u") is None
    # bundles are NOT removed in ALL_DATA scope
    assert len(artifacts.list_runs("u")) == 1
    assert result.active_cleared is True
    assert result.retrain_needed is True


def test_deletion_full_nukes_everything(tmp_path: Path):
    manager, store, artifacts, audit = _make_deletion_setup(tmp_path)
    store.save_raw_items("u", "gmail", [_raw_item(SourceType.EMAIL, "g")])
    artifacts.save_bundle("u", _fake_bundle(tmp_path), run_id="r1", promote_to_active=True)
    audit.log("u", "ingest", "loaded")

    result = manager.delete("u", scope=DeletionScope.FULL)

    assert result.bundles_removed == 1
    assert artifacts.list_runs("u") == []
    assert audit.events("u") == []
    assert manager.list_tombstones("u") == []
    assert result.retrain_needed is False


def test_deletion_writes_tombstone_and_audit(tmp_path: Path):
    manager, store, _, audit = _make_deletion_setup(tmp_path)
    store.save_raw_items("u", "gmail", [_raw_item(SourceType.EMAIL, "g")])

    manager.delete("u", scope=DeletionScope.SOURCES, sources=["gmail"], notes="user request")

    tombstones = manager.list_tombstones("u")
    assert len(tombstones) == 1
    assert tombstones[0].sources == ["gmail"]
    assert tombstones[0].retrain_needed is True
    assert tombstones[0].notes == "user request"

    events = audit.events("u", stage="delete")
    assert len(events) == 1
    assert events[0].event == "deletion_applied"


def test_deletion_retrain_needed_workflow(tmp_path: Path):
    manager, store, _, _ = _make_deletion_setup(tmp_path)
    store.save_raw_items("u", "src", [_raw_item(SourceType.EMAIL, "x")])
    manager.delete("u", scope=DeletionScope.SOURCES, sources=["src"])

    assert manager.is_retrain_needed("u") is True

    cleared = manager.mark_retrained("u", run_id="run-after")
    assert cleared == 1
    assert manager.is_retrain_needed("u") is False


def test_deletion_no_retrain_needed_if_nothing_changed(tmp_path: Path):
    manager, _, _, _ = _make_deletion_setup(tmp_path)
    # Nothing exists for this user, but we issue a SOURCES delete
    result = manager.delete("u", scope=DeletionScope.SOURCES, sources=["doesnt_exist"])
    assert result.raw_items_removed == 0
    assert result.retrain_needed is False


def test_deletion_multiple_tombstones_accumulate(tmp_path: Path):
    manager, store, _, _ = _make_deletion_setup(tmp_path)
    store.save_raw_items("u", "s1", [_raw_item(SourceType.EMAIL, "1")])
    store.save_raw_items("u", "s2", [_raw_item(SourceType.EMAIL, "2")])

    manager.delete("u", scope=DeletionScope.SOURCES, sources=["s1"])
    manager.delete("u", scope=DeletionScope.SOURCES, sources=["s2"])

    tombstones = manager.list_tombstones("u")
    assert len(tombstones) == 2


# ---------- Integration: storage ↔ ArtifactBundle ↔ AdapterRegistry ----------


def test_artifact_store_integrates_with_serve_registry(tmp_path: Path):
    """A bundle saved via ArtifactStore should be registrable in the serve AdapterRegistry."""
    from pmc.serve.registry import AdapterRegistry

    store = ArtifactStore(tmp_path / "artifacts")
    bundle = _fake_bundle(tmp_path, user_id="u")
    run_id = store.save_bundle("u", bundle, run_id="r1", promote_to_active=True)

    bundle_dir = store.paths.bundle_dir("u", run_id)
    registry = AdapterRegistry(tmp_path / "registry")
    record = registry.register_bundle(bundle_dir)
    assert record.user_id == "u"
    assert record.bundle_dir == str(bundle_dir.resolve())


def test_full_lifecycle_smoke(tmp_path: Path):
    """End-to-end smoke: ingest raw → curate → save → register → mark served → delete."""
    user_store = UserStore(tmp_path)
    artifact_store = ArtifactStore(tmp_path)
    audit_log = AuditLog(tmp_path)
    deletion = DeletionManager(user_store, artifact_store, audit_log)

    # 1. Ingest: persist raw items
    user_store.save_raw_items("u", "gmail-jan", [_raw_item(SourceType.EMAIL, f"e{i}", f"body {i}") for i in range(5)])
    audit_log.log("u", "ingest", "loaded", data={"items": 5})

    # 2. Curate: persist a dataset version
    completions = [_completion(f"q{i}", f"r{i}", SourceType.EMAIL) for i in range(5)]
    user_store.save_curated_dataset("u", "v1", completions)
    audit_log.log("u", "curate", "saved_dataset", data={"version": "v1", "n": 5})

    # 3. Train: persist a bundle
    bundle = _fake_bundle(tmp_path)
    run_id = artifact_store.save_bundle("u", bundle, promote_to_active=True)
    audit_log.log("u", "train", "completed", run_id=run_id)

    # 4. Delete: user removes a source → retrain needed
    deletion.delete("u", scope=DeletionScope.SOURCES, sources=["gmail-jan"])
    assert deletion.is_retrain_needed("u") is True
    assert artifact_store.get_active("u") is None

    # 5. Audit shows the full timeline
    events = audit_log.events("u")
    assert [e.stage for e in events] == ["ingest", "curate", "train", "delete"]
    assert events[-1].data["sources_deleted"] == ["gmail-jan"]
