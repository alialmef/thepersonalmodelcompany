"""Tests for the orchestrator layer: data sources, pipeline stages, scheduler, monitor, CLI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pmc.ingest.base import RawItem
from pmc.orchestrator import (
    DataSource,
    DataSourceKind,
    JobScheduler,
    JobStatus,
    Monitor,
    PMCPipeline,
    PipelineConfig,
    raw_source,
    text_source,
)
from pmc.orchestrator.cli import build_parser, main
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
    SourceType,
)
from pmc.schema.training import TrainingConfig
from pmc.schema.verification import (
    ActionDecision,
    ActionTrace,
    CandidateOrigin,
    JudgmentVerdict,
    PersonalProbe,
    ProbeCandidate,
    ProbeKind,
    UserJudgment,
)
from pmc.serve.registry import AdapterRegistry
from pmc.storage.artifact_store import ArtifactStore
from pmc.storage.audit import AuditLog
from pmc.storage.deletion import DeletionManager, DeletionScope
from pmc.storage.user_store import UserStore
from pmc.storage.verification_store import VerificationStore
from pmc.train.config import SFTRunResult


# ---------- Test fixtures: a fake training function & generator factory ----------


def _fake_adapter(path: Path) -> Path:
    """Create a minimal LoRA adapter directory on disk."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "Qwen/Qwen3-8B",
        })
    )
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    return path


def fake_train_fn(
    config: TrainingConfig,
    train: list[Completion],
    output_dir: Path,
    eval_completions: list[Completion] | None,
) -> SFTRunResult:
    _fake_adapter(output_dir)
    return SFTRunResult(
        user_id=config.user_id,
        base_model=config.base_model,
        adapter_dir=output_dir,
        num_train_examples=len(train),
        num_eval_examples=len(eval_completions) if eval_completions else 0,
        final_train_loss=0.5,
        final_eval_loss=0.6,
        elapsed_seconds=0.01,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        config=config,
    )


def fake_generator_factory(base_model: str, adapter_dir):
    from pmc.eval.generator import MockGenerator
    name = f"hf:{adapter_dir}" if adapter_dir else f"hf:{base_model}"
    return MockGenerator(default="generated response", name=name)


# ---------- Helpers ----------


def _make_pipeline(
    tmp_path: Path,
    *,
    with_registry: bool = False,
) -> tuple[PMCPipeline, UserStore, ArtifactStore, AuditLog, DeletionManager, AdapterRegistry | None]:
    user_store = UserStore(tmp_path / "storage")
    artifact_store = ArtifactStore(tmp_path / "storage")
    audit_log = AuditLog(tmp_path / "storage")
    deletion = DeletionManager(user_store, artifact_store, audit_log)
    registry = AdapterRegistry(tmp_path / "registry") if with_registry else None
    pipeline = PMCPipeline(
        user_store=user_store,
        artifact_store=artifact_store,
        audit_log=audit_log,
        deletion=deletion,
        registry=registry,
        train_fn=fake_train_fn,
        generator_factory=fake_generator_factory,
    )
    return pipeline, user_store, artifact_store, audit_log, deletion, registry


_DISTINCT_TOPICS = [
    "Shipping the launch next quarter means we have to settle the latency story now.",
    "Spent the morning rereading my notes on consumer pricing. Tiered makes more sense than per-seat.",
    "Coffee chat with Rana was useful. She thinks the onboarding flow is where we're bleeding.",
    "Quick reflection on the parents' visit: I underestimated how much I miss the routine.",
    "If we go with Qwen as the base, we need to plan for the tokenizer differences upfront.",
    "Reading habits this month: short essays beat full books for me right now. Less guilt.",
    "Hiring update: passed on the second-round candidate. Cultural mismatch outweighed the skills.",
    "Travel plan for May is set. Sticking to two weeks because three burns me out every time.",
    "Notes on the demo: cut the third slide. Open with the agent doing one specific thing well.",
    "Workout block worked. Eight weeks of three-times-a-week was enough to feel different.",
    "Thinking about the partnership pitch: lead with the integration, not the model architecture.",
    "Reread the old journal entries from January. The anxieties were real but the predictions were wrong.",
    "Saturday lesson: blocking calendars by category beats blocking by project for my style.",
    "Reviewed the marketing draft. Too many adjectives. The voice should be drier, more specific.",
    "Long walk through the park. Decided to push the trip back a month and finish the rewrite first.",
    "Lessons from the bad customer call: I lead with empathy too late. Frame the apology first.",
    "Looking at the cap table proposal — the dilution math doesn't pencil out under our current burn.",
    "Personal note: cut down on late-night reading. It's wrecking the next morning's focus.",
    "Watched the documentary on the artisanal bread movement. Felt seen in an unexpected way.",
    "Email to Mom drafted. Three short paragraphs is the right length for what I actually mean.",
    "Outline for the talk: open with the failure mode, then the principle, then the example.",
    "Friend's wedding next month. Wrote the toast on the train. Six sentences, no inside jokes.",
    "Coding session got derailed by a config quirk. Documented it so next time isn't a surprise.",
    "Realized I keep deferring the dental appointment. Booked it for next Tuesday. Done.",
    "Long talk with the team about ownership. The right unit is the outcome, not the artifact.",
]


def _good_raw_items(count: int = 20) -> list[RawItem]:
    """Items long and varied enough to pass curate's quality/dedup thresholds."""
    return [
        RawItem(
            source_type=SourceType.NOTES,
            source_id=f"note-{i:03d}",
            content=_DISTINCT_TOPICS[i % len(_DISTINCT_TOPICS)] + (
                f" (entry {i}, recorded for personal record-keeping.)"
            ),
            is_user=True,
        )
        for i in range(count)
    ]


# ---------- DataSource ----------


def test_text_source_factory(tmp_path: Path):
    f = tmp_path / "n.md"
    f.write_text("hello")
    src = text_source(f)
    items = list(src.ingest())
    assert len(items) == 1
    assert src.kind == DataSourceKind.TEXT


def test_raw_source_passthrough():
    items = _good_raw_items(3)
    src = raw_source(items, source_id="my-raw")
    out = list(src.ingest())
    assert len(out) == 3
    assert src.derived_source_id() == "my-raw"


def test_data_source_validates_path_required():
    with pytest.raises(ValueError):
        DataSource(kind=DataSourceKind.TEXT)


def test_data_source_validates_raw_requires_items():
    with pytest.raises(ValueError):
        DataSource(kind=DataSourceKind.RAW, items=[])


def test_data_source_mbox_requires_user_emails(tmp_path: Path):
    f = tmp_path / "x.mbox"
    f.write_text("")
    with pytest.raises(ValueError):
        DataSource(kind=DataSourceKind.EMAIL_MBOX, path=f)


def test_data_source_derived_id_from_path(tmp_path: Path):
    src = text_source(tmp_path / "my-notes.md")
    assert "my-notes" in src.derived_source_id()


# ---------- Pipeline stages ----------


def test_stage_ingest_persists_raw_items(tmp_path: Path):
    pipeline, store, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(5))],
    )
    count = pipeline.stage_ingest(config)
    assert count == 5
    assert store.count_raw_items("alice") == 5


def test_stage_curate_produces_dataset(tmp_path: Path):
    pipeline, store, _, audit, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(15))],
    )
    pipeline.stage_ingest(config)
    result, version = pipeline.stage_curate(config)
    assert version.startswith("v")
    assert result.stats.output_completions > 0
    # Dataset persisted
    loaded = store.load_curated_dataset("alice", version)
    assert len(loaded) == result.stats.output_completions
    # Audit event written
    curate_events = audit.events("alice", stage="curate")
    assert len(curate_events) == 1
    assert curate_events[0].event == "curate_completed"


def test_stage_train_writes_adapter_and_audits(tmp_path: Path):
    pipeline, *_, audit, _, _ = _make_pipeline(tmp_path)
    train = [
        Completion(
            conversation=Conversation(messages=[Message(role=Role.USER, content=f"Q{i}")]),
            candidates=[CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content=f"A{i} much longer answer")])],
        )
        for i in range(15)
    ]
    config = PipelineConfig(user_id="alice", data_sources=[])
    plan, result = pipeline.stage_train(config, train, [])
    assert result is not None
    assert plan.estimated_steps > 0
    assert result.adapter_dir.is_dir()
    assert (result.adapter_dir / "adapter_config.json").is_file()

    train_events = audit.events("alice", stage="train")
    event_names = {e.event for e in train_events}
    assert "training_planned" in event_names
    assert "training_started" in event_names
    assert "training_completed" in event_names


def test_stage_train_respects_dry_run(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    train = [
        Completion(
            conversation=Conversation(messages=[Message(role=Role.USER, content="q")]),
            candidates=[CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="some response")])],
        )
        for _ in range(15)
    ]
    config = PipelineConfig(user_id="alice", dry_run=True, data_sources=[])
    plan, result = pipeline.stage_train(config, train, [])
    assert result is None
    assert plan is not None
    assert plan.estimated_steps > 0


def test_stage_train_smart_epochs(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    small = [
        Completion(
            conversation=Conversation(messages=[Message(role=Role.USER, content="q")]),
            candidates=[CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="resp")])],
        )
    ] * 100
    config = PipelineConfig(user_id="u", dry_run=True, data_sources=[])
    plan_small, _ = pipeline.stage_train(config, small, [])
    # Effective batch size = 16, 100 examples, 3 epochs (smart) → ceil(100/16)*3 = 21
    assert plan_small.estimated_steps >= 18


def test_stage_train_includes_verification_feedback(tmp_path: Path):
    pipeline, store, *_rest, audit, _deletion, _registry = _make_pipeline(tmp_path)
    verification = VerificationStore(store.paths.root)
    probe = PersonalProbe(
        user_id="alice",
        kind=ProbeKind.VOICE,
        prompt=[Message(role=Role.USER, content="free thursday?")],
        candidates=[
            ProbeCandidate(
                id="cand-model",
                origin=CandidateOrigin.PERSONAL_MODEL,
                text="Sounds good, Thursday works.",
            )
        ],
    )
    verification.save_probes("alice", [probe])
    verification.append_judgment(
        "alice",
        UserJudgment(
            user_id="alice",
            probe_id=probe.id,
            verdict=JudgmentVerdict.EDIT,
            chosen_candidate_id="cand-model",
            edited_text="yeah thursday works",
            dimension="voice",
        ),
    )
    verification.append_action_trace(
        "alice",
        ActionTrace(
            user_id="alice",
            surface="messages",
            operation="draft_reply",
            proposed_text="yeah thursday works",
            decision=ActionDecision.APPROVED,
        ),
    )

    train = [
        Completion(
            conversation=Conversation(messages=[Message(role=Role.USER, content=f"Q{i}")]),
            candidates=[
                CompletionCandidate(
                    messages=[Message(role=Role.ASSISTANT, content=f"A{i} much longer answer")]
                )
            ],
        )
        for i in range(15)
    ]
    config = PipelineConfig(user_id="alice", data_sources=[])
    _plan, result = pipeline.stage_train(config, train, [])

    assert result is not None
    assert result.num_train_examples == 17
    events = audit.events("alice", stage="train")
    attached = [e for e in events if e.event == "verification_signal_attached"]
    assert attached
    assert attached[0].data["total_completions"] == 2


def test_pipeline_persists_holdout_split_for_private_eval(tmp_path: Path):
    pipeline, store, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(25))],
        holdout_fraction=0.2,
        skip_eval=True,
        skip_deploy=True,
    )
    result = pipeline.run(config)

    assert result.status == "completed"
    assert result.dataset_version is not None
    holdout = store.load_holdout("alice", result.dataset_version)
    train = store.load_curated_dataset("alice", result.dataset_version)
    assert len(holdout) > 0
    assert len(train) + len(holdout) == result.completions_curated


def test_verification_gate_blocks_unproven_model(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(user_id="alice", require_verification_to_deploy=True)

    decision = pipeline.stage_verification_gate(config)

    assert decision.deploy is False
    assert "personal verification" in decision.reason


def test_pipeline_blocks_deploy_when_private_verification_missing(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(20))],
        skip_eval=True,
        require_verification_to_deploy=True,
    )
    result = pipeline.run(config)

    assert result.status == "blocked"
    assert result.deployed is False
    assert "personal verification" in result.notes


# ---------- Full pipeline.run() ----------


def test_pipeline_full_run_end_to_end(tmp_path: Path):
    pipeline, _, artifacts, audit, _, _ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        user_name="Alice",
        user_email="alice@example.com",
        data_sources=[raw_source(_good_raw_items(20))],
        skip_eval=True,  # default benchmarks need raw text data we don't supply here
    )
    result = pipeline.run(config)
    assert result.status == "completed"
    assert result.run_id is not None
    assert result.training_result is not None
    assert result.deployed is True

    # Audit shows the full timeline
    events = audit.events("alice")
    stages_seen = {e.stage for e in events}
    assert {"ingest", "curate", "train", "deploy"}.issubset(stages_seen)

    # Bundle was saved and promoted to active
    assert artifacts.get_active("alice") is not None
    assert artifacts.get_active("alice").run_id == result.run_id

    bundle = artifacts.load_bundle("alice", result.run_id)
    assert bundle.metadata.user_id == "alice"
    assert bundle.metadata.user_name == "Alice"
    assert bundle.style_profile is not None


def test_pipeline_dry_run_short_circuits(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(15))],
        dry_run=True,
    )
    result = pipeline.run(config)
    assert result.status == "dry_run"
    assert result.training_plan is not None
    assert result.training_result is None
    assert result.deployed is False


def test_pipeline_no_data_status(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(user_id="empty", data_sources=[])
    result = pipeline.run(config)
    assert result.status == "no_data"
    assert result.raw_items_ingested == 0


def test_pipeline_min_examples_to_train(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(2))],
        min_examples_to_train=10,
    )
    result = pipeline.run(config)
    assert result.status == "no_data"
    assert "min" in result.notes.lower()


def test_pipeline_registers_with_serve_registry(tmp_path: Path):
    pipeline, *_, registry = _make_pipeline(tmp_path, with_registry=True)
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(15))],
        skip_eval=True,
    )
    result = pipeline.run(config)
    assert result.deployed is True
    assert registry is not None
    assert "alice" in registry


def test_pipeline_blocked_by_gate_does_not_deploy(tmp_path: Path):
    """If eval fails the gate, status=blocked and the model is not promoted."""
    pipeline, *_, registry = _make_pipeline(tmp_path, with_registry=True)

    # Build a gate that requires a benchmark we'll never produce
    from pmc.eval.gate import EvalGateConfig
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(15))],
        gate_config=EvalGateConfig(
            thresholds={"style_match": 0.99},
            required=["style_match"],
            treat_missing_as="fail",
        ),
    )
    result = pipeline.run(config)
    assert result.status == "blocked"
    assert result.gate_passed is False
    assert result.deployed is False
    assert registry is not None
    assert "alice" not in registry


def test_pipeline_clears_retrain_flag_after_successful_deploy(tmp_path: Path):
    pipeline, store, *_, deletion, registry = _make_pipeline(tmp_path, with_registry=True)
    # Set up: ingest, deploy a model
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(15))],
        skip_eval=True,
    )
    pipeline.run(config)

    # User deletes a source → retrain needed
    deletion.delete("alice", scope=DeletionScope.SOURCES, sources=["raw"])
    assert deletion.is_retrain_needed("alice") is True

    # Re-ingest fresh data and run again
    store.save_raw_items("alice", "fresh", _good_raw_items(15))
    result = pipeline.run(
        PipelineConfig(
            user_id="alice",
            data_sources=[],  # use already-stored raw items
            skip_eval=True,
        )
    )
    assert result.status == "completed"
    assert deletion.is_retrain_needed("alice") is False


def test_pipeline_skip_deploy(tmp_path: Path):
    pipeline, _, artifacts, *_ = _make_pipeline(tmp_path)
    config = PipelineConfig(
        user_id="u",
        data_sources=[raw_source(_good_raw_items(15))],
        skip_eval=True,
        skip_deploy=True,
    )
    result = pipeline.run(config)
    assert result.deployed is False
    assert artifacts.get_active("u") is None


def test_pipeline_handles_train_exception(tmp_path: Path):
    pipeline, _, _, audit, _, _ = _make_pipeline(tmp_path)
    pipeline._train_fn = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA OOM"))
    config = PipelineConfig(
        user_id="alice",
        data_sources=[raw_source(_good_raw_items(15))],
    )
    result = pipeline.run(config)
    assert result.status == "failed"
    assert "CUDA OOM" in result.error
    # Error was audited
    error_events = audit.events("alice", event="pipeline_error")
    assert len(error_events) == 1


# ---------- JobScheduler ----------


def test_scheduler_submit_and_wait(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    scheduler = JobScheduler(pipeline, max_workers=2)
    try:
        config = PipelineConfig(
            user_id="alice",
            data_sources=[raw_source(_good_raw_items(15))],
            skip_eval=True,
        )
        job = scheduler.submit(config)
        # Job may already be RUNNING by the time we check (executor races us);
        # the contract is just that it's not yet in a terminal state.
        assert job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
        assert job.user_id == "alice"

        final = scheduler.wait(job.id, timeout=10)
        assert final.status == JobStatus.COMPLETED
        assert final.result is not None
        assert final.result.deployed is True
    finally:
        scheduler.shutdown(wait=True)


def test_scheduler_list_jobs_filters(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    scheduler = JobScheduler(pipeline, max_workers=1)
    try:
        for uid in ["alice", "alice", "bob"]:
            scheduler.submit(PipelineConfig(
                user_id=uid,
                data_sources=[raw_source(_good_raw_items(15))],
                skip_eval=True,
            ))
        # Wait for everything
        for j in scheduler.list_jobs():
            scheduler.wait(j.id, timeout=10)

        alice_jobs = scheduler.list_jobs(user_id="alice")
        assert len(alice_jobs) == 2
        bob_jobs = scheduler.list_jobs(user_id="bob")
        assert len(bob_jobs) == 1
    finally:
        scheduler.shutdown(wait=True)


def test_scheduler_get_unknown_returns_none(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    scheduler = JobScheduler(pipeline)
    try:
        assert scheduler.get("nope") is None
    finally:
        scheduler.shutdown()


def test_scheduler_failed_pipeline_marks_job_failed(tmp_path: Path):
    pipeline, *_ = _make_pipeline(tmp_path)
    pipeline._train_fn = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    scheduler = JobScheduler(pipeline)
    try:
        job = scheduler.submit(PipelineConfig(
            user_id="u",
            data_sources=[raw_source(_good_raw_items(15))],
        ))
        final = scheduler.wait(job.id, timeout=10)
        assert final.status == JobStatus.FAILED
    finally:
        scheduler.shutdown()


# ---------- Monitor ----------


def test_monitor_user_status_after_full_run(tmp_path: Path):
    pipeline, _, _, _, deletion, registry = _make_pipeline(tmp_path, with_registry=True)
    pipeline.run(
        PipelineConfig(
            user_id="alice",
            user_name="Alice",
            user_email="alice@example.com",
            data_sources=[raw_source(_good_raw_items(15))],
            skip_eval=True,
        )
    )
    monitor = Monitor(
        pipeline.user_store,
        pipeline.artifact_store,
        pipeline.audit_log,
        deletion=deletion,
        registry=registry,
    )
    status = monitor.user_status("alice")
    assert status.has_profile is True
    assert status.total_runs == 1
    assert status.active_run_id is not None
    assert status.raw_item_count == 15
    assert "raw" in status.raw_sources
    assert len(status.dataset_versions) == 1
    assert status.registered_for_serving is True
    assert status.retrain_needed is False
    assert len(status.recent_events) > 0


def test_monitor_system_status(tmp_path: Path):
    pipeline, *_, deletion, registry = _make_pipeline(tmp_path, with_registry=True)
    for uid in ["alice", "bob"]:
        pipeline.run(
            PipelineConfig(
                user_id=uid,
                data_sources=[raw_source(_good_raw_items(15))],
                skip_eval=True,
            )
        )
    monitor = Monitor(
        pipeline.user_store, pipeline.artifact_store, pipeline.audit_log,
        deletion=deletion, registry=registry,
    )
    system = monitor.system_status()
    assert system.total_users == 2
    assert system.total_runs == 2
    assert system.deployed_users == 2
    assert system.users_needing_retrain == 0


def test_monitor_pending_retrains(tmp_path: Path):
    pipeline, _, _, _, deletion, _ = _make_pipeline(tmp_path)
    pipeline.run(
        PipelineConfig(
            user_id="alice",
            data_sources=[raw_source(_good_raw_items(15))],
            skip_eval=True,
        )
    )
    deletion.delete("alice", scope=DeletionScope.SOURCES, sources=["raw"])
    monitor = Monitor(
        pipeline.user_store, pipeline.artifact_store, pipeline.audit_log,
        deletion=deletion,
    )
    assert monitor.list_pending_retrains() == ["alice"]


# ---------- CLI ----------


def test_cli_ingest_command(tmp_path: Path, capsys):
    note = tmp_path / "thoughts.md"
    note.write_text("This is a personal note about my project ideas for the quarter.")
    root = tmp_path / "data"
    rc = main([
        "--root", str(root),
        "ingest",
        "--user", "alice",
        "--source", str(note),
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["raw_items_ingested"] == 1


def test_cli_full_run_dry(tmp_path: Path, capsys):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    # Need enough distinct content to survive dedup + meet min_examples_to_train
    for i, topic in enumerate(_DISTINCT_TOPICS[:20]):
        (notes_dir / f"note-{i:03d}.md").write_text(
            topic + f"\n\nMore reflection on this topic — written {i} days ago."
        )
    root = tmp_path / "data"
    rc = main([
        "--root", str(root),
        "run",
        "--user", "alice",
        "--source", str(notes_dir),
        "--dry-run",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["user_id"] == "alice"
    assert out["status"] == "dry_run"


def test_cli_status_for_user(tmp_path: Path, capsys):
    root = tmp_path / "data"
    # Ingest something so the user exists
    note = tmp_path / "n.md"
    note.write_text("a personal thought")
    main(["--root", str(root), "ingest", "--user", "alice", "--source", str(note)])
    capsys.readouterr()  # clear

    rc = main(["--root", str(root), "status", "--user", "alice"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["user_id"] == "alice"
    assert out["raw_item_count"] == 1


def test_cli_system_status_when_empty(tmp_path: Path, capsys):
    root = tmp_path / "data"
    rc = main(["--root", str(root), "status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_users"] == 0
    assert out["users"] == []


def test_cli_parser_builds():
    parser = build_parser()
    assert parser.prog == "pmc"
    args = parser.parse_args([
        "--root", "/tmp",
        "ingest",
        "--user", "u",
        "--source", "/tmp/x.md",
    ])
    assert args.user == "u"
    assert args.source == ["/tmp/x.md"]


def test_cli_delete_command(tmp_path: Path, capsys):
    root = tmp_path / "data"
    note = tmp_path / "n.md"
    note.write_text("a personal thought")
    main(["--root", str(root), "ingest", "--user", "alice", "--source", str(note)])
    capsys.readouterr()

    rc = main([
        "--root", str(root),
        "delete",
        "--user", "alice",
        "--scope", "all_data",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scope"] == "all_data"
