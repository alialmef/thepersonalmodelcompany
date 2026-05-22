"""End-to-end pipeline: ingest → curate → train → eval → gate → deploy.

The pipeline composes every layer. Each stage is a method on `PMCPipeline` so
tests can drive stages independently. The full `run()` invokes them in order
and records audit events between each.

Key injection points so tests can avoid GPU dependencies:
- `train_fn` — replaces `pmc.train.sft.run_sft` for tests / dry runs
- `generator_factory` — builds (personal, base) generators for eval; replaced
  with MockGenerators in tests
- `benchmarks_factory` — builds the eval Benchmark list (defaults to privacy +
  optional style/factual if probes are supplied)

The doc's "smart epoch" rule lives here: 3 epochs if <5K examples, 1 if more.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pmc.curate.pipeline import CurateConfig, CuratePipeline, CurateResult
from pmc.eval.benchmarks import Benchmark
from pmc.eval.gate import EvalGate, EvalGateConfig, GateCheck, GateDecision
from pmc.eval.generator import ModelGenerator
from pmc.eval.privacy_eval import PrivacyBenchmark
from pmc.eval.runner import EvalSuiteResult, PersonalEvalRunner
from pmc.ingest.normalize import Normalizer
from pmc.orchestrator.data_source import DataSource
from pmc.schema.conversation import Completion, Conversation
from pmc.schema.training import TrainingConfig
from pmc.schema.user import User
from pmc.serve.registry import AdapterRegistry
from pmc.storage.artifact_store import ArtifactStore, new_run_id
from pmc.storage.audit import AuditLog
from pmc.storage.deletion import DeletionManager
from pmc.storage.user_store import UserStore
from pmc.storage.verification_store import VerificationStore
from pmc.train.bundle import ArtifactBundle, AuditEvent as BundleAuditEvent, BundleMetadata
from pmc.train.config import SFTRunResult, TrainingPlan
from pmc.train.sft import plan_sft

PipelineStatus = Literal["completed", "blocked", "dry_run", "failed", "no_data"]


TrainFn = Callable[
    [TrainingConfig, list[Completion], Path, list[Completion] | None],
    SFTRunResult,
]
GeneratorFactory = Callable[[str, Path], ModelGenerator]
BenchmarksFactory = Callable[["PipelineConfig", list[Completion]], list[Benchmark]]


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Everything one pipeline run needs."""

    user_id: str
    user_name: str = ""
    user_email: str = ""
    base_model: str = "Qwen/Qwen3-8B"
    data_sources: list[DataSource] = Field(default_factory=list)

    # Stage controls
    skip_train: bool = False
    skip_eval: bool = False
    skip_deploy: bool = False
    dry_run: bool = False

    # Curation
    curate_config: CurateConfig = Field(default_factory=CurateConfig)

    # Training
    training_config: TrainingConfig | None = None
    smart_epochs: bool = True

    # Data split
    holdout_fraction: float = 0.1
    seed: int = 42
    min_examples_to_train: int = 10

    # Eval / gate
    gate_config: EvalGateConfig | None = None
    require_eval_pass_to_deploy: bool = True
    require_verification_to_deploy: bool = False
    privacy_eval_samples: int = 20

    # IDs (auto-generated if omitted)
    dataset_version: str | None = None
    run_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class PipelineResult(BaseModel):
    """The outcome of a single pipeline run."""

    user_id: str
    status: PipelineStatus
    run_id: str | None = None
    dataset_version: str | None = None
    raw_items_ingested: int = 0
    completions_curated: int = 0
    dropped_curate: dict[str, int] = Field(default_factory=dict)
    training_plan: TrainingPlan | None = None
    training_result: SFTRunResult | None = None
    eval_result: EvalSuiteResult | None = None
    gate_decision_reason: str = ""
    gate_passed: bool | None = None
    deployed: bool = False
    elapsed_seconds: float = 0.0
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PMCPipeline:
    """End-to-end orchestration tying every layer together."""

    def __init__(
        self,
        user_store: UserStore,
        artifact_store: ArtifactStore,
        audit_log: AuditLog,
        *,
        deletion: DeletionManager | None = None,
        registry: AdapterRegistry | None = None,
        verification_store: VerificationStore | None = None,
        train_fn: TrainFn | None = None,
        generator_factory: GeneratorFactory | None = None,
        benchmarks_factory: BenchmarksFactory | None = None,
        extra_benchmarks: list[Benchmark] | None = None,
    ) -> None:
        self.user_store = user_store
        self.artifact_store = artifact_store
        self.audit_log = audit_log
        self.deletion = deletion
        self.registry = registry
        self.verification_store = verification_store or VerificationStore(self.user_store.paths.root)
        self._train_fn = train_fn
        self._generator_factory = generator_factory
        self._benchmarks_factory = benchmarks_factory
        self._extra_benchmarks = extra_benchmarks or []

    # -- top-level entry --------------------------------------------------

    def run(self, config: PipelineConfig) -> PipelineResult:
        started = datetime.now()
        t0 = time.time()
        result = PipelineResult(user_id=config.user_id, status="failed", started_at=started)
        try:
            self._record_user_profile(config)

            raw_count = self.stage_ingest(config)
            result.raw_items_ingested = raw_count
            total_raw = self.user_store.count_raw_items(config.user_id)
            if total_raw == 0:
                result.status = "no_data"
                result.notes = "No raw items available for this user"
                return self._finalize(result, t0)

            # Voice reading events fire immediately — raw/*.jsonl exists
            # from /connect's ingest. Memory events fire later, inside
            # stage_memory after migrate, because that's when graph
            # data is guaranteed to have arrived from graph_kickoff and
            # been processed.
            try:
                self._emit_voice_sources(config.user_id)
            except Exception as e:
                self.audit_log.log(
                    config.user_id, stage="memory",
                    event="reading_sources_emit_failed",
                    data={"error": str(e)},
                )

            curate_result, dataset_version = self.stage_curate(config)
            config.dataset_version = dataset_version
            result.completions_curated = len(curate_result.completions)
            result.dataset_version = dataset_version
            result.dropped_curate = {
                "short": curate_result.stats.dropped_short,
                "duplicate": curate_result.stats.dropped_duplicate,
                "low_quality": curate_result.stats.dropped_low_quality,
            }

            if len(curate_result.completions) < config.min_examples_to_train:
                result.status = "no_data"
                result.notes = (
                    f"After curation, only {len(curate_result.completions)} examples "
                    f"remain (min {config.min_examples_to_train} required)"
                )
                return self._finalize(result, t0)

            # Memory stage — builds the recall layer (Episodes,
            # bi-temporal facts, working memory) on the same data we
            # just curated. Non-fatal: if anything in here errors, the
            # pipeline continues to training. Memory is parallelizable
            # with training but for V0 we run it serially before train
            # so audit events stream in a clean order.
            try:
                self.stage_memory(config)
            except Exception as e:
                self.audit_log.log(
                    config.user_id, stage="memory", event="memory_stage_skipped",
                    data={"error": str(e), "error_type": type(e).__name__},
                )

            train_ds, holdout_ds = self._split(curate_result.completions, config)
            self._persist_training_split(config, dataset_version, train_ds, holdout_ds)
            plan, training_result = self.stage_train(config, train_ds, holdout_ds)
            result.training_plan = plan
            result.training_result = training_result

            if config.dry_run or config.skip_train or training_result is None:
                result.status = "dry_run" if config.dry_run else "completed"
                result.notes = "Training was skipped (dry_run or skip_train)"
                return self._finalize(result, t0)

            run_id = self._save_bundle(config, training_result, curate_result, dataset_version)
            result.run_id = run_id
            if config.require_verification_to_deploy:
                self._register_candidate_for_verification(config, run_id)

            if config.skip_eval:
                result.status = "completed"
                result.notes = "Eval skipped"
                if not config.skip_deploy:
                    if config.require_verification_to_deploy:
                        verification_decision = self.stage_verification_gate(config)
                        result.gate_passed = verification_decision.deploy
                        result.gate_decision_reason = verification_decision.reason
                        if not verification_decision.deploy:
                            result.status = "blocked"
                            result.notes = verification_decision.reason
                            return self._finalize(result, t0)
                    self.stage_deploy(config, run_id, force=True)
                    result.deployed = True
                return self._finalize(result, t0)

            eval_result = self.stage_eval(config, run_id, holdout_ds, training_result)
            result.eval_result = eval_result

            decision = self.stage_gate(config, eval_result)
            result.gate_passed = decision.deploy
            result.gate_decision_reason = decision.reason

            if decision.deploy and config.require_verification_to_deploy:
                verification_decision = self.stage_verification_gate(config)
                if not verification_decision.deploy:
                    decision = verification_decision
                    result.gate_passed = False
                    result.gate_decision_reason = verification_decision.reason

            if decision.deploy and not config.skip_deploy:
                self.stage_deploy(config, run_id)
                result.deployed = True
                result.status = "completed"
            elif not decision.deploy:
                result.status = "blocked"
                result.notes = decision.reason
            else:
                result.status = "completed"
                result.notes = "Eval passed but deploy was skipped"

            return self._finalize(result, t0)

        except Exception as e:
            self.audit_log.log(
                config.user_id,
                stage="train",
                event="pipeline_error",
                run_id=result.run_id,
                data={"error": str(e), "error_type": type(e).__name__},
            )
            result.error = f"{type(e).__name__}: {e}"
            result.status = "failed"
            return self._finalize(result, t0)

    # -- stages -----------------------------------------------------------

    def stage_ingest(self, config: PipelineConfig) -> int:
        """Persist raw items from every data source. Returns total count."""
        total = 0
        for source in config.data_sources:
            source_id = source.derived_source_id()
            items = list(source.ingest())
            n = self.user_store.save_raw_items(config.user_id, source_id, items)
            total += n
            self.audit_log.log(
                config.user_id,
                stage="ingest",
                event="source_loaded",
                data={"source_id": source_id, "kind": source.kind.value, "items": n},
            )
        self.audit_log.log(
            config.user_id,
            stage="ingest",
            event="ingest_completed",
            data={"total_items": total, "num_sources": len(config.data_sources)},
        )
        return total

    def stage_curate(
        self, config: PipelineConfig
    ) -> tuple[CurateResult, str]:
        """Normalize raw items, run curate, persist dataset. Returns (result, version)."""
        raw_items = list(self.user_store.load_raw_items(config.user_id))
        conversations: list[Conversation] = list(Normalizer().normalize(raw_items))

        pipeline = CuratePipeline(config=config.curate_config)
        curate_result = pipeline.curate(conversations)

        # Curate supervisor: Claude reviews a sample of the curated
        # examples and (a) auto-drops obvious noise that slipped past
        # the heuristic filters, (b) flags potential leaks / third-party
        # content for user review on /eval. Soft layer — failures
        # never block training.
        import os
        if os.environ.get("ANTHROPIC_API_KEY") and not getattr(config, "skip_supervisor", False):
            try:
                from pmc.curate.supervisor import supervise, apply_report
                report = supervise(curate_result.completions)
                if report.noise_indices or report.has_flags or report.errors:
                    self.audit_log.log(
                        config.user_id, stage="curate",
                        event="curate_supervisor_report",
                        data={
                            "summary": report.summary(),
                            "flags": [
                                {"index": v.index, "decision": v.decision, "reason": v.reason}
                                for v in (report.leaked + report.third_party)
                            ],
                            "errors": report.errors,
                        },
                    )
                if report.noise_indices:
                    curate_result.completions = apply_report(
                        curate_result.completions, report,
                    )
            except Exception as e:
                self.audit_log.log(
                    config.user_id, stage="curate",
                    event="curate_supervisor_failed",
                    data={"error": str(e), "error_type": type(e).__name__},
                )

        dataset_version = config.dataset_version or _default_dataset_version()
        manifest = self.user_store.save_curated_dataset(
            config.user_id,
            dataset_version,
            train=curate_result.completions,
        )

        self.audit_log.log(
            config.user_id,
            stage="curate",
            event="curate_completed",
            data={
                "dataset_version": dataset_version,
                "input_conversations": curate_result.stats.input_conversations,
                "split_completions": curate_result.stats.split_completions,
                "output_completions": curate_result.stats.output_completions,
                "dropped_short": curate_result.stats.dropped_short,
                "dropped_duplicate": curate_result.stats.dropped_duplicate,
                "dropped_low_quality": curate_result.stats.dropped_low_quality,
                "redacted_severe": curate_result.stats.redacted_severe,
                "manifest_checksum": manifest.checksum,
            },
        )
        return curate_result, dataset_version

    # ------------------------------------------------------------------
    # Filters for the /reading screen's memory section. The point of
    # each filter is to *break tie of trust at the first sign of noise*
    # rather than show a polluted list. The user looking at the screen
    # has to recognize every example; one bot or one raw address ruins
    # the demonstration.

    @staticmethod
    def _BOT_DOMAINS() -> set[str]:
        # Domains we never want to appear as "a person you talk to."
        # These are all transactional or business-messaging.
        return {
            "rbm.goog",         # Google Rich Business Messaging bots
            "messagebird.com",  # commercial SMS bridges
            "twilio.com",
            "e.delta.com", "uber.com", "lyft.com", "doordash.com",
            "amazon.com", "amazonses.com", "amazon-shipping.com",
            "stripe.com", "noreply.com", "no-reply.com",
        }

    @staticmethod
    def _BOT_PATTERNS() -> tuple[str, ...]:
        # Substring patterns in the local-part of an email that flag
        # automated senders.
        return (
            "_agent_", "_agent@", "noreply", "no-reply", "donotreply",
            "do-not-reply", "notification", "support@", "transactions@",
            "billing@", "receipts@", "alerts@",
        )

    def _emit_voice_sources(self, user_id: str) -> None:
        """Emit voice-bucket reading events. Raw data already exists
        on disk after /connect's ingest, so these fire immediately."""
        from pathlib import Path
        root = Path(self.user_store.paths.root) / "users" / user_id
        raw_dir = root / "raw"

        def jsonl_records(path: Path) -> int:
            if not path.is_file():
                return 0
            try:
                with path.open("r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
            except Exception:
                return 0

        voice = [
            ("messages",  raw_dir / "imessage.jsonl",  "messages"),
            ("notes",     raw_dir / "notes.jsonl",     "notes"),
            ("mail",      raw_dir / "mail.jsonl",      "sent emails"),
        ]
        for kind, path, phrase in voice:
            count = jsonl_records(path)
            if count == 0:
                continue
            self.audit_log.log(
                user_id, stage="memory", event="reading_source_found",
                data={"bucket": "voice", "kind": kind, "count": count, "phrase": phrase},
            )

    def _emit_reading_sources(self, user_id: str) -> None:
        """Backward-compat wrapper — emits both voice and memory in one
        pass. Used by tests / direct callers that don't have a
        graph-extraction-await step. Production runs call voice + memory
        separately (voice early, memory after migrate)."""
        self._emit_voice_sources(user_id)
        self._emit_memory_sources(user_id)

    def _emit_memory_sources(self, user_id: str) -> None:
        """Emit memory-bucket reading events with names baked in.

        Call this AFTER graph_kickoff (Tauri-side) has had a chance to
        run AND after stage_memory.migrate has folded the graph into
        recall.db. By that point every memory line is reading from
        settled data.

        Each event names:
          * `bucket`  — "voice" (training corpus) or "memory" (recall context)
          * `kind`    — source name (messages, notes, people, places, …)
          * `count`   — number of items found (GROUNDED, not raw)
          * `phrase`  — short plain-English context for the typed line

        IMPORTANT: counts on the memory side are *curated*, not raw.
        Showing "6,464 people" makes the user say "6,464 people in my
        life? really?" — because most of those are spam, businesses,
        one-time contacts, photo-detected strangers. The grounded
        version says "23 close people you actually talk to" — that
        feels true.

        Voice counts stay raw because they describe the training
        corpus directly (every message matters as a training example).
        Memory counts get filtered down to what a real human would
        recognize as the *real* people / places / events in their life.
        """
        import json
        from pathlib import Path

        root = Path(self.user_store.paths.root) / "users" / user_id
        graph_dir = root / "graph"

        def jsonl_records(path: Path) -> int:
            if not path.is_file():
                return 0
            try:
                with path.open("r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
            except Exception:
                return 0

        def load_jsonl(path: Path) -> list[dict]:
            if not path.is_file():
                return []
            out: list[dict] = []
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            out.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                pass
            return out

        # ---------- MEMORY — counts get filtered + anchored by names ----------
        #
        # Every memory line includes 2-3 concrete examples from the
        # user's actual data, so the count is *anchored*. Reading
        # "people you actually talk to: Dad, Aisha, Ben" creates
        # recognition; reading "163 people" creates skepticism.
        #
        # We currently emit three lines — people, places, themes.
        # Events and photo moments are off until we can produce
        # equally specific labels for them (e.g., "the Vermont
        # weekend in October", not "Photo cluster — 2026-10-14").
        memory_emissions: list[tuple[str, int, str]] = []

        # PEOPLE — only those the user explicitly saved in Contacts.
        # Anyone who appears as an iMessage handle / email without a
        # `display_name` is filtered out: this catches bot domains
        # (rbm.goog, *_agent_*), business-messaging addresses,
        # one-off numbers, and other noise. The trade-off is we
        # under-count (a friend the user never saved doesn't appear) —
        # but recognition beats recall here. The user has to *see*
        # the right names to trust the system; one bot in the list
        # destroys that trust.
        people = load_jsonl(graph_dir / "people.jsonl")
        real_people: list[tuple[int, dict, str]] = []
        for p in people:
            name = (p.get("display_name") or "").strip()
            if not _is_real_person_name(name):
                continue
            cc = p.get("channel_counts") or {}
            total = sum(v for v in cc.values() if isinstance(v, (int, float)))
            real_people.append((int(total), p, name))
        real_people.sort(key=lambda x: x[0], reverse=True)
        top_people_names: list[str] = []
        for _, _, name in real_people:
            if name not in top_people_names:
                top_people_names.append(name)
            if len(top_people_names) >= 3:
                break
        # Skip the line entirely unless we have at least 3 clear examples
        # — better to omit than to show one bot and break trust.
        if len(top_people_names) >= 3:
            named = ", ".join(top_people_names)
            memory_emissions.append((
                "people", len(real_people),
                f"people you actually talk to: {named}.",
            ))

        # PLACES — only emit if we have clean human-readable labels.
        # Calendar location strings are often raw addresses or URLs
        # pulled from notes; we don't want any of those leaking
        # through. For V0 we require:
        #   * label has alphabetic content (not "(40.72, -73.95)")
        #   * label doesn't look like a URL
        #   * label doesn't look like a street address (has a comma
        #     and is mostly digits — strong sign it's a postal address)
        # If we can't find 3 clean labels, skip the line.
        places = load_jsonl(graph_dir / "places.jsonl")
        candidates = [pl for pl in places
                       if pl.get("kind") in {"frequent", "trip", "home", "work", "venue"}
                       and (pl.get("visit_count") or 0) >= 1]
        candidates.sort(key=lambda p: (p.get("visit_count") or 0), reverse=True)
        top_place_names: list[str] = []
        for pl in candidates:
            label = (pl.get("label") or "").strip()
            if _is_clean_place_label(label) and label not in top_place_names:
                top_place_names.append(label)
            if len(top_place_names) >= 3:
                break
        if len(top_place_names) >= 3:
            named = ", ".join(top_place_names)
            memory_emissions.append((
                "places", min(len(candidates), 40),
                f"places you keep going back to: {named}.",
            ))

        # THEMES — top 5 by recent activity, named inline.
        themes = load_jsonl(graph_dir / "themes.jsonl")
        themes_sorted = sorted(
            themes,
            key=lambda t: (t.get("mentions_30d") or 0) * 4 + (t.get("mentions_180d") or 0),
            reverse=True,
        )
        top_themes = [t.get("label", "").strip() for t in themes_sorted[:5] if t.get("label")]
        if top_themes:
            named = ", ".join(top_themes[:3])
            memory_emissions.append((
                "themes", len(top_themes),
                f"things you keep coming back to: {named}.",
            ))

        for kind, count, phrase in memory_emissions:
            self.audit_log.log(
                user_id, stage="memory", event="reading_source_found",
                data={"bucket": "memory", "kind": kind, "count": count, "phrase": phrase},
            )

    def stage_memory(self, config: PipelineConfig) -> dict:
        """Build the recall layer (Episodes, embeddings, optional Claude
        consolidation, working memory snapshot) for this user.

        Three sub-steps:
          1. Migrate raw substrate + structural graph (if extractors
             have written one) into the user's `recall.db`.
          2. Preview-consolidate any new Episodes — heuristic summaries
             plus local BGE embeddings. Cheap, no API cost.
          3. Claude consolidation pass over the same Episodes, only if
             ANTHROPIC_API_KEY is in the environment. Produces real
             semantic summaries + bi-temporal facts + importance.
          4. Working-memory snapshot for today (anticipation items).
             Requires Claude as well.

        Each sub-step emits its own audit event so the UI sees granular
        progress. Failures are logged and skipped — the pipeline
        always continues to training.
        """
        import os
        from pmc.memory.recall.migrate import build_episodes, raw_text_for_episode, storage_paths
        from pmc.memory.recall.preview import preview_consolidate

        storage_root = str(self.user_store.paths.root)
        user_id = config.user_id

        # Note: _emit_reading_sources is now called from run() right
        # after stage_ingest, before curate, so the /reading screen
        # populates within seconds rather than waiting for curate to
        # finish (~1.5 min).

        # Step 1: migrate (always runs). After this lands, graph_kickoff
        # has had time to populate graph/*.jsonl and we've folded it
        # into recall.db — both inputs to _emit_memory_sources are
        # finally settled, so it's safe to fire the memory reading
        # events with grounded counts + named examples.
        try:
            counts = build_episodes(storage_root, user_id)
            self.audit_log.log(
                user_id, stage="memory", event="memory_migrate_completed",
                data={"counts_per_source": counts, "total": sum(counts.values())},
            )
            # Memory reading events — voice has already fired by now;
            # this fills in "Structuring your memory." with grounded
            # counts + names.
            try:
                self._emit_memory_sources(user_id)
            except Exception as e:
                self.audit_log.log(
                    user_id, stage="memory",
                    event="memory_reading_emit_failed",
                    data={"error": str(e)},
                )
        except Exception as e:
            self.audit_log.log(
                user_id, stage="memory", event="memory_migrate_failed",
                data={"error": str(e), "error_type": type(e).__name__},
            )
            return {"status": "migrate_failed"}

        # Step 2: preview consolidation (always runs — cheap)
        from pmc.memory.recall.store import RecallStore
        paths = storage_paths(storage_root, user_id)
        store = RecallStore(paths["recall"])
        try:
            def lookup(ep):
                return raw_text_for_episode(storage_root, user_id, ep)
            n_preview = preview_consolidate(store, lookup, limit=10_000)
            self.audit_log.log(
                user_id, stage="memory", event="memory_preview_completed",
                data={"episodes_summarized": n_preview, "model": "preview/heuristic-v1"},
            )
        except Exception as e:
            self.audit_log.log(
                user_id, stage="memory", event="memory_preview_failed",
                data={"error": str(e), "error_type": type(e).__name__},
            )

        # Step 3: Claude consolidation (only if API key + not skipped)
        if os.environ.get("ANTHROPIC_API_KEY") and not getattr(config, "skip_smart_memory", False):
            try:
                from pmc.memory.recall.consolidate import Consolidator
                c = Consolidator(max_episodes_per_run=10_000, max_concurrency=12)
                def lookup_full(ep):
                    return raw_text_for_episode(storage_root, user_id, ep)
                # Stream a progress event every 50 episodes so the UI shows life.
                done = [0]
                def progress(d, total):
                    if d - done[0] >= 50 or d == total:
                        self.audit_log.log(
                            user_id, stage="memory", event="memory_consolidate_progress",
                            data={"done": d, "total": total},
                        )
                        done[0] = d
                results = c.consolidate_pending(store, lookup_full, progress=progress)
                self.audit_log.log(
                    user_id, stage="memory", event="memory_consolidate_completed",
                    data={"episodes_processed": len(results), "model": c.model},
                )
            except Exception as e:
                self.audit_log.log(
                    user_id, stage="memory", event="memory_consolidate_failed",
                    data={"error": str(e), "error_type": type(e).__name__},
                )

            # Step 3.5: memory supervisor — Claude reviews a sample
            # of consolidated summaries for fabrication, surveillance,
            # identity leaks. Flags go to /eval. Failures are
            # non-fatal — never block the rest of the pipeline.
            try:
                from pmc.memory.recall.supervisor import supervise_memory
                def lookup_full(ep):
                    return raw_text_for_episode(storage_root, user_id, ep)
                mem_report = supervise_memory(store, lookup_full)
                if mem_report.has_flags or mem_report.errors:
                    self.audit_log.log(
                        user_id, stage="memory",
                        event="memory_supervisor_report",
                        data={
                            "summary": mem_report.summary(),
                            "flags": [
                                {"episode_id": f.episode_id,
                                 "decision": f.decision,
                                 "reason": f.reason}
                                for f in mem_report.flags
                            ],
                            "errors": mem_report.errors,
                        },
                    )
            except Exception as e:
                self.audit_log.log(
                    user_id, stage="memory",
                    event="memory_supervisor_failed",
                    data={"error": str(e), "error_type": type(e).__name__},
                )

            # Step 4: working memory snapshot
            try:
                from pmc.memory.recall.working import build_working_memory
                snap = build_working_memory(store)
                self.audit_log.log(
                    user_id, stage="memory", event="memory_working_built",
                    data={
                        "anticipation_count": len(snap.anticipation),
                        "hot_people_count": len(snap.hot_people),
                        "rising_themes_count": len(snap.rising_themes),
                    },
                )
            except Exception as e:
                self.audit_log.log(
                    user_id, stage="memory", event="memory_working_failed",
                    data={"error": str(e), "error_type": type(e).__name__},
                )
        else:
            self.audit_log.log(
                user_id, stage="memory", event="memory_smart_skipped",
                data={"reason": "no ANTHROPIC_API_KEY in env"},
            )

        stats = store.stats()
        store.close()
        return {"status": "ok", "stats": stats}

    def stage_train(
        self,
        config: PipelineConfig,
        train_completions: list[Completion],
        holdout_completions: list[Completion],
    ) -> tuple[TrainingPlan, SFTRunResult | None]:
        """Plan and (unless dry_run) run SFT. Returns (plan, result-or-None)."""
        verification_feedback = self._verification_training_completions(config.user_id)
        if verification_feedback:
            train_completions = [*train_completions, *verification_feedback]

        training_config = self._resolve_training_config(config, len(train_completions))
        plan = plan_sft(training_config, train_completions, holdout_completions)

        self.audit_log.log(
            config.user_id,
            stage="train",
            event="training_planned",
            data={
                "base_model": plan.base_model,
                "num_train": plan.num_train_examples,
                "num_eval": plan.num_eval_examples,
                "steps": plan.estimated_steps,
                "minutes": plan.estimated_minutes,
                "adapter_mb": plan.estimated_adapter_mb,
                "warnings": plan.warnings,
                "verification_examples": len(verification_feedback),
            },
        )

        if config.dry_run or config.skip_train:
            return plan, None

        run_id = config.run_id or new_run_id()
        output_dir = self.artifact_store.paths.bundle_dir(config.user_id, run_id) / "adapter"
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        train_fn = self._train_fn or _default_train_fn
        self.audit_log.log(
            config.user_id, stage="train", event="training_started", run_id=run_id,
            data={"output_dir": str(output_dir)},
        )

        # Forward train_fn events into the audit log so the UI's SSE
        # stream picks them up. Critical for the live-voice viz
        # (checkpoint_sample events) and Together job heartbeats.
        def _audit_event(kind: str, data: dict) -> None:
            self.audit_log.log(
                config.user_id, stage="train", event=kind,
                run_id=run_id, data=data,
            )
        try:
            result = train_fn(training_config, train_completions, output_dir,
                               holdout_completions, on_event=_audit_event)
        except TypeError:
            # Older train_fns (e.g. test mocks) don't accept on_event.
            result = train_fn(training_config, train_completions, output_dir,
                               holdout_completions)
        # If the train_fn doesn't carry through our run_id, override the user_id at least.
        result.user_id = config.user_id
        self.audit_log.log(
            config.user_id, stage="train", event="training_completed", run_id=run_id,
            data={
                "elapsed_seconds": result.elapsed_seconds,
                "final_train_loss": result.final_train_loss,
                "final_eval_loss": result.final_eval_loss,
                "adapter_dir": str(result.adapter_dir),
            },
        )
        # Cache the chosen run_id on the config for downstream stages
        config.run_id = run_id
        return plan, result

    def stage_eval(
        self,
        config: PipelineConfig,
        run_id: str,
        holdout: list[Completion],
        training_result: SFTRunResult,
    ) -> EvalSuiteResult:
        """Build generators + benchmarks, run the eval suite."""
        gen_factory = self._generator_factory or _default_generator_factory
        personal = gen_factory(config.base_model, Path(training_result.adapter_dir))
        base = gen_factory(config.base_model, None)  # type: ignore[arg-type]

        if self._benchmarks_factory is not None:
            benchmarks = self._benchmarks_factory(config, holdout)
        else:
            benchmarks = self._default_benchmarks(config)
        benchmarks.extend(self._extra_benchmarks)

        runner = PersonalEvalRunner(
            benchmarks=benchmarks,
            user_id=config.user_id,
            adapter_dir=str(training_result.adapter_dir),
            base_model=config.base_model,
        )
        eval_result = runner.run(personal, base)
        self.audit_log.log(
            config.user_id, stage="eval", event="eval_completed", run_id=run_id,
            data={"scores": eval_result.to_summary(), "elapsed_seconds": eval_result.elapsed_seconds},
        )
        return eval_result

    def stage_gate(
        self, config: PipelineConfig, eval_result: EvalSuiteResult
    ) -> GateDecision:
        gate = EvalGate(config.gate_config)
        decision = gate.decide(eval_result)
        self.audit_log.log(
            config.user_id, stage="gate", event="gate_decision",
            data={
                "deploy": decision.deploy,
                "failed": decision.failed,
                "reason": decision.reason,
            },
        )
        return decision

    def stage_verification_gate(self, config: PipelineConfig) -> GateDecision:
        """Gate deployment on private user verification feedback."""
        report = self.verification_store.trust_report(config.user_id)
        ready = report.readiness in {"voice", "sandbox", "supervised"}
        failed: list[str] = []
        if not ready:
            failed.append(f"personal verification ({report.readiness})")
        if report.privacy_flags:
            failed.append(f"privacy flags ({report.privacy_flags})")
        decision = GateDecision(
            deploy=not failed,
            checks=[
                GateCheck(
                    name="personal_verification",
                    score=report.scores.get("voice_acceptance"),
                    threshold=0.7,
                    passed=ready and report.privacy_flags == 0,
                )
            ],
            failed=failed,
            reason=(
                "Private verification passed."
                if not failed
                else f"Failed: {', '.join(failed)}"
            ),
        )
        self.audit_log.log(
            config.user_id,
            stage="gate",
            event="verification_gate_decision",
            data={
                "deploy": decision.deploy,
                "failed": decision.failed,
                "reason": decision.reason,
                "readiness": report.readiness,
                "voice_total": report.voice_total,
                "voice_approved": report.voice_approved,
                "privacy_flags": report.privacy_flags,
            },
        )
        return decision

    def stage_deploy(
        self, config: PipelineConfig, run_id: str, *, force: bool = False
    ) -> bool:
        """Promote a run to active and register for serving."""
        if config.skip_deploy and not force:
            return False

        # Pre-deploy supervisor: sample the trained model on a diverse
        # prompt set, ask Claude if voice took. Emits an audit event
        # the /eval screen renders. If verdict is "hold" and the user
        # hasn't already approved on /eval, we still proceed for V0 —
        # /eval handles the user-facing pause. Soft layer.
        import os
        if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("TOGETHER_API_KEY") \
                and not getattr(config, "skip_supervisor", False):
            try:
                self._supervise_deploy(config, run_id)
            except Exception as e:
                self.audit_log.log(
                    config.user_id, stage="deploy",
                    event="deploy_supervisor_failed", run_id=run_id,
                    data={"error": str(e), "error_type": type(e).__name__},
                )

        self.artifact_store.set_active(config.user_id, run_id, notes=f"pipeline run {run_id}")
        if self.registry is not None:
            bundle_dir = self.artifact_store.paths.bundle_dir(config.user_id, run_id)
            self.registry.register_bundle(bundle_dir)

        # Clear retrain-needed tombstones since we've just produced a fresh model.
        if self.deletion is not None:
            self.deletion.mark_retrained(config.user_id, run_id=run_id)

        self.audit_log.log(
            config.user_id, stage="deploy", event="adapter_deployed", run_id=run_id,
            data={"registered": self.registry is not None},
        )
        return True

    def _supervise_deploy(self, config: PipelineConfig, run_id: str) -> None:
        """Read the remote.json from the bundle, sample the model,
        get Claude's verdict, emit audit events for /eval."""
        import json as _json
        from pmc.eval.deploy_supervisor import supervise_deploy

        bundle_dir = self.artifact_store.paths.bundle_dir(config.user_id, run_id)
        remote_path = bundle_dir / "adapter" / "remote.json"
        if not remote_path.is_file():
            # No remote.json means this wasn't a Together run — skip.
            return
        try:
            remote = _json.loads(remote_path.read_text())
        except Exception:
            return
        output_model = remote.get("output_model")
        base_model = remote.get("base_model")
        if not output_model or not base_model:
            return

        # Pull a few authored snippets so Claude has voice ground truth.
        completions = self.user_store.load_curated_dataset(
            config.user_id, config.dataset_version or ""
        ) if config.dataset_version else []
        writing_samples: list[str] = []
        for c in completions[:8]:
            for cand in c.candidates[:1]:
                for m in cand.messages:
                    if m.content.strip():
                        writing_samples.append(m.content[:400])
                        break

        report = supervise_deploy(
            base_model=base_model,
            output_model=output_model,
            user_writing_samples=writing_samples,
        )
        self.audit_log.log(
            config.user_id, stage="deploy",
            event="deploy_supervisor_report", run_id=run_id,
            data={
                "verdict": report.verdict,
                "summary": report.summary,
                "issues": [
                    {"prompt_index": i.prompt_index, "kind": i.kind, "note": i.note}
                    for i in report.issues
                ],
                "samples": report.samples,
                "errors": report.errors,
            },
        )

    # -- helpers ----------------------------------------------------------

    def record_user_profile(self, config: PipelineConfig) -> None:
        """Publicly callable from CLI stage commands. Idempotent."""
        return self._record_user_profile(config)

    def _record_user_profile(self, config: PipelineConfig) -> None:
        existing = self.user_store.load_user(config.user_id)
        if existing is not None:
            return
        if not config.user_email and not config.user_name:
            return
        user = User(
            email=config.user_email or f"{config.user_id}@unknown",
            name=config.user_name or config.user_id,
        )
        self.user_store.save_user(user, user_id=config.user_id)
        self.audit_log.log(
            config.user_id, stage="user", event="profile_created",
            data={"name": user.name, "email": user.email},
        )

    def _verification_training_completions(self, user_id: str) -> list[Completion]:
        """Return user-corrected training examples for the next SFT run."""
        try:
            preference = self.verification_store.preference_completions(user_id)
            action_sft = self.verification_store.action_sft_completions(user_id)
        except Exception as e:
            self.audit_log.log(
                user_id,
                stage="train",
                event="verification_signal_failed",
                data={"error": str(e), "error_type": type(e).__name__},
            )
            return []

        examples = [*preference, *action_sft]
        if examples:
            self.audit_log.log(
                user_id,
                stage="train",
                event="verification_signal_attached",
                data={
                    "preference_completions": len(preference),
                    "action_sft_completions": len(action_sft),
                    "total_completions": len(examples),
                },
            )
        return examples

    def _persist_training_split(
        self,
        config: PipelineConfig,
        dataset_version: str,
        train: list[Completion],
        holdout: list[Completion],
    ) -> None:
        """Persist the exact train/holdout split used by this run.

        `stage_curate` persists the raw curated output for stage-level callers.
        A full pipeline run immediately overwrites that version with the exact
        training split so private eval probes can be generated from real
        held-out user examples.
        """
        manifest = self.user_store.save_curated_dataset(
            config.user_id,
            dataset_version,
            train=train,
            holdout=holdout,
        )
        self.audit_log.log(
            config.user_id,
            stage="curate",
            event="dataset_split_persisted",
            data={
                "dataset_version": dataset_version,
                "train_examples": len(train),
                "holdout_examples": len(holdout),
                "manifest_checksum": manifest.checksum,
            },
        )

    def _register_candidate_for_verification(
        self,
        config: PipelineConfig,
        run_id: str,
    ) -> None:
        """Make a just-trained adapter available for private eval prompts.

        This intentionally does not set `active.json`. The run can generate
        candidate replies for the eval screen, but it is not promoted as the
        user's active model until the trust report passes.
        """
        if self.registry is None:
            return
        bundle_dir = self.artifact_store.paths.bundle_dir(config.user_id, run_id)
        try:
            self.registry.register_bundle(bundle_dir)
        except Exception as e:
            self.audit_log.log(
                config.user_id,
                stage="deploy",
                event="candidate_registration_failed",
                run_id=run_id,
                data={"error": str(e), "error_type": type(e).__name__},
            )
            return
        self.audit_log.log(
            config.user_id,
            stage="deploy",
            event="candidate_registered_for_verification",
            run_id=run_id,
            data={"active": False},
        )

    def _split(
        self, completions: list[Completion], config: PipelineConfig
    ) -> tuple[list[Completion], list[Completion]]:
        if config.holdout_fraction <= 0 or len(completions) < 10:
            return completions, []
        import random
        rng = random.Random(config.seed)
        shuffled = completions.copy()
        rng.shuffle(shuffled)
        split = int(len(shuffled) * (1 - config.holdout_fraction))
        return shuffled[:split], shuffled[split:]

    def _resolve_training_config(
        self, config: PipelineConfig, num_examples: int
    ) -> TrainingConfig:
        if config.training_config is not None:
            tc = config.training_config.model_copy(update={"user_id": config.user_id})
        else:
            tc = TrainingConfig(user_id=config.user_id, base_model=config.base_model)
        if config.smart_epochs:
            tc = tc.model_copy(update={"num_epochs": 3 if num_examples < 5000 else 1})
        return tc

    def _save_bundle(
        self,
        config: PipelineConfig,
        training_result: SFTRunResult,
        curate_result: CurateResult,
        dataset_version: str,
    ) -> str:
        manifest = self.user_store.load_manifest(config.user_id, dataset_version)
        bundle = ArtifactBundle(
            metadata=BundleMetadata(
                user_id=config.user_id,
                user_name=config.user_name or None,
                user_email=config.user_email or None,
                base_model=config.base_model,
                job_type="sft",
            ),
            adapter_dir=Path(training_result.adapter_dir),
            style_profile=curate_result.style_profile,
            training_manifest=manifest,
            audit_log=[
                BundleAuditEvent(
                    stage="train", event="sft_completed",
                    data={
                        "elapsed_seconds": training_result.elapsed_seconds,
                        "final_train_loss": training_result.final_train_loss,
                        "num_train_examples": training_result.num_train_examples,
                    },
                )
            ],
        )
        run_id = config.run_id or new_run_id()
        config.run_id = run_id
        # The training step already wrote adapter into bundle_dir/adapter; the
        # ArtifactStore.save_bundle call below writes the sidecars next to it.
        self.artifact_store.save_bundle(
            config.user_id, bundle, run_id=run_id, copy_adapter=False
        )
        return run_id

    def _default_benchmarks(self, config: PipelineConfig) -> list[Benchmark]:
        """Default eval set: just privacy. Style/factual need probes/judges
        the caller should provide via benchmarks_factory."""
        training_texts = []
        for item in self.user_store.load_raw_items(config.user_id):
            if item.is_user and len(item.content) >= 100:
                training_texts.append(item.content)
                if len(training_texts) >= 200:
                    break
        return [
            PrivacyBenchmark(
                training_texts=training_texts,
                num_samples=min(config.privacy_eval_samples, len(training_texts)),
            )
        ]

    def _finalize(self, result: PipelineResult, t0: float) -> PipelineResult:
        result.elapsed_seconds = round(time.time() - t0, 2)
        result.completed_at = datetime.now()
        self.audit_log.log(
            result.user_id, stage="train", event="pipeline_finished", run_id=result.run_id,
            data={
                "status": result.status,
                "elapsed_seconds": result.elapsed_seconds,
                "deployed": result.deployed,
            },
        )
        return result


# ---------------------------------------------------------------------------
# Defaults that lazy-import heavy deps
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filters for the /reading screen — keep them at module scope so they're
# easy to test in isolation. The threshold for "shows on screen" is
# *clearly recognizable human/place*. Anything ambiguous gets dropped.
# ---------------------------------------------------------------------------


_BOT_LOCAL_PATTERNS = (
    "_agent_", "_agent@", "noreply", "no-reply", "donotreply",
    "do-not-reply", "notification", "support@", "transactions@",
    "billing@", "receipts@", "alerts@",
)

_BOT_DOMAINS = {
    "rbm.goog",
    "messagebird.com",
    "twilio.com",
    "e.delta.com", "uber.com", "lyft.com", "doordash.com",
    "amazon.com", "amazonses.com", "amazon-shipping.com",
    "stripe.com", "noreply.com", "no-reply.com",
}


def _is_real_person_name(name: str) -> bool:
    """True if `name` looks like a name a user might recognize.

    Rejects:
      * empty
      * raw phone numbers (digits + + - spaces)
      * bare email addresses (any '@')
      * bot-pattern locals or known bot domains
      * single-token all-lowercase strings (likely a handle, not a name)
    """
    s = (name or "").strip()
    if not s:
        return False
    # Email address — never show as a "person name"
    if "@" in s:
        local, _, domain = s.partition("@")
        domain = domain.lower()
        if domain in _BOT_DOMAINS:
            return False
        if any(pat in s.lower() for pat in _BOT_LOCAL_PATTERNS):
            return False
        # Bare email without a paired display_name elsewhere isn't a
        # name we want to surface.
        return False
    # Phone number
    stripped = s.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("+", "")
    if stripped.isdigit():
        return False
    # Must have at least one letter
    if not any(c.isalpha() for c in s):
        return False
    return True


_STREET_SUFFIXES = (
    " st.", " st,", " st ", " st\n",
    " ave.", " ave,", " ave ",
    " avenue", " road", " rd.", " rd,", " rd ",
    " blvd.", " blvd,", " blvd ",
    " dr.", " dr,", " dr ",
    " way", " lane", " ln.", " ln,",
    " court", " ct.", " ct,",
    " parkway", " pkwy",
    " highway", " hwy",
    " suite ", " ste ", " apt ", " unit ",
)


def _is_clean_place_label(label: str) -> bool:
    """True if `label` is a name-shaped place rather than a raw
    address, URL, or coordinate. Conservative: when in doubt, reject —
    showing one bad place destroys recognition for the whole list."""
    s = (label or "").strip()
    if not s:
        return False
    lower = " " + s.lower() + " "  # pad for boundary matching
    # URLs / shortlinks
    if any(lower.lstrip().startswith(p) for p in ("http://", "https://", "www.", "bit.ly", "t.co/", "goo.gl")):
        return False
    if "://" in lower or "bit.ly/" in lower:
        return False
    # Coordinate-only labels from the photos extractor
    if s.startswith("(") and s.endswith(")") and "," in s and any(c.isdigit() for c in s):
        return False
    # Street-address signals — any of these substrings means raw postal.
    if any(suf in lower for suf in _STREET_SUFFIXES):
        return False
    # US ZIP code at the end (5 digits, optional -4)
    import re as _re
    if _re.search(r"\b\d{5}(?:-\d{4})?\b", s):
        return False
    # Anything with both a comma AND >=3 digits is almost always an
    # address / coordinate / appended numeric — skip.
    digits = sum(1 for c in s if c.isdigit())
    if "," in s and digits >= 3:
        return False
    # Must have alphabetic content
    if not any(c.isalpha() for c in s):
        return False
    return True


def _default_train_fn(
    config: TrainingConfig,
    train: list[Completion],
    output_dir: Path,
    eval_completions: list[Completion] | None,
    *,
    on_event=None,
) -> SFTRunResult:
    """Pick the best trainer available.

    Preference order (env overrides win, otherwise auto-detect):
      1. `PMC_TRAINER=together` → Together fine-tuning API (Kimi-K2
         LoRA). The production path; produces the model the user
         actually wants. Needs TOGETHER_API_KEY in env.
      2. `PMC_TRAINER=mlx` → forces local MLX (dev-only fast path).
      3. Auto-detect: if TOGETHER_API_KEY is present, use Together.
      4. Else if MLX is available on this host, use it.
      5. Else fall back to PyTorch + TRL + PEFT (HF stack).
    """
    import os
    forced = os.environ.get("PMC_TRAINER", "").strip().lower()
    has_together_key = bool(os.environ.get("TOGETHER_API_KEY"))

    if forced == "together" or (not forced and has_together_key):
        from pmc.train.together_trainer import together_train_fn
        return together_train_fn(config, train, output_dir, eval_completions, on_event=on_event)

    if forced == "mlx" or (not forced and _mlx_available()):
        from pmc.train.mlx_trainer import mlx_train_fn
        return mlx_train_fn(config, train, output_dir, eval_completions, on_event=on_event)

    from pmc.train.sft import run_sft
    return run_sft(config, train, output_dir, eval_completions)


def _mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _default_generator_factory(base_model: str, adapter_dir: Path | None) -> ModelGenerator:
    from pmc.eval.generator import HFGenerator
    return HFGenerator(base_model=base_model, adapter_dir=adapter_dir)


def _default_dataset_version() -> str:
    return "v" + datetime.now().strftime("%Y%m%d-%H%M%S")


__all__ = [
    "BenchmarksFactory",
    "GeneratorFactory",
    "PMCPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStatus",
    "TrainFn",
]
