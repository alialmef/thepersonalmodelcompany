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
from pmc.eval.gate import EvalGate, EvalGateConfig, GateDecision
from pmc.eval.generator import ModelGenerator
from pmc.eval.privacy_eval import PrivacyBenchmark
from pmc.eval.runner import EvalSuiteResult, PersonalEvalRunner
from pmc.ingest.normalize import Normalizer
from pmc.orchestrator.data_source import DataSource
from pmc.schema.conversation import Completion, Conversation
from pmc.schema.training import TrainingConfig
from pmc.schema.user import DataManifest, User
from pmc.serve.registry import AdapterRegistry
from pmc.storage.artifact_store import ArtifactStore, new_run_id
from pmc.storage.audit import AuditLog
from pmc.storage.deletion import DeletionManager
from pmc.storage.user_store import UserStore
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

            curate_result, dataset_version = self.stage_curate(config)
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

            train_ds, holdout_ds = self._split(curate_result.completions, config)
            plan, training_result = self.stage_train(config, train_ds, holdout_ds)
            result.training_plan = plan
            result.training_result = training_result

            if config.dry_run or config.skip_train or training_result is None:
                result.status = "dry_run" if config.dry_run else "completed"
                result.notes = "Training was skipped (dry_run or skip_train)"
                return self._finalize(result, t0)

            run_id = self._save_bundle(config, training_result, curate_result, dataset_version)
            result.run_id = run_id

            if config.skip_eval:
                result.status = "completed"
                result.notes = "Eval skipped"
                if not config.skip_deploy:
                    self.stage_deploy(config, run_id, force=True)
                    result.deployed = True
                return self._finalize(result, t0)

            eval_result = self.stage_eval(config, run_id, holdout_ds, training_result)
            result.eval_result = eval_result

            decision = self.stage_gate(config, eval_result)
            result.gate_passed = decision.deploy
            result.gate_decision_reason = decision.reason

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

    def stage_train(
        self,
        config: PipelineConfig,
        train_completions: list[Completion],
        holdout_completions: list[Completion],
    ) -> tuple[TrainingPlan, SFTRunResult | None]:
        """Plan and (unless dry_run) run SFT. Returns (plan, result-or-None)."""
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
        result = train_fn(training_config, train_completions, output_dir, holdout_completions)
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

    def stage_deploy(
        self, config: PipelineConfig, run_id: str, *, force: bool = False
    ) -> bool:
        """Promote a run to active and register for serving."""
        if config.skip_deploy and not force:
            return False
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


def _default_train_fn(
    config: TrainingConfig,
    train: list[Completion],
    output_dir: Path,
    eval_completions: list[Completion] | None,
) -> SFTRunResult:
    """Pick the best trainer available on this host.

    Preference order:
      1. `PMC_TRAINER=mlx` env override (forces MLX even if MPS unavailable)
      2. MLX-LM on Apple Silicon — fastest local path, our default for V0
      3. PyTorch + TRL + PEFT (HF stack) — requires torch installed,
         works on any GPU including remote
    """
    import os
    forced = os.environ.get("PMC_TRAINER", "").strip().lower()

    if forced == "mlx" or _mlx_available():
        from pmc.train.mlx_trainer import mlx_train_fn
        return mlx_train_fn(config, train, output_dir, eval_completions)

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
