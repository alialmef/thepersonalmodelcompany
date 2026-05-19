"""`pmc` CLI — basic commands to drive the pipeline from the shell.

Commands:
    pmc ingest --user U --source PATH[:KIND] [--source ...]
    pmc curate --user U
    pmc plan --user U
    pmc run --user U --source PATH[:KIND] ... [--dry-run] [--skip-eval] [--skip-deploy]
    pmc status [--user U]
    pmc delete --user U [--source SRC] [--scope sources|all_data|full]

The `--root` flag controls where storage lives (default `./pmc_data`).
Heavy training/inference deps are not required for `ingest`, `curate`, `plan`,
`status`, or `delete`. `run` (without `--dry-run`) requires `pmc[train]`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pmc.orchestrator.data_source import (
    DataSource,
    DataSourceKind,
    document_source,
    imessage_source,
    mbox_source,
    text_source,
    whatsapp_source,
)
from pmc.orchestrator.monitor import Monitor
from pmc.orchestrator.pipeline import PMCPipeline, PipelineConfig
from pmc.storage.artifact_store import ArtifactStore
from pmc.storage.audit import AuditLog
from pmc.storage.deletion import DeletionManager, DeletionScope
from pmc.storage.user_store import UserStore


def _wire(root: Path) -> tuple[PMCPipeline, Monitor, DeletionManager]:
    user_store = UserStore(root)
    artifact_store = ArtifactStore(root)
    audit_log = AuditLog(root)
    deletion = DeletionManager(user_store, artifact_store, audit_log)
    pipeline = PMCPipeline(
        user_store=user_store,
        artifact_store=artifact_store,
        audit_log=audit_log,
        deletion=deletion,
    )
    monitor = Monitor(user_store, artifact_store, audit_log, deletion=deletion)
    return pipeline, monitor, deletion


def _parse_source_spec(spec: str) -> DataSource:
    """Parse `PATH[:KIND][@id]`. KIND defaults to `text` for files, else inferred."""
    raw, _, source_id = spec.partition("@")
    if ":" in raw:
        path_str, _, kind = raw.partition(":")
    else:
        path_str, kind = raw, ""
    path = Path(path_str).expanduser()
    inferred = _infer_kind(path, kind)
    source_id = source_id or None

    if inferred == DataSourceKind.EMAIL_MBOX:
        raise SystemExit(
            f"mbox source {path!r} requires --user-email; use the Python API for now"
        )
    if inferred == DataSourceKind.WHATSAPP:
        raise SystemExit(
            f"WhatsApp source {path!r} requires --user-name; use the Python API for now"
        )
    if inferred == DataSourceKind.DOCUMENT:
        return document_source(path, source_id)
    if inferred == DataSourceKind.IMESSAGE:
        return imessage_source(path, source_id)
    return text_source(path, source_id)


def _infer_kind(path: Path, explicit: str) -> DataSourceKind:
    if explicit:
        try:
            return DataSourceKind(explicit)
        except ValueError as e:
            raise SystemExit(f"Unknown source kind {explicit!r}") from e
    suffix = path.suffix.lower()
    if suffix in {".pdf", ".docx"}:
        return DataSourceKind.DOCUMENT
    if suffix == ".db" or path.name == "chat.db":
        return DataSourceKind.IMESSAGE
    if suffix in {".mbox", ".mbx"}:
        return DataSourceKind.EMAIL_MBOX
    return DataSourceKind.TEXT


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline, _, _ = _wire(Path(args.root))
    config = PipelineConfig(
        user_id=args.user,
        user_name=args.user_name or "",
        user_email=args.user_email or "",
        base_model=args.base_model,
        data_sources=[_parse_source_spec(s) for s in args.source],
    )
    pipeline.record_user_profile(config)
    count = pipeline.stage_ingest(config)
    print(json.dumps({"user_id": args.user, "raw_items_ingested": count}, indent=2))
    return 0


def cmd_curate(args: argparse.Namespace) -> int:
    pipeline, _, _ = _wire(Path(args.root))
    config = PipelineConfig(
        user_id=args.user,
        base_model=args.base_model,
        data_sources=[],
    )
    result, version = pipeline.stage_curate(config)
    print(json.dumps({
        "user_id": args.user,
        "dataset_version": version,
        "output_completions": result.stats.output_completions,
        "dropped_short": result.stats.dropped_short,
        "dropped_duplicate": result.stats.dropped_duplicate,
        "dropped_low_quality": result.stats.dropped_low_quality,
    }, indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    pipeline, _, _ = _wire(Path(args.root))
    versions = pipeline.user_store.list_dataset_versions(args.user)
    if not versions:
        raise SystemExit(f"No curated dataset for user {args.user!r}. Run `pmc curate` first.")
    version = args.version or versions[-1]
    completions = pipeline.user_store.load_curated_dataset(args.user, version)
    config = PipelineConfig(
        user_id=args.user,
        base_model=args.base_model,
        data_sources=[],
        dry_run=True,
        dataset_version=version,
    )
    training_config = pipeline._resolve_training_config(config, len(completions))
    from pmc.train.sft import plan_sft
    plan = plan_sft(training_config, completions, None)
    print(plan.model_dump_json(indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    pipeline, _, _ = _wire(Path(args.root))
    config = PipelineConfig(
        user_id=args.user,
        user_name=args.user_name or "",
        user_email=args.user_email or "",
        base_model=args.base_model,
        data_sources=[_parse_source_spec(s) for s in args.source],
        dry_run=args.dry_run,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        skip_deploy=args.skip_deploy,
    )
    result = pipeline.run(config)
    print(result.model_dump_json(indent=2))
    return 0 if result.status not in {"failed"} else 1


def cmd_status(args: argparse.Namespace) -> int:
    _, monitor, _ = _wire(Path(args.root))
    if args.user:
        status = monitor.user_status(args.user)
        print(status.model_dump_json(indent=2))
    else:
        system = monitor.system_status()
        users = monitor.list_users()
        print(json.dumps({
            **system.model_dump(mode="json"),
            "users": users,
        }, indent=2, default=str))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    _, _, deletion = _wire(Path(args.root))
    try:
        scope = DeletionScope(args.scope)
    except ValueError as e:
        raise SystemExit(f"Unknown scope {args.scope!r}") from e
    result = deletion.delete(
        args.user,
        scope=scope,
        sources=args.source,
        notes=args.notes or "",
    )
    print(result.model_dump_json(indent=2))
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pmc", description="Personal Model Company CLI")
    p.add_argument("--root", default="./pmc_data", help="Storage root directory")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Load raw data into a user's store")
    ing.add_argument("--user", required=True)
    ing.add_argument("--user-name")
    ing.add_argument("--user-email")
    ing.add_argument("--source", action="append", required=True, help="PATH[:KIND][@source_id]")
    ing.set_defaults(func=cmd_ingest)

    cur = sub.add_parser("curate", help="Curate ingested raw items into a dataset")
    cur.add_argument("--user", required=True)
    cur.set_defaults(func=cmd_curate)

    plan = sub.add_parser("plan", help="Estimate cost/steps for the next training run")
    plan.add_argument("--user", required=True)
    plan.add_argument("--version", help="Dataset version to plan against (default: latest)")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="Run the end-to-end pipeline")
    run.add_argument("--user", required=True)
    run.add_argument("--user-name")
    run.add_argument("--user-email")
    run.add_argument("--source", action="append", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-train", action="store_true")
    run.add_argument("--skip-eval", action="store_true")
    run.add_argument("--skip-deploy", action="store_true")
    run.set_defaults(func=cmd_run)

    stat = sub.add_parser("status", help="Show user or system status")
    stat.add_argument("--user")
    stat.set_defaults(func=cmd_status)

    delete = sub.add_parser("delete", help="Delete user data (sources / all data / full)")
    delete.add_argument("--user", required=True)
    delete.add_argument("--scope", default="sources",
                        choices=[s.value for s in DeletionScope])
    delete.add_argument("--source", action="append")
    delete.add_argument("--notes")
    delete.set_defaults(func=cmd_delete)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
