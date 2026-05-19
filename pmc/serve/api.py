"""FastAPI app — OpenAI-compatible HTTP surface for PMCServer.

Endpoints:
- `POST /v1/chat/completions` — OpenAI Chat Completions, routes by `model` (= user_id).
                                 Set `"stream": true` for SSE token streaming.
- `GET  /v1/models`           — list registered user adapters
- `GET  /v1/models/{user_id}` — info about one user's model
- `GET  /v1/models/{user_id}/export?adapter_only=false` — download bundle/adapter zip
- `DELETE /v1/models/{user_id}?delete_files=false` — unregister (and optionally delete)
- `GET  /healthz`             — health check

Web-app endpoints (require `storage_root` to be set):
- `POST /v1/users/{user_id}/sources/upload` — multipart upload of a single
                                              source file; ingests + persists
- `GET  /v1/users/{user_id}/status`         — user dashboard state (Monitor)
- `DELETE /v1/users/{user_id}/sources/{source_id}` — drop one source

FastAPI is imported inside `create_app()` so this module is importable without
fastapi/uvicorn installed. Install with `pip install pmc[serve]`.

We deliberately do NOT use `from __future__ import annotations` — FastAPI
needs to resolve UploadFile/Form types at route-registration time, and the
closure-scoped lazy import doesn't survive deferred annotation evaluation.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from pmc.serve.schema import ChatCompletionRequest
from pmc.serve.server import PMCServer


def create_app(
    server: PMCServer,
    *,
    storage_root: Optional[Path | str] = None,
    cors_origins: Optional[list[str]] = None,
) -> Any:
    """Build a FastAPI application bound to a PMCServer instance.

    `storage_root` — when provided, enables the per-user endpoints (upload,
    status, source delete). Without it, only the OpenAI-compatible endpoints
    are mounted.

    `cors_origins` — list of allowed origins for browser-side calls. Pass
    `["http://localhost:3000"]` for local Next.js dev.
    """
    try:
        from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, File
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise ImportError(
            "fastapi is required to build the API. Install with `pip install pmc[serve]`."
        ) from e

    app = FastAPI(title="PMC — Personal Model Company", version="0.1.0")

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ----- health + chat completions (no storage_root needed) -----

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "base_model": server.engine.base_model,
            "num_models": len(server.registry),
            "storage_enabled": storage_root is not None,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Any:
        if request.stream:
            # Pre-validate before the streaming generator starts — errors raised
            # mid-iteration become 500s, not the 404/400 they should be.
            user_id = request.user or request.model
            try:
                record = server.registry.require(user_id)
                if record.base_model != server.engine.base_model:
                    raise ValueError(
                        f"Adapter for {user_id!r} expects base "
                        f"{record.base_model!r}, engine is serving "
                        f"{server.engine.base_model!r}"
                    )
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            stream = server.chat_stream(request)

            def sse() -> Any:
                for chunk in stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # disable nginx buffering
                    "Connection": "keep-alive",
                },
            )
        try:
            response = server.chat(request)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return response.model_dump()

    @app.get("/v1/models")
    def list_models() -> Any:
        return server.list_models().model_dump()

    @app.get("/v1/models/{user_id}")
    def get_model(user_id: str) -> Any:
        try:
            return server.get_model(user_id).model_dump()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/v1/models/{user_id}/export")
    def export_model(
        user_id: str,
        adapter_only: bool = Query(default=False),
    ) -> Any:
        try:
            record = server.registry.require(user_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        tmp = Path(tempfile.mkdtemp(prefix=f"pmc-export-{user_id}-"))
        suffix = "adapter" if adapter_only else "bundle"
        zip_path = tmp / f"{user_id}-{suffix}.zip"
        server.export_model(user_id, zip_path, adapter_only=adapter_only)
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_path.name,
            headers={"X-PMC-Base-Model": record.base_model},
        )

    @app.delete("/v1/models/{user_id}")
    def delete_model(
        user_id: str,
        delete_files: bool = Query(default=False),
    ) -> Any:
        removed = server.delete_model(user_id, delete_files=delete_files)
        if not removed:
            raise HTTPException(status_code=404, detail=f"No model {user_id!r}")
        return {"deleted": user_id, "files_deleted": delete_files}

    # ----- web-app endpoints (require storage_root) -----

    if storage_root is not None:
        from datetime import datetime
        import asyncio
        import json as _json

        from pmc.orchestrator.data_source import DataSourceKind
        from pmc.orchestrator.monitor import Monitor
        from pmc.orchestrator.pipeline import PMCPipeline, PipelineConfig
        from pmc.orchestrator.scheduler import JobScheduler, JobStatus
        from pmc.schema.base_models import ModelTier, list_specs, spec_for_tier
        from pmc.storage.artifact_store import ArtifactStore
        from pmc.storage.audit import AuditLog
        from pmc.storage.deletion import DeletionManager
        from pmc.storage.founders import FounderTracker
        from pmc.storage.user_store import UserStore

        user_store = UserStore(storage_root)
        artifact_store = ArtifactStore(storage_root)
        audit_log = AuditLog(storage_root)
        deletion = DeletionManager(user_store, artifact_store, audit_log)
        monitor = Monitor(
            user_store,
            artifact_store,
            audit_log,
            deletion=deletion,
            registry=server.registry,
        )
        pipeline = PMCPipeline(
            user_store=user_store,
            artifact_store=artifact_store,
            audit_log=audit_log,
            deletion=deletion,
            registry=server.registry,
        )
        scheduler = JobScheduler(pipeline, max_workers=1)
        # Map job_id → submitted_at timestamp so SSE can window events by job
        job_started: dict[str, datetime] = {}
        # First-100-users free Try-tier training (per project-founder-pricing memory)
        founders = FounderTracker(storage_root)

        @app.post("/v1/users/{user_id}/sources/upload")
        async def upload_source(
            user_id: str,
            file: UploadFile = File(...),
            kind: str = Form(...),
            source_id: Optional[str] = Form(None),
            user_emails: Optional[str] = Form(None),
            user_names: Optional[str] = Form(None),
        ) -> dict[str, Any]:
            try:
                kind_enum = DataSourceKind(kind)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Unknown kind: {kind!r}") from e
            if kind_enum == DataSourceKind.RAW:
                raise HTTPException(
                    status_code=400, detail="RAW kind is for in-process use only"
                )

            # Save uploaded file to a temp location with the original extension
            suffix = Path(file.filename or "upload").suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                try:
                    source = _build_source(
                        kind_enum, tmp_path, source_id, user_emails, user_names
                    )
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e

                items = list(source.ingest())
                effective_source_id = source.derived_source_id()
                count = user_store.save_raw_items(user_id, effective_source_id, items)
                audit_log.log(
                    user_id,
                    stage="ingest",
                    event="source_uploaded",
                    data={
                        "source_id": effective_source_id,
                        "kind": kind_enum.value,
                        "items": count,
                        "filename": file.filename or "",
                    },
                )
                return {
                    "raw_items_ingested": count,
                    "source_id": effective_source_id,
                    "kind": kind_enum.value,
                    "total_raw_items": user_store.count_raw_items(user_id),
                }
            finally:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

        @app.get("/v1/users/{user_id}/status")
        def get_user_status(user_id: str) -> Any:
            return monitor.user_status(user_id).model_dump(mode="json")

        @app.post("/v1/users/{user_id}/sources/items")
        def upload_items(user_id: str, body: dict[str, Any]) -> dict[str, Any]:
            """Direct batch upload of pre-parsed RawItems.

            Used by the native desktop app: parses locally (e.g. iMessage chat.db
            → RawItems), then POSTs the batch here instead of uploading a file.

            Body:
              kind: str       — for audit log + dispatch (e.g. "imessage")
              source_id: str  — partition key for storage
              items: list[RawItem JSON]
            """
            from pmc.ingest.base import RawItem
            kind = body.get("kind", "")
            source_id = body.get("source_id", "")
            raw = body.get("items", [])

            if not source_id:
                raise HTTPException(status_code=400, detail="source_id required")
            if not isinstance(raw, list) or not raw:
                raise HTTPException(status_code=400, detail="items required (non-empty list)")

            parsed: list[Any] = []
            for i, entry in enumerate(raw):
                try:
                    parsed.append(RawItem.model_validate(entry))
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid item at index {i}: {e}",
                    ) from e

            count = user_store.save_raw_items(user_id, source_id, parsed)
            audit_log.log(
                user_id,
                stage="ingest",
                event="items_pushed_via_native",
                data={
                    "source_id": source_id,
                    "kind": kind,
                    "items": count,
                },
            )
            return {
                "ingested": count,
                "source_id": source_id,
                "kind": kind,
                "total_raw_items": user_store.count_raw_items(user_id),
            }

        # ----- Pricing (founder-aware) -----

        @app.get("/v1/pricing")
        def get_pricing() -> dict[str, Any]:
            """Current effective pricing — Try tier flips to $0 if founder slots remain.

            Frontend renders the pricing card from this. No countdown / urgency
            messaging — just "Free" or "$19/mo" based on `is_free_now`.
            """
            slots_remaining = founders.slots_remaining()
            try_tier_free = slots_remaining > 0
            tiers = []
            for spec in list_specs():
                effective_price = (
                    0.0 if (try_tier_free and spec.tier == ModelTier.TRY)
                    else spec.subscription_usd_per_month
                )
                tiers.append({
                    "tier": spec.tier.value,
                    "hf_id": spec.hf_id,
                    "display_name": spec.display_name,
                    "list_price_usd_per_month": spec.subscription_usd_per_month,
                    "effective_price_usd_per_month": effective_price,
                    "is_free_now": effective_price == 0.0,
                    "agentic_grade": spec.agentic_grade.value,
                    "context_length": spec.context_length,
                })
            return {
                "tiers": tiers,
                "founder": {
                    "try_tier_free": try_tier_free,
                    # Internal — UI doesn't show this as a countdown.
                    "slots_remaining": slots_remaining,
                    "slots_total": founders.total_slots,
                },
            }

        # ----- Pipeline runs (Act 3: curate + train + eval as a single job) -----

        @app.post("/v1/users/{user_id}/runs")
        def submit_run(
            user_id: str,
            body: Optional[dict[str, Any]] = None,
        ) -> dict[str, Any]:
            """Submit a pipeline run for this user.

            Body fields (all optional):
              base_model: str  — override default tier
              skip_train, skip_eval, skip_deploy, dry_run: bool
              user_name, user_email: str — profile metadata if first time
            """
            body = body or {}
            config = PipelineConfig(
                user_id=user_id,
                user_name=body.get("user_name", ""),
                user_email=body.get("user_email", ""),
                base_model=body.get("base_model", PipelineConfig.model_fields["base_model"].default),
                data_sources=[],  # reuses already-ingested raw items from UserStore
                skip_train=body.get("skip_train", False),
                skip_eval=body.get("skip_eval", False),
                skip_deploy=body.get("skip_deploy", False),
                dry_run=body.get("dry_run", False),
            )

            # Founder grant: first 100 users training on Try tier get it free.
            # Idempotent — repeat calls don't double-grant.
            try_hf_id = spec_for_tier(ModelTier.TRY).hf_id
            grant = None
            if config.base_model == try_hf_id and not config.dry_run:
                grant = founders.grant_if_available(user_id)
                if grant.granted_now:
                    audit_log.log(
                        user_id,
                        stage="user",
                        event="founder_granted",
                        data={
                            "slots_remaining": grant.slots_remaining,
                            "total_slots": founders.total_slots,
                        },
                    )

            job = scheduler.submit(config)
            job_started[job.id] = job.submitted_at
            return {
                "job_id": job.id,
                "user_id": user_id,
                "status": job.status.value,
                "submitted_at": job.submitted_at.isoformat(),
                "founder": {
                    "is_founder": grant.is_founder if grant else founders.is_founder(user_id),
                    "training_free": bool(grant and grant.is_founder),
                },
            }

        @app.get("/v1/users/{user_id}/runs/{job_id}")
        def get_run(user_id: str, job_id: str) -> Any:
            job = scheduler.get(job_id)
            if job is None or job.user_id != user_id:
                raise HTTPException(status_code=404, detail=f"No job {job_id!r} for {user_id!r}")
            return job.model_dump(mode="json")

        @app.get("/v1/users/{user_id}/runs/{job_id}/events")
        def stream_run_events(user_id: str, job_id: str) -> Any:
            """SSE stream of audit events for this job.

            Strategy: filter the user's audit log by timestamp >= job.submitted_at.
            Polls the log every 0.5s and yields new events as they appear. Closes
            when the job reaches a terminal state (completed/failed/cancelled).
            """
            job = scheduler.get(job_id)
            if job is None or job.user_id != user_id:
                raise HTTPException(status_code=404, detail=f"No job {job_id!r} for {user_id!r}")
            since = job_started.get(job_id) or job.submitted_at

            async def event_stream() -> Any:
                last_emitted = since
                terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

                # First, replay any events that already happened since the job started.
                # (User may connect SSE after curate has already finished.)
                events = audit_log.events(user_id, since=last_emitted)
                for ev in events:
                    yield f"data: {ev.model_dump_json()}\n\n"
                    if ev.timestamp > last_emitted:
                        last_emitted = ev.timestamp

                # Then poll for new events until the job is terminal + we've flushed.
                idle_iterations = 0
                while True:
                    current_job = scheduler.get(job_id)
                    new_events = audit_log.events(user_id, since=last_emitted)
                    # Filter strictly > last_emitted (since= is inclusive)
                    new_events = [e for e in new_events if e.timestamp > last_emitted]
                    for ev in new_events:
                        yield f"data: {ev.model_dump_json()}\n\n"
                        if ev.timestamp > last_emitted:
                            last_emitted = ev.timestamp

                    if current_job and current_job.status in terminal:
                        # Send the final job summary and close
                        summary = {
                            "event": "job_finished",
                            "job_id": job_id,
                            "status": current_job.status.value,
                            "elapsed_seconds": (
                                (current_job.completed_at - current_job.started_at).total_seconds()
                                if current_job.started_at and current_job.completed_at else None
                            ),
                            "result": current_job.result.model_dump(mode="json") if current_job.result else None,
                        }
                        yield f"data: {_json.dumps(summary, default=str)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Backoff: 0.3s while events are flowing, up to 1s when idle
                    await asyncio.sleep(0.3 if new_events else min(1.0, 0.3 + 0.1 * idle_iterations))
                    idle_iterations = 0 if new_events else idle_iterations + 1
                    # Hard safety cap: 10 min of idle = bail
                    if idle_iterations > 600:
                        yield 'data: {"event":"timeout","message":"no events for 10 minutes"}\n\n'
                        yield "data: [DONE]\n\n"
                        return

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        @app.delete("/v1/users/{user_id}/sources/{source_id}")
        def delete_source(user_id: str, source_id: str) -> dict[str, Any]:
            count_before = user_store.count_raw_items(user_id, source_id)
            removed = user_store.delete_source(user_id, source_id)
            if not removed:
                raise HTTPException(
                    status_code=404, detail=f"No source {source_id!r} for user {user_id!r}"
                )
            # If the user had an active adapter, deleting data invalidates it
            # — clear the active pointer so they know retrain is needed.
            cleared = artifact_store.clear_active(user_id)
            audit_log.log(
                user_id,
                stage="delete",
                event="source_deleted_via_api",
                data={
                    "source_id": source_id,
                    "items_removed": count_before,
                    "active_cleared": cleared,
                },
            )
            return {
                "deleted_source": source_id,
                "items_removed": count_before,
                "active_cleared": cleared,
                "retrain_needed": cleared,
            }

    return app


def _build_source(
    kind: Any,  # DataSourceKind
    path: Path,
    source_id: str | None,
    user_emails: str | None,
    user_names: str | None,
) -> Any:  # DataSource
    """Construct the right DataSource for a kind + temp file path."""
    from pmc.orchestrator.data_source import (
        DataSourceKind,
        document_source,
        imessage_source,
        mbox_source,
        text_source,
        whatsapp_source,
    )

    if kind == DataSourceKind.TEXT:
        return text_source(path, source_id)
    if kind == DataSourceKind.DOCUMENT:
        return document_source(path, source_id)
    if kind == DataSourceKind.IMESSAGE:
        return imessage_source(path, source_id)
    if kind == DataSourceKind.EMAIL_MBOX:
        emails = [e.strip() for e in (user_emails or "").split(",") if e.strip()]
        if not emails:
            raise ValueError("email_mbox kind requires user_emails")
        return mbox_source(path, emails, source_id)
    if kind == DataSourceKind.WHATSAPP:
        names = [n.strip() for n in (user_names or "").split(",") if n.strip()]
        if not names:
            raise ValueError("whatsapp kind requires user_names")
        return whatsapp_source(path, names, source_id)
    raise ValueError(f"Unsupported kind for upload: {kind}")


def run(
    server: PMCServer,
    *,
    storage_root: Path | str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    cors_origins: list[str] | None = None,
) -> None:
    """Convenience wrapper: build the app and start uvicorn."""
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn is required to run the server. Install with `pip install pmc[serve]`."
        ) from e
    uvicorn.run(
        create_app(server, storage_root=storage_root, cors_origins=cors_origins),
        host=host,
        port=port,
    )
