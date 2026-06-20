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

import json
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
        from fastapi import (
            Depends,
            FastAPI,
            File,
            Form,
            HTTPException,
            Query,
            Request,
            UploadFile,
        )
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise ImportError(
            "fastapi is required to build the API. Install with `pip install pmc[serve]`."
        ) from e

    from pmc.auth.middleware import optional_session

    app = FastAPI(title="PMC — Personal Model Company", version="0.1.0")

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Auth — email-anchored accounts + session tokens. The store is
    # attached to app.state so the FastAPI dependency in
    # pmc.auth.middleware can pick it up without globals. Always
    # mounted when storage_root is configured; in tests / sub-app
    # mode the caller can attach their own store before include_router.
    if storage_root is not None:
        from pmc.auth import AuthStore, auth_router
        app.state.auth_store = AuthStore(storage_root=Path(storage_root))
        app.include_router(auth_router)

    # ----- health + chat completions (no storage_root needed) -----

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "base_model": server.engine.base_model,
            "num_models": len(server.registry),
            "storage_enabled": storage_root is not None,
        }

    @app.get("/v1/runtime/capabilities")
    def runtime_capabilities() -> dict[str, Any]:
        """Report configured providers without exposing secrets."""
        import os

        forced_trainer = os.environ.get("PMC_TRAINER", "").strip().lower()
        together_key_present = bool(os.environ.get("TOGETHER_API_KEY"))
        openai_key_present = bool(os.environ.get("OPENAI_API_KEY"))
        anthropic_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
        try:
            from pmc.orchestrator.pipeline import _mlx_available
            mlx_available = _mlx_available()
        except Exception:
            mlx_available = False

        if forced_trainer:
            training_provider = forced_trainer
        elif together_key_present:
            training_provider = "together"
        elif mlx_available:
            training_provider = "mlx"
        else:
            training_provider = "hf"

        training_available = True
        unavailable_reason = None
        if training_provider == "together" and not together_key_present:
            training_available = False
            unavailable_reason = "TOGETHER_API_KEY is not set"
        elif training_provider == "mlx" and not mlx_available:
            training_available = False
            unavailable_reason = "MLX is not installed"

        engine_name = type(server.engine).__name__
        if engine_name == "TogetherEngine":
            inference_provider = "together"
        elif engine_name == "MLXEngine":
            inference_provider = "mlx"
        elif engine_name == "MockEngine":
            inference_provider = "mock"
        else:
            inference_provider = engine_name

        return {
            "training": {
                "provider": training_provider,
                "forced": forced_trainer or None,
                "available": training_available,
                "unavailable_reason": unavailable_reason,
                "together_key_present": together_key_present,
                "mlx_available": mlx_available,
            },
            "inference": {
                "provider": inference_provider,
                "engine": engine_name,
                "base_model": server.engine.base_model,
            },
            "memory": {
                "openai_key_present": openai_key_present,
            },
            "supervision": {
                "anthropic_key_present": anthropic_key_present,
            },
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Any:
        # Inject retrieved memory before handing the request to the
        # engine. If the user has a recall.db, the latest user message
        # becomes a retrieval query; the top fragments get prepended
        # as a system block. If memory is unavailable or empty the
        # request passes through unchanged. Memory is a boost, never
        # a gate — never block inference here.
        if storage_root is not None:
            try:
                from pmc.memory.recall.inject import inject_memory
                request = inject_memory(request, Path(storage_root))
            except Exception:
                pass

        if request.stream:
            # Pre-validate before the streaming generator starts — errors raised
            # mid-iteration become 500s, not the 404/400 they should be.
            user_id = request.user or request.model
            try:
                record = server.registry.require(user_id)
                if not server._engine_accepts(record):
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

        from pmc.actions.registry import build_default_action_registry
        from pmc.orchestrator.data_source import DataSourceKind
        from pmc.orchestrator.monitor import Monitor
        from pmc.orchestrator.pipeline import PMCPipeline, PipelineConfig
        from pmc.orchestrator.scheduler import JobScheduler, JobStatus
        from pmc.actions.service import ActionService
        from pmc.schema.base_models import ModelTier, list_specs, spec_for_tier
        from pmc.schema.conversation import Message, Role
        from pmc.schema.verification import (
            CandidateOrigin,
            JudgmentVerdict,
            PersonalProbe,
            ProbeCandidate,
            ProbeKind,
            UserJudgment,
        )
        from pmc.serve.routes.actions import build_actions_router
        from pmc.serve.routes.world import build_world_router
        from pmc.storage.action_store import ActionStore
        from pmc.storage.artifact_store import ArtifactStore
        from pmc.storage.audit import AuditLog
        from pmc.storage.deletion import DeletionManager
        from pmc.storage.founders import FounderTracker
        from pmc.storage.graph_store import GraphStore as PmcGraphStore
        from pmc.storage.redactions import RedactionsStore
        from pmc.storage.user_store import UserStore
        from pmc.storage.verification_store import VerificationStore
        from pmc.world import WorldStore

        user_store = UserStore(storage_root)
        artifact_store = ArtifactStore(storage_root)
        verification_store = VerificationStore(storage_root)
        action_store = ActionStore(storage_root)
        world_store = WorldStore(storage_root)
        audit_log = AuditLog(storage_root)
        redactions_store = RedactionsStore(storage_root)
        # GraphStore reads the typed-entity JSONL files the Rust
        # extractors wrote. When storage_root matches ~/.pmc-dev/storage
        # the backend reads exactly what Tauri wrote — no network hop.
        # In prod the Mac app will push entities over HTTP; the layout
        # stays identical.
        graph_store = PmcGraphStore(storage_root)
        deletion = DeletionManager(user_store, artifact_store, audit_log)
        action_service = ActionService(
            verification_store,
            audit_log,
            action_store=action_store,
            adapter_registry=build_default_action_registry(storage_root),
        )
        app.include_router(build_actions_router(action_service))
        app.include_router(build_world_router(world_store, audit_log))
        monitor = Monitor(
            user_store,
            artifact_store,
            audit_log,
            deletion=deletion,
            registry=server.registry,
            graph_store=graph_store,
        )
        pipeline = PMCPipeline(
            user_store=user_store,
            artifact_store=artifact_store,
            audit_log=audit_log,
            deletion=deletion,
            registry=server.registry,
            verification_store=verification_store,
        )
        scheduler = JobScheduler(pipeline, max_workers=1)
        # Map job_id → submitted_at timestamp so SSE can window events by job
        job_started: dict[str, datetime] = {}
        # First-100-users free Try-tier training (per project-founder-pricing memory)
        founders = FounderTracker(storage_root)

        # Billing — Stripe glue. Mounted unconditionally; individual
        # endpoints fail with a clear 503 if STRIPE_SECRET_KEY isn't set,
        # so the frontend can distinguish "not wired" from "declined".
        if hasattr(app.state, "auth_store"):
            from pmc.billing import BillingService
            from pmc.billing.router import build_billing_router
            app.state.billing_service = BillingService(app.state.auth_store)
            app.include_router(build_billing_router(founders=founders))

        # Agent (BYOM) — bring-your-own frontier-model. /v1/agent/* routes
        # let the user pick provider + model + paste their API key, and
        # proxies chat through whichever provider they chose. Individual
        # endpoints fail with a clear 503 if PMC_KEY_ENCRYPTION_SECRET
        # isn't set on the deploy.
        if hasattr(app.state, "auth_store"):
            from pmc.agent.router import build_agent_router
            app.include_router(build_agent_router())

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
            auth: Any = Depends(optional_session),
        ) -> dict[str, Any]:
            """Submit a pipeline run for this user.

            Gated on payment-or-founder. The caller must be one of:
              - an active subscriber (Stripe subscription_status =
                'active' or 'trialing'), OR
              - a founder (one of the first 100 to ever fire a run for
                this user_id), OR
              - the dry-run path (no Together spend, no gate)

            If a session is presented, the gate consults *the signed-in
            account's* subscription state — so when an account claims a
            pmc_user_id, the user_id's runs inherit the account's paid
            status. Anonymous calls (no session) only pass when the
            user_id has founder slots remaining.

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
                require_verification_to_deploy=body.get("require_verification_to_deploy", True),
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

            # Payment gate. Skipped for dry runs (validates pipeline
            # without spending compute). Otherwise: subscription OR
            # founder is required.
            if not config.dry_run:
                is_subscribed = bool(auth and auth.account.is_subscribed())
                is_founder_now = (
                    bool(grant and grant.is_founder)
                    or founders.is_founder(user_id)
                )
                if not (is_subscribed or is_founder_now):
                    raise HTTPException(
                        status_code=402,
                        detail={
                            "error": "payment_required",
                            "message": (
                                "Subscribe to train a model. "
                                "Hit POST /v1/billing/checkout to start."
                            ),
                            "checkout_url": "/v1/billing/checkout",
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

        @app.post("/v1/users/{user_id}/runs/{run_id}/promote")
        def promote_verified_run(user_id: str, run_id: str) -> dict[str, Any]:
            """Promote a trained run after private verification passes."""
            report = verification_store.trust_report(user_id)
            if report.readiness not in {"voice", "sandbox", "supervised"}:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "private verification has not passed",
                        "trust_report": report.model_dump(mode="json"),
                    },
                )
            if report.privacy_flags:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "privacy flags must be resolved before promotion",
                        "trust_report": report.model_dump(mode="json"),
                    },
                )
            bundle_dir = artifact_store.paths.bundle_dir(user_id, run_id)
            if not bundle_dir.is_dir():
                raise HTTPException(status_code=404, detail=f"No run {run_id!r} for {user_id!r}")

            pointer = artifact_store.set_active(
                user_id,
                run_id,
                notes="promoted after private verification",
            )
            registered = False
            if server.registry is not None:
                server.registry.register_bundle(bundle_dir)
                registered = True
            audit_log.log(
                user_id,
                stage="deploy",
                event="adapter_promoted_after_verification",
                run_id=run_id,
                data={
                    "readiness": report.readiness,
                    "registered": registered,
                },
            )
            return {
                "ok": True,
                "active": pointer.model_dump(mode="json"),
                "registered": registered,
                "trust_report": report.model_dump(mode="json"),
            }

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

        # ----- memory search (recall) -----
        #
        # Exposes the per-user vector store for semantic retrieval. Used by
        # the chat tool layer's `recall_memory` tool so the model can ask
        # "find me snippets about X" beyond what's already in its system
        # prompt context.
        #
        # Lazily initializes a MemoryContextProvider — if OPENAI_API_KEY isn't
        # configured, returns 503 with a clear reason instead of crashing.

        import os
        from pmc.serve.memory_context import MemoryContextProvider
        from pmc.storage.paths import StoragePaths

        _memory_paths = StoragePaths(storage_root)
        _memory_provider: MemoryContextProvider | None = None

        def _get_memory_provider() -> MemoryContextProvider | None:
            nonlocal _memory_provider
            if _memory_provider is not None:
                return _memory_provider
            if not os.environ.get("OPENAI_API_KEY"):
                return None
            from pmc.memory.embeddings import OpenAIEmbeddings
            _memory_provider = MemoryContextProvider(
                paths=_memory_paths,
                embeddings=OpenAIEmbeddings(),
            )
            return _memory_provider

        @app.post("/v1/users/{user_id}/memory/search")
        def memory_search(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            provider = _get_memory_provider()
            if provider is None:
                raise HTTPException(
                    status_code=503,
                    detail="memory search unavailable — set OPENAI_API_KEY",
                )
            query = payload.get("query", "")
            k = int(payload.get("k", 5))
            if not query:
                return {"results": []}

            ctx = provider.get(user_id)
            results = ctx.retrieve(query, k=k)
            return {
                "results": [
                    {
                        "source": r.item.source,
                        "text": r.item.text,
                        "score": r.score,
                        "when": (
                            datetime.fromtimestamp(r.item.created_at).isoformat()
                            if r.item.created_at
                            else None
                        ),
                    }
                    for r in results
                ]
            }

        # ----- eval / first-meeting (designed app arc) -----
        #
        # See design/app/README.md. The eval screen collects DPO-grade
        # preference signal between training and first chat. The first-meeting
        # endpoint returns a generated opening line conditioned on the user's
        # style profile.

        # Fallback situations only. The frontier path builds private probes
        # from held-out user data; these keep the first-user path usable before
        # a curated holdout exists.
        EVAL_SITUATIONS = [
            "Maya texted: 'free for dinner thursday?'",
            "A friend you haven't seen in months: 'how have you been? we should catch up'",
            "Boss: 'can you push the deadline by a week? need to talk about it'",
            "Group chat going off about a movie everyone loved except you",
            "Old friend who moved away: 'i'm in town this weekend — coffee?'",
        ]

        def _eval_response_for(
            user_id: str,
            situation: str,
            *,
            prompt: list[Message] | None = None,
        ) -> str:
            """Generate the model's draft reply to a situation.

            Uses the user's adapter via the chat completions path if it's
            registered; otherwise returns an obviously-placeholder so the UI
            still flows and the user can keep judging.
            """
            try:
                record = server.registry.require(user_id)
            except Exception:
                return "(your model isn't loaded yet — placeholder reply)"
            messages = (
                [{"role": msg.role.value, "content": msg.content} for msg in prompt]
                if prompt
                else [{"role": "user", "content": situation}]
            )
            try:
                text, _usage = server.engine.chat(
                    record=record,
                    messages=server._prepared_messages(user_id, messages),
                    max_tokens=120,
                    temperature=0.7,
                )
                return text.strip()
            except Exception as e:
                return f"(generation error: {e})"

        def _candidate_identity(user_id: str) -> tuple[CandidateOrigin, str | None]:
            try:
                record = server.registry.require(user_id)
                return CandidateOrigin.PERSONAL_MODEL, record.base_model
            except Exception:
                return CandidateOrigin.SYNTHETIC, None

        def _first_candidate_text(completion: Any) -> str | None:
            if not completion.candidates:
                return None
            candidate = completion.candidates[0]
            if not candidate.messages:
                return None
            text = "\n".join(m.content for m in candidate.messages if m.content.strip()).strip()
            return text or None

        def _build_eval_probes(user_id: str, *, limit: int = 5) -> list[PersonalProbe]:
            """Build private probes from held-out examples, then fallback scenarios."""
            probes: list[PersonalProbe] = []
            origin, model = _candidate_identity(user_id)

            for version in reversed(user_store.list_dataset_versions(user_id)):
                holdout = user_store.load_holdout(user_id, version)
                if not holdout:
                    continue
                for completion in holdout:
                    if len(probes) >= limit:
                        break
                    reference = _first_candidate_text(completion)
                    if not completion.conversation.messages or not reference:
                        continue
                    situation = " ".join(
                        msg.content.strip()
                        for msg in completion.conversation.messages
                        if msg.content.strip()
                    ).strip()
                    response = _eval_response_for(
                        user_id,
                        situation,
                        prompt=completion.conversation.messages,
                    )
                    probes.append(
                        PersonalProbe(
                            user_id=user_id,
                            kind=ProbeKind.VOICE,
                            prompt=completion.conversation.messages,
                            candidates=[
                                ProbeCandidate(
                                    origin=origin,
                                    text=response,
                                    model=model,
                                )
                            ],
                            reference=reference,
                            source_completion_id=str(completion.id),
                            dataset_version=version,
                            surface="eval",
                        )
                    )
                if probes:
                    break

            if probes:
                return probes

            for situation in EVAL_SITUATIONS[:limit]:
                response = _eval_response_for(user_id, situation)
                probes.append(
                    PersonalProbe(
                        user_id=user_id,
                        kind=ProbeKind.VOICE,
                        prompt=[Message(role=Role.USER, content=situation)],
                        candidates=[
                            ProbeCandidate(
                                origin=origin,
                                text=response,
                                model=model,
                            )
                        ],
                        surface="eval",
                        metadata={"fallback": True},
                    )
                )
            return probes

        def _probe_payload(probe: PersonalProbe) -> dict[str, Any]:
            first_candidate = probe.candidates[0] if probe.candidates else None
            return {
                "id": probe.id,
                "kind": probe.kind.value,
                "situation": probe.prompt_text(),
                "response": first_candidate.text if first_candidate else "",
                "reference": probe.reference,
                "source_completion_id": probe.source_completion_id,
                "dataset_version": probe.dataset_version,
                "candidates": [
                    {
                        "id": candidate.id,
                        "origin": candidate.origin.value,
                        "text": candidate.text,
                        "model": candidate.model,
                    }
                    for candidate in probe.candidates
                ],
            }

        def _normalize_verdict(value: Any) -> JudgmentVerdict:
            raw = str(value or "").strip().lower().replace("-", "_")
            aliases = {
                "": JudgmentVerdict.UNSURE,
                "yes": JudgmentVerdict.APPROVE,
                "good": JudgmentVerdict.APPROVE,
                "approve": JudgmentVerdict.APPROVE,
                "approved": JudgmentVerdict.APPROVE,
                "accept": JudgmentVerdict.APPROVE,
                "accepted": JudgmentVerdict.APPROVE,
                "no": JudgmentVerdict.REJECT,
                "bad": JudgmentVerdict.REJECT,
                "reject": JudgmentVerdict.REJECT,
                "rejected": JudgmentVerdict.REJECT,
                "edit": JudgmentVerdict.EDIT,
                "edited": JudgmentVerdict.EDIT,
                "choose": JudgmentVerdict.CHOOSE,
                "chosen": JudgmentVerdict.CHOOSE,
                "not_me": JudgmentVerdict.NOT_ME,
                "too_formal": JudgmentVerdict.TOO_FORMAL,
                "too_casual": JudgmentVerdict.TOO_CASUAL,
                "private": JudgmentVerdict.PRIVATE,
                "wrong": JudgmentVerdict.WRONG,
                "unsure": JudgmentVerdict.UNSURE,
            }
            if raw in aliases:
                return aliases[raw]
            try:
                return JudgmentVerdict(raw)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Unknown verdict: {value!r}") from e

        @app.get("/v1/users/{user_id}/eval/prompts")
        def eval_prompts(user_id: str) -> dict[str, Any]:
            probes = verification_store.list_probes(user_id, limit=20)
            if not probes:
                probes = _build_eval_probes(user_id)
                verification_store.save_probes(user_id, probes)
                audit_log.log(
                    user_id,
                    stage="eval",
                    event="verification_probes_built",
                    data={
                        "count": len(probes),
                        "kinds": sorted({probe.kind.value for probe in probes}),
                    },
                )
            return {
                "prompts": [_probe_payload(probe) for probe in probes[:5]],
                "trust_report": verification_store.trust_report(user_id).model_dump(mode="json"),
            }

        @app.post("/v1/users/{user_id}/eval/judgments")
        def eval_judgment(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            """Persist one structured judgment.

            This is training data, not UI state. Edits and pairwise choices can
            become SFT/DPO examples on the next run.
            """
            probe_id = str(
                payload.get("probeId")
                or payload.get("promptId")
                or payload.get("id")
                or ""
            )
            if not probe_id:
                raise HTTPException(status_code=400, detail="probeId or promptId required")
            probe = verification_store.get_probe(user_id, probe_id)
            if probe is None:
                raise HTTPException(status_code=404, detail=f"No verification probe {probe_id!r}")

            verdict = _normalize_verdict(payload.get("verdict"))
            chosen_candidate_id = payload.get("chosenCandidateId") or payload.get("candidateId")
            if chosen_candidate_id is None and probe.candidates:
                chosen_candidate_id = probe.candidates[0].id
            rejected_candidate_ids = payload.get("rejectedCandidateIds") or []
            if not isinstance(rejected_candidate_ids, list):
                rejected_candidate_ids = [str(rejected_candidate_ids)]

            judgment = UserJudgment(
                user_id=user_id,
                probe_id=probe_id,
                verdict=verdict,
                chosen_candidate_id=chosen_candidate_id,
                rejected_candidate_ids=[str(v) for v in rejected_candidate_ids],
                edited_text=payload.get("editedText") or payload.get("edited_text"),
                reason=payload.get("reason"),
                dimension=str(payload.get("dimension") or "overall"),
                score=payload.get("score"),
                metadata={
                    "surface": str(payload.get("surface") or "eval"),
                },
            )
            verification_store.append_judgment(user_id, judgment)
            report = verification_store.trust_report(user_id)
            audit_log.log(
                user_id, stage="eval", event="judgment_recorded",
                data={
                    "verdict": judgment.verdict.value,
                    "probe_id": judgment.probe_id,
                    "readiness": report.readiness,
                },
            )
            return {
                "ok": True,
                "judgment": judgment.model_dump(mode="json"),
                "trust_report": report.model_dump(mode="json"),
            }

        @app.get("/v1/users/{user_id}/verification/trust-report")
        def verification_trust_report(user_id: str) -> dict[str, Any]:
            return verification_store.trust_report(user_id).model_dump(mode="json")

        @app.get("/v1/users/{user_id}/verification/training-signal")
        def verification_training_signal(user_id: str) -> dict[str, Any]:
            preference = verification_store.preference_completions(user_id)
            action_sft = verification_store.action_sft_completions(user_id)
            return {
                "user_id": user_id,
                "preference_completions": len(preference),
                "action_sft_completions": len(action_sft),
                "total_completions": len(preference) + len(action_sft),
            }

        @app.get("/v1/users/{user_id}/first-meeting/opening")
        def first_meeting_opening(user_id: str) -> dict[str, Any]:
            """Generate the model's opening line — recognition → known-you-
            forever → humility → hand over control. Conditioned on the
            user's style profile so it lands in their own register.

            New: prefers the working-memory snapshot's `anticipation`
            items if present. Those are the 3-5 specific things Claude
            decided the agent might surface today. Picking the first
            anticipation item as the opening gives a specific, live,
            forward-looking opening line — exactly what the /meet spec
            calls for."""
            import json as _json2

            # First-best source: working-memory snapshot's anticipation.
            try:
                from pmc.memory.recall.store import RecallStore
                recall_path = user_store.paths.user_root(user_id) / "recall.db"
                if recall_path.is_file():
                    store = RecallStore(recall_path)
                    snap = store.latest_working_memory()
                    store.close()
                    if snap and snap.anticipation:
                        # Take the first anticipation item verbatim. The
                        # working-memory builder is already prompted to
                        # produce single-sentence, lowercase, in-register
                        # lines.
                        return {"line": snap.anticipation[0], "generated": True}
            except Exception:
                pass

            identity_path = user_store.paths.user_root(user_id) / "identity.json"
            facts: list[str] = []
            display_name = user_id
            if identity_path.exists():
                try:
                    data = _json2.loads(identity_path.read_text())
                    facts = list(data.get("style_facts") or [])
                    display_name = data.get("display_name") or user_id
                except Exception:
                    pass

            # If we have an adapter, ask it to write the opening itself.
            try:
                record = server.registry.require(user_id)
                style_fact_str = ", ".join(facts[:3]) if facts else "the way you write"
                instr = (
                    f"Write one short opening message to {display_name}. "
                    "Address them as 'you', never 'I'. Structure: "
                    "recognition → 'i've read everything you've ever written' "
                    "→ humility → hand control over with a question. "
                    f"Reflect: {style_fact_str}. One sentence each part, lowercase, no exclamation, no emoji."
                )
                messages = [{"role": "user", "content": instr}]
                text, _ = server.engine.chat(
                    record=record,
                    messages=server._prepared_messages(user_id, messages),
                    max_tokens=80,
                    temperature=0.6,
                )
                line = text.strip()
                if line:
                    return {"line": line, "generated": True}
            except Exception:
                pass

            # Fallback — the canonical opening, used when no adapter or
            # generation fails.
            return {
                "line": (
                    "there you are. i've read everything you've ever written — "
                    "i think i get you. where do you want to start?"
                ),
                "generated": False,
            }

        @app.post("/v1/users/{user_id}/reset")
        def reset_user(user_id: str) -> dict[str, Any]:
            """Wipe everything for this user and unregister any deployed
            adapter. Used by the "Start over" affordance in the app —
            takes the user back to a fresh state where /connect re-runs
            from scratch.

            Irreversible. The user is the only one allowed to call this
            for their own id; gating happens at the auth layer above
            (V0 dev backend trusts the caller)."""
            existed = user_store.delete_user(user_id)
            # Also unregister the adapter if it was deployed.
            unregistered = False
            if server is not None and server.registry is not None:
                try:
                    server.registry.unregister(user_id)
                    unregistered = True
                except KeyError:
                    pass
            audit_log.log(
                user_id, stage="user", event="user_reset",
                data={"existed": existed, "adapter_unregistered": unregistered},
            )
            return {
                "ok": True,
                "user_id": user_id,
                "data_wiped": existed,
                "adapter_unregistered": unregistered,
            }

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

        # ----- Knowledge update (pause / private / search-and-forget) -----
        #
        # Backing surface for the /knowledge-update screen in the Mac app.
        # Auto-opt-in means we ingest everything by default; this is the
        # contract on the other side: queryable, redactable, pauseable.

        @app.get("/v1/users/{user_id}/knowledge/overview")
        def knowledge_overview(user_id: str) -> dict[str, Any]:
            """Combined snapshot for the Manage screen: every source with
            its current item count + paused state, plus the list of
            active redactions."""
            sources = user_store.list_sources(user_id)
            paused_set = set(redactions_store.paused_source_ids(user_id))
            source_rows = []
            for sid in sources:
                kind = sid.split("-", 1)[0] if "-" in sid else sid
                try:
                    item_count = user_store.count_raw_items(user_id, sid)
                except Exception:
                    item_count = 0
                source_rows.append({
                    "source_id": sid,
                    "kind": kind,
                    "item_count": item_count,
                    "paused": sid in paused_set,
                })
            state = redactions_store.state(user_id)
            return {
                "sources": source_rows,
                "redactions": [r.model_dump(mode="json") for r in state.redactions],
                "paused_sources": [s.model_dump(mode="json") for s in state.paused_sources],
            }

        @app.post("/v1/users/{user_id}/knowledge/sources/{source_id}/pause")
        def pause_source(user_id: str, source_id: str) -> dict[str, Any]:
            redactions_store.pause_source(user_id, source_id)
            audit_log.log(
                user_id, stage="redact", event="source_paused",
                data={"source_id": source_id},
            )
            return {"ok": True, "source_id": source_id, "paused": True}

        @app.post("/v1/users/{user_id}/knowledge/sources/{source_id}/resume")
        def resume_source(user_id: str, source_id: str) -> dict[str, Any]:
            redactions_store.resume_source(user_id, source_id)
            audit_log.log(
                user_id, stage="redact", event="source_resumed",
                data={"source_id": source_id},
            )
            return {"ok": True, "source_id": source_id, "paused": False}

        @app.post("/v1/users/{user_id}/knowledge/redactions")
        def add_redaction(user_id: str, body: dict[str, Any]) -> dict[str, Any]:
            kind = (body.get("kind") or "").strip()
            value = (body.get("value") or "").strip()
            note = body.get("note")
            if kind not in ("person", "topic", "date_range"):
                raise HTTPException(
                    status_code=400, detail="kind must be person|topic|date_range",
                )
            if not value:
                raise HTTPException(status_code=400, detail="value required")
            try:
                r = redactions_store.add_redaction(
                    user_id, kind=kind, value=value, note=note,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            audit_log.log(
                user_id, stage="redact", event="redaction_added",
                data={"id": r.id, "kind": kind, "value_len": len(value)},
            )
            return r.model_dump(mode="json")

        @app.delete("/v1/users/{user_id}/knowledge/redactions/{redaction_id}")
        def remove_redaction(user_id: str, redaction_id: str) -> dict[str, Any]:
            ok = redactions_store.remove_redaction(user_id, redaction_id)
            if not ok:
                raise HTTPException(status_code=404, detail="redaction not found")
            audit_log.log(
                user_id, stage="redact", event="redaction_removed",
                data={"id": redaction_id},
            )
            return {"ok": True, "removed": redaction_id}

        @app.get("/v1/users/{user_id}/knowledge/search")
        def knowledge_search(
            user_id: str,
            q: str = "",
            limit: int = 50,
        ) -> dict[str, Any]:
            """V1 search: case-insensitive substring scan over raw items.
            Slow for users with deep ingest, but honest and useful as the
            redact UX. Phase 3 (the PKG) replaces this with hybrid
            vector + BM25 + graph traversal."""
            q_low = q.strip().lower()
            if not q_low:
                return {"query": q, "results": [], "truncated": False}
            results: list[dict[str, Any]] = []
            truncated = False
            try:
                for item in user_store.load_raw_items(user_id):
                    blob = (
                        item.model_dump_json() if hasattr(item, "model_dump_json")
                        else json.dumps(item, default=str)
                    )
                    if q_low in blob.lower():
                        results.append({
                            "id": getattr(item, "id", None),
                            "source_id": getattr(item, "source_id", None),
                            "kind": getattr(item, "kind", None),
                            "preview": blob[:280],
                            "timestamp": str(getattr(item, "timestamp", "")) or None,
                        })
                        if len(results) >= limit:
                            truncated = True
                            break
            except Exception:
                pass
            return {"query": q, "results": results, "truncated": truncated}

        @app.delete("/v1/users/{user_id}/knowledge/items/{item_id}")
        def forget_item(user_id: str, item_id: str) -> dict[str, Any]:
            """V1: not yet implemented — UserStore needs per-item delete.
            For now, returns 501 with a clear hint so the UI can disable
            the per-result Forget button until Phase 3 lands the PKG
            (which makes per-item delete trivial)."""
            raise HTTPException(
                status_code=501,
                detail={
                    "error": "not_implemented",
                    "message": (
                        "Per-item forget lands with Phase 3 (PKG). "
                        "Today: pause the source it came from, or mark "
                        "a Person/Topic private, or wipe and start over."
                    ),
                },
            )

        # ----- Graph snapshot (for the browser memory web) -----
        #
        # The Tauri webview reads the graph from the local Rust store
        # via `graph_snapshot` (faster, in-process). The browser can't
        # invoke Tauri commands, so it goes through this endpoint —
        # same shape, same data.

        @app.get("/v1/users/{user_id}/graph/snapshot")
        def graph_snapshot(user_id: str) -> dict[str, Any]:
            if not graph_store.exists(user_id):
                return {"nodes": [], "edges": []}
            return graph_store.snapshot(user_id)

        @app.get("/v1/users/{user_id}/graph/counts")
        def graph_counts(user_id: str) -> dict[str, Any]:
            if not graph_store.exists(user_id):
                return {"counts": {}, "total": 0, "raw_counts": {}}
            from pmc.storage.graph_store import NODE_KINDS
            # Quality-filtered counts are the surface contract — what
            # the agent/UI actually sees. Raw counts available for
            # debugging in `raw_counts`.
            quality = graph_store.quality_counts(user_id)
            raw = graph_store.counts(user_id)
            total = sum(quality.get(k, 0) for k in NODE_KINDS)
            return {"counts": quality, "total": total, "raw_counts": raw}

        # ----- Synthesis (agent-driven validation + reasoning) -----
        #
        # /confirm calls /synthesis/claims — backend asks the user's
        # configured frontier provider to produce 5-10 short factual
        # claims about the user with evidence, using the prompts
        # module's GENERATE_CLAIMS overlay. The Rust extractors
        # produced the graph; the user's agent produces the validation
        # questions over it.

        @app.post("/v1/users/{user_id}/synthesis/claims")
        @app.get("/v1/users/{user_id}/synthesis/claims")
        async def synthesis_claims(
            user_id: str,
            request: Request,
            auth: Any = Depends(optional_session),
        ) -> dict[str, Any]:
            """Returns {"claims": [...]} for the /confirm screen.
            V1: stub when the agent isn't configured / no graph data yet
            — surfaces a friendly placeholder so the flow stays unbroken.
            """
            from pmc.agent import crypto
            from pmc.agent.prompts import TaskKind, compose
            from pmc.agent.providers.base import (
                Message as AgentMessage,
                ProviderConfig as AgentProviderConfig,
                ProviderError as AgentProviderError,
            )
            from pmc.agent.providers.registry import get_provider

            store = getattr(app.state, "auth_store", None)
            cfg = store.get_provider_config(auth.account.id) if (store and auth) else None
            if not cfg or not crypto.is_configured():
                return _stub_claims("Agent not configured yet.")

            try:
                api_key = crypto.decrypt(cfg["api_key_ciphertext"])
            except Exception:
                return _stub_claims("Couldn't unlock your stored key.")

            provider = get_provider(cfg["provider"])
            if provider is None:
                return _stub_claims("Provider isn't recognized.")

            # Build the agent context from the actual structured graph
            # (people, places, themes, projects, apps, open loops) — not
            # just per-source counts. This is the load-bearing change:
            # claims become specific because the agent now sees real
            # entities, not bag-of-source-totals.
            from pmc.storage.graph_store import summarize_for_agent
            if graph_store.exists(user_id) and graph_store.total_node_count(user_id) > 0:
                graph_summary = summarize_for_agent(
                    graph_store, user_id, per_kind_limit=10,
                )
                user_msg = (
                    "Here is the user's structured personal knowledge graph "
                    "so far. Counts per entity kind + a representative "
                    "sample of each. Use this to generate concrete claims "
                    "about the user — name real people, real places, real "
                    "patterns. Cite the entity kind + a salient field as "
                    "evidence.\n\n"
                    + json.dumps(graph_summary, indent=2, default=str)
                )
            else:
                # Fall back to source counts if the graph is empty (no
                # extractor has produced typed entities yet, but raw
                # items may have been uploaded).
                try:
                    sources = user_store.list_sources(user_id)
                    counts = [
                        {"kind": (sid.split("-", 1)[0] if "-" in sid else sid),
                         "items": user_store.count_raw_items(user_id, sid)}
                        for sid in sources
                    ]
                except Exception:
                    counts = []
                if not counts:
                    return _stub_claims("Nothing ingested yet — give it a minute.")
                user_msg = (
                    "The structured graph isn't ready yet — generate the "
                    "best concrete claims you can about this person from "
                    "the shape of what's been ingested (per-source "
                    "counts).\n\n"
                    + json.dumps(counts, indent=2)
                )

            system_prompt = compose(auth.account.email, TaskKind.GENERATE_CLAIMS)
            try:
                resp = await provider.chat(
                    [AgentMessage(role="user", content=user_msg)],
                    config=AgentProviderConfig(
                        provider=cfg["provider"],
                        model=cfg["model"],
                        api_key=api_key,
                    ),
                    max_tokens=1500,
                    system=system_prompt,
                )
            except AgentProviderError as e:
                return _stub_claims(f"Agent error: {e}")

            text = resp.text.strip()
            if text.startswith("```"):
                # Strip markdown fences if the model added them despite
                # the prompt asking it not to.
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()
            try:
                parsed = json.loads(text)
            except Exception:
                return _stub_claims("Agent didn't return valid JSON.")
            claims = parsed.get("claims") if isinstance(parsed, dict) else None
            if not isinstance(claims, list):
                return _stub_claims("Agent didn't return any claims.")
            return {"claims": claims}

    return app


def _stub_claims(reason: str) -> dict[str, Any]:
    """Polite placeholder when /synthesis/claims can't reach a real agent
    response yet. The /confirm screen renders these the same way as
    real claims so the user can still click through and reach
    /right-now without a dead-end."""
    return {
        "claims": [
            {
                "claim": reason,
                "kind": "system",
                "evidence": [
                    {"source": "system", "summary": "Agent isn't producing claims yet."}
                ],
            }
        ]
    }


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
