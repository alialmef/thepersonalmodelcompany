# Current Architecture Audit

Date: 2026-05-22

This is a working map of how the repo actually fits together today, with the
integration gaps that matter before using PMC as a real first-user product.

## Executive Summary

The repo has the right ingredients for a frontier personal-model product:
native Mac ingestion, local graph extraction, curation, memory, Together AI
training, MLX/local serving, a Together serving engine, verification, action
traces, and a Next/Tauri frontend.

The main problem is not missing ambition. The main problem is connection:
several critical user flows are still wired for UX validation or local dev,
not for "train my real personal model and use it."

The highest-priority fix is to make the production path explicit:

```
Mac ingestion -> curate -> memory -> Together fine-tune -> verification -> Together serving -> chat/actions
```

Today, that path is much closer to real: Together training and Together
serving are connected, desktop chat streams directly to FastAPI, and the
desktop connect flow now starts a real candidate training run by default.
The remaining work is to make runtime capabilities visible and unify the two
laptop-world memory paths.

## Actual System Map

### Frontend

- `web/app/connect/page.tsx`
  - Native/Tauri source connection.
  - Calls Rust ingestion commands for iMessage, Notes, Mail, Documents.
  - Starts backend pipeline with `POST /v1/users/{id}/runs`.
  - Desktop default now starts real training (`NEXT_PUBLIC_PMC_RUN_MODE` can
    force `real` or keep browser/demo dry).
  - Deployment is blocked behind private verification; the trained run is
    registered only as a candidate until the user approves it.

- `web/app/curate/page.tsx`
  - Subscribes to `/runs/{job}/events`.
  - Renders `reading_source_found` events from the backend.
  - Moves to `/train` once memory/pipeline events complete.

- `web/app/train/page.tsx`
  - Subscribes to the same run event stream.
  - Renders `checkpoint_sample` events emitted by Together training.
  - Current issue: if `skip_train` is true, no baseline/final samples appear.

- `web/app/eval/page.tsx`
  - Loads private verification prompts from `/eval/prompts`.
  - Persists user judgments.
  - Calls `/runs/{run_id}/promote` when verification is good enough.

- `web/app/chat/page.tsx` and `web/app/api/chat/route.ts`
  - Web chat still uses the AI SDK through a Next route.
  - Desktop/Tauri chat now streams directly from backend
    `/v1/chat/completions`, because static export has no Next API routes.

- `web/app/actions/page.tsx`
  - Now wired to action capabilities, proposal review, simulate/execute/undo,
    and laptop-world scan.
  - `/actions` is protected by middleware.

### Desktop

- `desktop/src/lib.rs`
  - Exposes Tauri commands for iMessage, Apple Mail, Notes, document upload,
    Full Disk Access settings, and graph kickoff.

- `desktop/src/extract/*`
  - Broad Mac graph extraction: calendar, contacts, files, mail enrichment,
    notes enrichment, photos, reminders, Safari, music, call history.

- `desktop/src/graph/*` and `desktop/src/synthesis/*`
  - Entity graph, watermarks, episodes, open loops, themes, entity resolution.

This is stronger than the current frontend makes it look. The laptop-world
substrate exists in Rust already; the new Python `pmc/world` scanner should
align with it rather than become a competing memory path.

### Backend Pipeline

- `pmc/serve/api.py`
  - FastAPI app.
  - Owns upload/status/run/eval/chat endpoints.
  - Now mounts action and world route modules.
  - Exposes `/v1/runtime/capabilities` so the frontend can distinguish real
    Together training from local/dev fallback before starting a run.

- `pmc/orchestrator/pipeline.py`
  - Real end-to-end pipeline:
    - ingest
    - curate
    - memory
    - split train/holdout
    - train
    - eval
    - gate
    - deploy
  - Uses verification feedback as future training signal.
  - Can block deployment on private verification.

- `pmc/train/together_trainer.py`
  - Production fine-tuning path.
  - Uses Together fine-tuning API.
  - Default production model path resolves to Kimi K2 Instruct on Together.
  - Emits `checkpoint_sample`, job status, heartbeats, and adapter download
    events.
  - Writes `remote.json` containing Together remote handles.

- `pmc/train/mlx_trainer.py`
  - Local Apple Silicon dev path.

- `pmc/serve/engine_together.py`
  - Intended Together hosted inference path.
  - Routes directly to Together `output_model` IDs from `remote.json`.
  - Can also route adapter IDs against the recorded Together base model.

- `scripts/dev.sh`
  - Local default prefers MLX if installed, else MockEngine.
  - `PMC_INFERENCE=together` starts the backend with `TogetherEngine`.

## Together AI Reality Check

Yes, Together is already in the repo:

- Training: `pmc/train/together_trainer.py`
- Serving: `pmc/serve/engine_together.py`
- Pipeline trainer selection: `_default_train_fn()` prefers Together when
  `TOGETHER_API_KEY` is present or `PMC_TRAINER=together`.
- Deploy supervisor: samples Together output when `remote.json` exists.

The connection status now:

1. Desktop onboarding starts real candidate training by default.
2. Server startup can choose Together inference with `PMC_INFERENCE=together`.
3. The registry promotes Together `remote.json` handles into
   `AdapterRecord.metadata`.
4. `TogetherEngine` routes directly to the fine-tuned `output_model`.
5. Tauri chat uses direct backend streaming.

That means a Together fine-tune can now become the candidate model used by
verification and, after approval, active chat.

## P0 Integration Fixes

1. **Make run mode explicit**
   - Done at the route level: desktop defaults to real training; browser/demo
     can stay dry.
   - Done at the API contract level: runtime capabilities report training and
     inference providers without exposing secrets.
   - Still needed: visible runtime/provider state in the UI.

2. **Wire Together serving end to end**
   - Done: read `adapter/remote.json` during bundle registration.
   - Done: store `provider=together`, `output_model`, and `job_id` in
     `AdapterRecord.metadata`.
   - Done: update `TogetherEngine` to use `output_model` directly when present.
   - Done: relax `PMCServer` base-model equality for engines that can serve remote
     fine-tuned model IDs.

3. **Fix desktop chat transport**
   - Done: Tauri static export no longer depends on `/api/chat`.
   - Desktop chat streams direct backend `/v1/chat/completions`.

4. **Protect new app routes**
   - Done: `/actions` is in middleware protected paths.

5. **Unify laptop-world memory**
   - Rust graph extraction and Python `pmc/world` scanner should converge into
     one memory substrate.
   - The product needs one world index, not two competing local context maps.

## P1 Architecture Cleanup

1. Split `pmc/serve/api.py` further into route modules.
2. Split `pmc/orchestrator/pipeline.py` into stage services once the flow is
   stable.
3. Move frontend direct `fetch()` calls onto `web/lib/api/client.ts` so routes
   use one backend contract.
4. Surface runtime capabilities in the UI:
   - training provider selected
   - inference provider selected
   - Together key present
   - Anthropic key present
   - OpenAI embeddings key present
   - MLX available
5. Add a product-level state machine:
   - fresh
   - connected
   - reading
   - training
   - verifying
   - active
   - retrain_needed

## Current Confidence

Backend foundations: high.

Frontend-to-backend product continuity: medium.

Production Together path: connected for train -> candidate registration ->
verification -> direct/streaming serve. Runtime capability visibility is still
missing.

Desktop/Tauri ingestion: stronger than the web flow currently exposes.

Action runtime: now has a clean backend foundation, but should be integrated
into chat and memory after runtime capabilities/state are visible.
