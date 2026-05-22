# Deploying the PMC Python backend to Railway

The PMC repo deploys **two services** on the same Railway project:

| Service | Root | Config | What it serves |
|---|---|---|---|
| `thepersonalmodelcompany` (existing) | `web/` | `web/railway.json` | Next.js marketing site at thepersonalmodelcompany.com |
| `thepersonalmodelcompany-backend` (new) | `/` (repo root) | `railway.json` | FastAPI Python backend — ingest, curate, memory, training proxy, inference proxy |

This doc covers setting up the new backend service so user-clicks-Train in the Mac app fires a real Together fine-tune from PMC's account.

## One-time setup

1. **Add a new service in the Railway project** — `thepersonalmodelcompany-backend`.
2. Connect it to the same GitHub repo.
3. In service settings, set **Root Directory** to `/` (or leave blank — defaults to repo root). The root-level `railway.json` configures the build via NIXPACKS with Python 3.12 + uv.
4. Set the environment variables below.
5. Attach a **Railway Volume** at mount path `/data` for persistent storage (`recall.db`, training bundles, audit logs). Without this, user data resets on every redeploy.
6. Deploy.

## Required environment variables

| Var | Required? | Why |
|---|---|---|
| `TOGETHER_API_KEY` | yes | Together fine-tuning + hosted inference. Bills to PMC's Together account. |
| `ANTHROPIC_API_KEY` | yes | Memory consolidation (Claude) + curate / memory / deploy supervisors. |
| `PMC_INFERENCE` | recommended | Set to `together` to force `TogetherEngine` on boot. (Auto-detects if unset.) |
| `PMC_TRAINER` | recommended | Set to `together` to force the Together trainer. (Auto-detects if unset.) |
| `PMC_DEV_ROOT` | recommended | Set to `/data` so the mounted volume holds storage. Default: `~/.pmc-dev`. |
| `PMC_PORT` | optional | HTTP port. Railway sets `$PORT` automatically; we honor `PMC_PORT` first. Default `8000`. |
| `PMC_CORS_ORIGINS` | optional | Comma-separated CORS allowlist. Defaults include localhost (dev) + thepersonalmodelcompany.com (prod). |
| `OPENAI_API_KEY` | optional | Enables the recall memory provider for in-chat context retrieval (legacy V0 path). |

## Verifying the deploy

Hit these in order:

```bash
# 1. Server is alive
curl https://<backend-url>/healthz

# 2. What's actually configured?
curl https://<backend-url>/v1/runtime/capabilities
```

You want to see, in the capabilities response:

```json
{
  "training": { "provider": "together", "available": true },
  "inference": { "provider": "together", "available": true }
}
```

If `inference.provider` is `"mlx"` or `"mock"`, the engine didn't pick Together — usually means `PMC_INFERENCE=together` isn't set or the Together SDK couldn't init.

If `training.provider` is `"mlx"` instead of `"together"`, the trainer fallback hit. Either `TOGETHER_API_KEY` isn't on this service or `PMC_TRAINER` is forcing MLX.

## Point the Mac app at the deployed backend

In `desktop/tauri.conf.json` and / or via env var at build time:

```bash
PMC_API_URL=https://<backend-url> bun --filter web run build
```

The Tauri binary picks up `PMC_API_URL` at build time. Without it, the app calls `http://localhost:8000` and you're talking to your dev Mac.

## What runs where

```
Mac app  (Tauri webview) ─────────────────────────────────────────┐
   │ /v1/users/<uid>/sources/items, /runs, /chat/completions, ... │
   │                                                              ▼
   │                              Railway: backend service (this doc)
   │                                  python -m pmc.serve
   │                                  └─ FastAPI on $PORT
   │                                  └─ TogetherEngine for /chat
   │                                  └─ together_train_fn for /runs
   │                                              │
   │                                              ▼
   │                                      Together AI
   │                                  (fine-tuning + inference,
   │                                   billed to PMC's account)
   │
   └─ also (separately) hits thepersonalmodelcompany.com Next.js
      site for marketing / download / auth surfaces.
```

The Python backend never sees the user's raw data unless the user explicitly clicks Train (which POSTs the curated dataset). Even then, the data is in transit only — it gets uploaded to Together, the adapter comes back, the curated dataset is discarded.
