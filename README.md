# The Personal Model Company

> Train an AI model on your own writing. You own it. Host it. Take it anywhere.

PMC is a **native Mac app** that trains a personal LoRA adapter on top of an
open-weights base model (Llama 3.1 8B / Qwen 3.6 27B / Kimi K2.6). Your
writing — iMessage, Apple Notes, Mail, WhatsApp, browser history, documents —
stays on your Mac until *you* trigger training. The trained adapter, the
style profile, the audit log, and an export bundle are all yours.

## See it locally

One command starts the full stack (backend + frontend + Mac app window):

```bash
./scripts/dev.sh
```

It will:
1. Start the Python FastAPI backend on `:8000` (MockEngine — no GPU needed)
2. Start `cargo tauri dev`, which auto-starts Next.js on `:3000` and opens a
   native Mac window pointing at it

Prerequisites: Rust (`rustup`), Tauri CLI (`cargo install tauri-cli`), Bun, uv.
The script checks for these and tells you what's missing.

## Project layout

```
thepersonalmodelcompany/
  pmc/                Python backend — ingest/curate/train/eval/serve/storage
  tests/              324 backend tests
  web/                Next.js frontend (shared web marketing + Tauri webview)
  desktop/            Tauri 2 Mac shell + Rust native ingestion (iMessage, ...)
  scripts/
    dev.sh            One-command local dev launcher
  pyproject.toml      uv-managed Python deps with optional extras
  personal-model-company-analysis.md   The original architecture spec
```

## Status (May 2026)

- **Backend:** 324 tests passing. Full pipeline (ingest → curate → train →
  eval → serve) works end-to-end against a MockEngine. SSE streaming for chat
  and job progress. Per-user storage with audit log + deletion → retrain.
- **Frontend:** Acts 1–2 live (landing, sign-in form, connect data). Three
  pricing tiers in the registry (Try $19 / Personal $79 / Frontier $299).
  First-100-free founder counter wired.
- **Desktop app:** Tauri 2 scaffold + iMessage native ingestion working.
  Reads `~/Library/Messages/chat.db`, handles Full Disk Access, batches to
  the backend.
- **Hosting:** Modal (training) + Together AI (multi-tenant LoRA serving) +
  Railway (marketing site + billing) + Cloudflare (DNS + CDN). Production
  not yet deployed.

See the `MEMORY.md` notes for the full set of design + product decisions.

## Run the backend tests

```bash
uv run --extra dev pytest -q
```

## Run the Rust tests

```bash
cd desktop && cargo test --lib
```

## Build a production Mac app

Requires Apple Developer Program + code signing setup (see
`desktop/README.md`). One-line build:

```bash
cd desktop && cargo tauri build
```

Produces a signed `.dmg` ready to upload to your distribution CDN.
