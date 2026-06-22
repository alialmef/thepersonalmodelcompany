# The Personal Model Company

> Have your AIs know you based on what you actually do — not what
> you've typed into the chat window.

PMC is building a continuous picture of your computer life, and helps the
agent you work with get to know you better.

It runs locally on your Mac, watches your messages, mail, photos, calendar,
voice memos, browsing, code, and locations as they change, and structures
that flow into a personal knowledge graph + a continuously-refined model
of you. Then it exposes that model to whichever agent you're using —
Claude, Cursor, Continue, anything that speaks MCP — so the first prompt
you type already lands with someone who knows you.

You own the picture. It lives on your machine. You can take it anywhere.

## Talk to it from a terminal

If you just want to try it: one command installs the CLI, you point it
at your own API key (Anthropic / OpenAI / Gemini / OpenRouter), and
you're in a REPL with your agent.

```bash
curl -sSL https://raw.githubusercontent.com/alialmef/thepersonalmodelcompany/main/scripts/install.sh | bash
pmc configure   # pick provider + paste your key
pmc chat        # start talking
```

`pmc chat` loads everything the Mac app has structured about you (the
graph + synthesis layers — people, places, themes, live threads,
patterns, drift, voice-memo transcripts) as the agent's context. If
the Mac app isn't installed yet the graph is empty and the agent will
say so plainly.

Slash commands inside `pmc chat`:

```
/threads    list what the agent thinks is alive in your life right now
/patterns   list the steady-state patterns of how you spend time
/drift      list what's different about now vs the recent past
/context    print the full context block the agent sees
/reload     re-read the graph (after the Mac app has ingested more)
/reset      clear conversation history (keep the context block)
/quit
```

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
