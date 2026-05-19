# Personal Model Company — Mac Desktop App

Tauri 2 shell that wraps the existing `web/` Next.js frontend as a native
Mac app. Native Rust modules handle local data ingestion (iMessage, Apple
Notes, Apple Mail, WhatsApp) and call the Python FastAPI backend for
training, serving, and the rest of the pipeline.

## Prerequisites

```bash
# Rust (one-time, see https://rustup.rs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Tauri CLI
cargo install tauri-cli --version "^2.0"

# Verify
cargo tauri --version
```

Bun (for the `web/` frontend) is also needed — see `web/README.md`.

## Dev mode

From the project root:

```bash
cd desktop
cargo tauri dev
```

This will:
1. Auto-start `bun run dev` in `../web/` (configured via `beforeDevCommand`)
2. Wait for `http://localhost:3000` to respond
3. Launch the Mac app window pointing at that URL
4. Hot-reload as you edit either Rust (`desktop/src/`) or Next.js (`web/app/`)

If you'd rather run Next.js manually in a separate terminal, point
`devUrl` at it without `beforeDevCommand` triggering a second instance.

Make sure the **backend** is also running:

```bash
# In another terminal, at project root:
TMPDIR_E2E=/tmp/pmc-data \
uv run --extra dev --extra ingest --extra serve python -c "
import os
from pmc.serve.api import run
from pmc.serve.server import PMCServer
from pmc.serve.registry import AdapterRegistry
from pmc.serve.engine import MockEngine
root = os.environ['TMPDIR_E2E']
reg = AdapterRegistry(os.path.join(root, 'registry'))
server = PMCServer(reg, MockEngine(base_model='mock/base'))
run(server, storage_root=os.path.join(root, 'storage'),
    cors_origins=['http://localhost:3000', 'tauri://localhost'],
    port=8000)
"
```

## Production build

Builds a signed, notarized `.dmg` for distribution.

```bash
cd desktop
cargo tauri build
```

Requires:
- Apple Developer Program enrollment ($99/yr)
- Developer ID Application certificate installed in Keychain
- App-specific password from appleid.apple.com set as `APPLE_PASSWORD` env
- Apple ID set as `APPLE_ID` env
- Team ID set as `APPLE_TEAM_ID` env

The first build will fail until icons exist — see `icons/README.md`.

## Project layout

```
desktop/
  Cargo.toml             Rust deps (tauri 2, rusqlite, reqwest, etc.)
  tauri.conf.json        Tauri configuration (window, bundle, build hooks)
  build.rs               Tauri build script
  src/
    main.rs              Entry point (calls into lib.rs)
    lib.rs               Tauri commands exposed to the webview
    ingest/              Native ingestion modules (TODO: iMessage, Notes, Mail, WhatsApp)
  capabilities/
    default.json         Tauri 2 capability config (permissions for the main window)
  icons/                 App icons (.icns for Mac)
  .gitignore             target/, gen/, *.dmg, signing keys
```

## How the pieces connect

```
   Mac app (Tauri shell)
        │
        ├── webview ─── loads web/ Next.js
        │                  │
        │                  └── window.__TAURI__ detection → app-mode UI
        │
        ├── tauri::command app_info, ping, ...   ← exposed to JS
        │
        ├── Rust ingest modules (iMessage, Notes, Mail, WhatsApp)
        │       ↓ HTTP POST raw items
        │
        └── Python FastAPI backend (pmc.serve.api)
                │
                ├── UserStore, ArtifactStore, AuditLog
                ├── PMCPipeline (curate → train → eval)
                └── Together AI / Modal for compute
```

## What's next

After this scaffold runs `cargo tauri dev` cleanly:

1. Native ingestion modules — `src/ingest/imessage.rs`, `notes.rs`, `mail.rs`, `whatsapp.rs`. Each reads from a Mac-only data source and POSTs RawItems to the FastAPI backend. Full Disk Access prompt for iMessage.
2. Code signing + notarization pipeline (see top-level project-platform-pivot memory)
3. Auto-update (Tauri-updater pointing at Cloudflare R2)
4. Brand icon assets
