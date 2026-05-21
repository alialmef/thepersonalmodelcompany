# Where we left off — 2026-05-21 ~15:00

## What was just done

- Built + installed Mac app at `/Applications/Personal Model Company.app`
  (binary timestamp `May 21 14:55`).
- Published the same build as `web/public/downloads/PersonalModelCompany.dmg`
  via commit `7a6b8a2`; Railway redeployed.
- Apple notarization submission `7481375c-2b32-4484-a9dc-11284d183add` was
  still "In Progress" when we last checked. DMG is signed (Developer ID) but
  NOT yet stapled. Gatekeeper does a live notary check on first launch.

## What we're in the middle of

**Validating the iMessage attributedBody decoder.** Up to commit `7a6b8a2`,
iMessage ingest pulled only ~49 messages out of ~20,000 in chat.db. Root
cause: macOS Ventura+ stores text in `m.attributedBody` (Apple typedstream
blob), not the `m.text` column we were SELECTing. A first-pass decoder is
in `desktop/src/ingest/imessage.rs` (`decode_attributed_body` +
`try_take_string`). It scans for the "NSString" class marker, then reads a
length-prefixed UTF-8 substring (handles short / 0x81 / 0x82 sentinels) and
filters known class-name strings.

The decoder is in the installed app and the published DMG. **It has not
been validated against the user's real chat.db yet.** The 49-count
screenshots from earlier are all from BEFORE the decoder existed.

## Why the session was paused

User chose to set up the **CLI dev loop** at `desktop/src/bin/test_imessage.rs`
so we can iterate on the decoder in ~10s per change instead of rebuilding
the whole Tauri app (~40s). The CLI runs the same code path as the Tauri
command (snapshot chat.db → open readonly → SELECT → decode → filter).

The shell hosting Claude Code runs inside **Cursor.app**, which does NOT
have Full Disk Access. Without FDA, `std::fs::copy(chat.db, /tmp/...)`
returns `PermissionDenied`. User went to System Settings → Privacy &
Security → Full Disk Access to grant Cursor.app FDA; macOS requires a
Cursor.app quit + relaunch to apply, which kills this session.

## How to pick back up (fresh session)

1. **Confirm FDA actually landed.** From a Bash tool call:
   ```
   /bin/cp "$HOME/Library/Messages/chat.db" /tmp/test-fda.db && \
     /bin/rm /tmp/test-fda.db && echo "FDA OK" || echo "FDA missing"
   ```

2. **Run the CLI smoke test:**
   ```
   cd desktop && \
     PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$HOME/.cargo/bin:$PATH" \
     cargo run --bin test_imessage --release
   ```
   Expected on first run: a multi-thousand item count (~10k+ messages),
   non-trivial avg length, sample bodies looking like real chat text.

3. **If the count is low or sample bodies look like garbage:** iterate on
   `decode_attributed_body` in `desktop/src/ingest/imessage.rs`. The
   typedstream format references that help:
   - Apple Typed Stream format: search "Apple typedstream NSAttributedString format"
   - `reagentX/imessage-database` Rust crate has a full parser for reference

4. **Once the count looks right:** rebuild + reinstall the Mac app so the
   user's actual installed app picks up the fix:
   ```
   cd desktop && cargo tauri build && \
     pkill -f "Personal Model Company"; sleep 1; \
     rm -rf "/Applications/Personal Model Company.app" && \
     cp -R "target/release/bundle/macos/Personal Model Company.app" /Applications/ && \
     xattr -dr com.apple.quarantine "/Applications/Personal Model Company.app" && \
     open "/Applications/Personal Model Company.app"
   ```

5. **Then publish:** copy the fresh DMG to `web/public/downloads/`, submit
   for notarization, commit + push.

## Other in-flight pieces

- **Notarization watcher** ran in a background subshell that died when the
  parent shell exited. The DMG at `web/public/downloads/PersonalModelCompany.dmg`
  is signed but unstapled. To re-attempt stapling later:
  ```
  xcrun stapler staple "web/public/downloads/PersonalModelCompany.dmg"
  ```
  If it returns "Worked", commit + push the stapled file.

- **MLX training validator fix** (commit `7a6b8a2`) — accepts
  `adapters.safetensors` (MLX-LM naming) alongside PEFT's
  `adapter_model.safetensors`. Should make local training pipeline runs
  succeed end-to-end now. Backend was restarted ~14:50 to pick up
  `pmc/orchestrator/monitor.py` (per-source breakdown) and
  `pmc/train/checkpoint.py` (validator fix).

- **/curate page** no longer auto-skips to /train. User must click Continue.
  Live scoreboard reads `raw_source_breakdown` from `/v1/users/{id}/status`.
  Style profile right column synthesizes lines from real curate stats when
  the backend doesn't ship a text summary.

## Don't repeat these mistakes

- The Apple ID app-specific password lives ONLY in the macOS Keychain
  under profile `pmc-notary`. NEVER put it in a file or on the command
  line. Use `--keychain-profile pmc-notary`.

- `bash scripts/dev.sh` kills the backend if Next.js dev fails to start
  within 30s — fine for full dev but BAD when you only want the backend
  alive. To start backend only:
  ```
  export PMC_DEV_ROOT="$HOME/.pmc-dev"
  nohup "$HOME/Desktop/Sites/thepersonalmodelcompany/.venv/bin/python" \
    -c "import os; from pmc.serve.api import run; from pmc.serve.server import PMCServer; \
        from pmc.serve.registry import AdapterRegistry; from pmc.serve.engine_mlx import MLXEngine, DEFAULT_MLX_BASE; \
        engine = MLXEngine(base_model=DEFAULT_MLX_BASE); \
        root = os.environ['PMC_DEV_ROOT']; \
        registry = AdapterRegistry(os.path.join(root, 'registry')); \
        server = PMCServer(registry, engine); \
        run(server, storage_root=os.path.join(root, 'storage'), \
            cors_origins=['http://localhost:3000','tauri://localhost','http://tauri.localhost','https://tauri.localhost'], \
            port=8000)" \
    > /tmp/pmc-backend.log 2>&1 &
  ```

- The user's local pmcUserId (in localStorage in the Tauri webview) for
  this session is **`11c7ace3-f395-4353-8acb-d6f7a2ec6113`**. Their raw
  data is at `~/.pmc-dev/storage/users/11c7ace3-f395-4353-8acb-d6f7a2ec6113/`.

- `cargo tauri build` produces both `.app` and `.dmg`. The DMG bundling
  step has been intermittently failing (`bundle_dmg.sh` exit 1) but the
  `.app` is what we install locally, so the failure is OK for dev. For
  publishing the website DMG, retry until it succeeds OR use a previous
  good DMG and replace the .app inside it.
