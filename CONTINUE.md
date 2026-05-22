# Where we left off — 2026-05-21 ~16:15

## What was just done

- **iMessage decoder bug fixed.** The previous decoder was over-reading by
  one byte — every message came out with a junk leading char (e.g.
  `eAccept slack invite...` instead of `Accept slack invite...e`). Root
  cause: the scan window treated the `+` separator (0x2b = 43) as a length
  byte and read 43 chars starting at the actual length byte's position.
  Fixed by anchoring on the `+` separator explicitly, then parsing the
  length encoding immediately after. See `decode_attributed_body` in
  `desktop/src/ingest/imessage.rs`.

- **Real-blob validation.** CLI smoke test now decodes 27,453 messages
  from chat.db (vs. 28,215 before the fix — small drop because we no
  longer accept malformed near-empty strings). Total content went from
  1.05M → 1.23M chars; messages over 100 chars went from 1,409 → 2,097.
  Sample bodies look clean (no leading junk, no truncated tails).

- **Decoder regression tests** added in
  `desktop/src/ingest/imessage.rs::tests` covering: short messages, the
  length-43 over-read bug, and long messages (`\x81 <u16>` encoding).
  All 10 unit tests pass.

- **Mac app rebuilt + reinstalled** with the fix. New DMG at
  `web/public/downloads/PersonalModelCompany.dmg` (signed, notarization
  submission in flight via background shell — see `/tmp/notary-*.log`).
  DMG is 15.1 MB. App bundle is clean (one binary: `pmc-desktop`).

- **Dev CLI moved to `examples/`.** Was previously in `src/bin/`, which
  Tauri's bundler treats as "must ship inside the .app". Examples are
  invisible to the bundler. Run with
  `cargo run --example test_imessage --release` (was
  `--bin test_imessage`). The doc comment + this file reflect the new
  path.

## What we're in the middle of

Nothing blocking. The full iMessage path (decoder → ingest → curate →
train) is now ready to test end-to-end through the actual app UI. Click
"Connect iMessage" in the running app; should see ~27k items land in
`~/.pmc-dev/storage/users/<user-id>/raw/imessage.jsonl` (was 49 before
the decoder fix).

User's local `pmcUserId` for this session:
`11c7ace3-f395-4353-8acb-d6f7a2ec6113`.

## How to pick up next

1. **Verify end-to-end ingest.** Open the installed app, walk through
   the Connect step, confirm message count.
2. **Submit notarization stapling once submission returns "Accepted":**
   ```
   xcrun stapler staple "web/public/downloads/PersonalModelCompany.dmg"
   ```
3. **Commit + push** the fix + DMG. Uncommitted changes:
   - `desktop/Cargo.toml`
   - `desktop/src/ingest/imessage.rs`
   - `desktop/examples/test_imessage.rs` (moved from `src/bin/`)
   - `web/public/downloads/PersonalModelCompany.dmg`

## Other in-flight pieces (unchanged from last session)

- **MLX training validator fix** (commit `7a6b8a2`) — accepts
  `adapters.safetensors` (MLX-LM naming) alongside PEFT's
  `adapter_model.safetensors`.

- **/curate page** no longer auto-skips to /train. Live scoreboard
  reads `raw_source_breakdown` from `/v1/users/{id}/status`.

## Don't repeat these mistakes

- The Apple ID app-specific password lives ONLY in the macOS Keychain
  under profile `pmc-notary`. NEVER put it in a file or on the command
  line. Use `--keychain-profile pmc-notary`.

- `bash scripts/dev.sh` kills the backend if Next.js dev fails to start
  within 30s — fine for full dev but BAD when you only want the backend
  alive. To start backend only, see the inline command in the previous
  CONTINUE.md (kept in git history).

- **Don't put dev CLIs in `src/bin/`** — Tauri's macOS bundler will try
  to ship them inside the `.app`. Use `examples/` instead.

- `cargo tauri build` produces both `.app` and `.dmg`. The DMG bundling
  step has been intermittently failing (`bundle_dmg.sh` exit 1) but the
  `.app` is what we install locally, so the failure is OK for dev.

## Key reference: Apple typedstream NSAttributedString format

For future decoder work, the layout we care about is:

```
NSString \x01 \x94|\x95 \x84 \x01 + <length-encoding> <utf-8 bytes>
```

- `+` (0x2b) is the typedstream `char` type code — marks the start of
  the length-prefixed C-string field.
- `<length-encoding>` is one of:
  - single byte `n` in 0x01..0x7f      → length `n`
  - `\x81` then 2-byte LE u16          → length up to 65,535
  - `\x82` then 4-byte LE u32          → length up to ~4 GB

The byte right before `+` varies (`\x94` for short messages, `\x95` for
long ones) — that's a class-reference index, not a fixed sentinel. Don't
match it.

Reference parser: `reagentX/imessage-database` (Rust crate, full
parser, more permissive about edge cases than we are).
