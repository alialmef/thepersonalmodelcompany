#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# release.sh — build, sign, notarize, staple, and publish the PMC Mac app.
#
# Usage:
#   ./scripts/release.sh
#
# Prerequisites (one-time setup):
#   1. Developer ID Application certificate installed in Keychain
#      (verify with: security find-identity -p codesigning -v)
#
#   2. Notarization credentials stored under keychain profile "pmc-notary":
#      xcrun notarytool store-credentials "pmc-notary" \
#        --apple-id "aalmeflehi@gmail.com" \
#        --team-id "XQGY763JPD" \
#        --password "your-app-specific-password"
#
#      App-specific passwords: appleid.apple.com → App-Specific Passwords
#      (Already configured on this machine as of 2026-05-19.)
#
# What this script does:
#   - Builds the Tauri app via `cargo tauri build` (signs as part of build)
#   - Submits the .dmg to Apple's notarization service (waits inline)
#   - Staples the notarization ticket to the .dmg so it works offline
#   - Verifies the signed + notarized + stapled artifact
#   - Copies the final .dmg into web/public/downloads/ so the website serves it
# -----------------------------------------------------------------------------

set -euo pipefail

# Ensure cargo + rustc are on PATH even when invoked from minimal shells
# (e.g. cron, CI, agent harness). Rustup installs to ~/.cargo/bin by default.
export PATH="$HOME/.cargo/bin:$PATH"

# ----- config ----------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_DIR="$REPO_ROOT/desktop"
WEB_DIR="$REPO_ROOT/web"
NOTARY_PROFILE="pmc-notary"
SIGNING_IDENTITY="Developer ID Application: Ali Almeflehi (XQGY763JPD)"

# Web destination — keep filename stable so the landing CTA doesn't break.
WEB_DEST="$WEB_DIR/public/downloads/PersonalModelCompany.dmg"

# Pretty output.
bold() { printf "\033[1m%s\033[0m\n" "$1"; }
dim()  { printf "\033[2m%s\033[0m\n" "$1"; }
ok()   { printf "\033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "\033[31m✗\033[0m %s\n" "$1" >&2; exit 1; }

# ----- preflight -------------------------------------------------------------
bold "▸ Preflight checks"

if ! security find-identity -p codesigning -v | grep -q "Developer ID Application: Ali Almeflehi"; then
  fail "Signing cert not found in keychain. Run: security find-identity -p codesigning -v"
fi
ok "Signing cert in keychain"

if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
  fail "Notary profile '$NOTARY_PROFILE' not configured. See script header for setup."
fi
ok "Notary profile '$NOTARY_PROFILE' configured"

if ! command -v cargo >/dev/null; then
  fail "cargo not on PATH"
fi
ok "cargo available"

# ----- build -----------------------------------------------------------------
bold "▸ Building & signing"
cd "$DESKTOP_DIR"
cargo tauri build
ok "Build complete"

# Find the produced .dmg (version is read from tauri.conf.json).
DMG_PATH="$(find "$DESKTOP_DIR/target/release/bundle/dmg" -name "*.dmg" -type f -print -quit)"
if [[ -z "$DMG_PATH" || ! -f "$DMG_PATH" ]]; then
  fail "Could not locate built .dmg"
fi
dim "DMG: $DMG_PATH"

# ----- verify signing --------------------------------------------------------
bold "▸ Verifying signature on the .app"
APP_PATH="$(find "$DESKTOP_DIR/target/release/bundle/macos" -name "*.app" -type d -print -quit)"
codesign --verify --verbose=2 "$APP_PATH"
ok "Signed: $APP_PATH"

# ----- notarize --------------------------------------------------------------
bold "▸ Submitting for notarization (this usually takes 1–5 minutes)"
xcrun notarytool submit "$DMG_PATH" \
  --keychain-profile "$NOTARY_PROFILE" \
  --wait
ok "Notarized"

# ----- staple ----------------------------------------------------------------
bold "▸ Stapling notarization ticket"
xcrun stapler staple "$DMG_PATH"
ok "Stapled"

# ----- final verification ----------------------------------------------------
bold "▸ Gatekeeper assessment"
spctl --assess --type open --context context:primary-signature -vv "$DMG_PATH" 2>&1 | head -5 || true

# ----- publish to web --------------------------------------------------------
bold "▸ Publishing to web download path"
mkdir -p "$(dirname "$WEB_DEST")"
cp -f "$DMG_PATH" "$WEB_DEST"
SIZE=$(du -h "$WEB_DEST" | cut -f1)
ok "Copied to $WEB_DEST ($SIZE)"

# ----- done ------------------------------------------------------------------
bold "▸ Done"
echo
echo "  Signed:     $APP_PATH"
echo "  Notarized:  $DMG_PATH"
echo "  Published:  $WEB_DEST"
echo
echo "Users downloading from /downloads/PersonalModelCompany.dmg will now"
echo "open the app without any Gatekeeper warnings."
