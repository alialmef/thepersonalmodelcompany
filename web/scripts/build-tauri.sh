#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build the Next.js static export for the Tauri desktop bundle.
#
# Next.js's static export (`output: "export"`) can't render API routes that
# use server-only APIs like `cookies()` or DB connections. Our auth + chat
# routes do exactly that, but they're not meant to ship in the desktop bundle
# anyway — the Mac app calls those endpoints over the network at
# thepersonalmodelcompany.com.
#
# So: temporarily move `app/api/` aside, run the static build, then put it
# back. The resulting `web/out/` has the UI but no API handlers, which is
# exactly what we want for the desktop bundle.
# -----------------------------------------------------------------------------

set -eu

cd "$(dirname "$0")/.."

API_DIR="app/api"
STASH_DIR="app/_api_stash_for_tauri"

# Ensure we put the api back even if next build fails.
restore() {
  if [ -d "$STASH_DIR" ]; then
    mv "$STASH_DIR" "$API_DIR"
    echo "[build-tauri] restored $API_DIR"
  fi
}
trap restore EXIT INT TERM

if [ -d "$API_DIR" ]; then
  echo "[build-tauri] stashing $API_DIR for the static build"
  mv "$API_DIR" "$STASH_DIR"
fi

# Also stash middleware — middleware doesn't work in static export and would
# cause warnings/failures.
if [ -f "middleware.ts" ]; then
  mv "middleware.ts" "_middleware_stash_for_tauri.ts"
fi
restore_middleware() {
  if [ -f "_middleware_stash_for_tauri.ts" ]; then
    mv "_middleware_stash_for_tauri.ts" "middleware.ts"
    echo "[build-tauri] restored middleware.ts"
  fi
}
# Restore middleware on exit too — chain the trap
trap "restore; restore_middleware" EXIT INT TERM

echo "[build-tauri] running next build (static export)"
NEXT_PUBLIC_TAURI_BUILD=true next build
echo "[build-tauri] build complete — web/out/ ready"
