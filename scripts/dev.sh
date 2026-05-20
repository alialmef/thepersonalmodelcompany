#!/usr/bin/env bash
# Local dev — start the full PMC stack with one command.
#
# What this does:
#   1. Starts the Python FastAPI backend on :8000 (MockEngine, no GPU needed)
#   2. Starts `bun run dev` in web/ on :3000
#   3. Launches `cargo tauri dev` which opens a native Mac window
#
# Hit Ctrl+C to shut everything down cleanly.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Source Rust env if not already in PATH
if ! command -v cargo >/dev/null 2>&1; then
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    else
        echo "✗ Rust not installed. Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
fi

if ! cargo tauri --version >/dev/null 2>&1; then
    echo "✗ Tauri CLI not installed. Run: cargo install tauri-cli --version '^2.0'"
    exit 1
fi

for cmd in bun uv; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "✗ $cmd not installed. See README for setup."
        exit 1
    fi
done

# Persistent dev data directory so state survives restarts
export PMC_DEV_ROOT="${PMC_DEV_ROOT:-$HOME/.pmc-dev}"
mkdir -p "$PMC_DEV_ROOT"

# Make sure web/ has its deps
if [ ! -d "$REPO_ROOT/web/node_modules" ]; then
    echo "→ Installing web/ dependencies..."
    (cd "$REPO_ROOT/web" && bun install)
fi

echo
echo "  Personal Model Company — local dev"
echo "  ───────────────────────────────────"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  Storage:  $PMC_DEV_ROOT"
echo "  Ctrl+C to stop everything."
echo

BACKEND_PID=""
BUN_PID=""

cleanup() {
    local code=$?
    echo
    echo "→ Stopping services..."
    [ -n "$BUN_PID" ] && kill "$BUN_PID" 2>/dev/null || true
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "$BUN_PID" ] && wait "$BUN_PID" 2>/dev/null || true
    [ -n "$BACKEND_PID" ] && wait "$BACKEND_PID" 2>/dev/null || true
    echo "✓ Stopped."
    exit $code
}
trap cleanup EXIT INT TERM

# --- 1. Start backend on :8000 ---
echo "→ Starting Python backend on :8000..."
PMC_DEV_ROOT="$PMC_DEV_ROOT" \
uv run python -c "
import os
from pmc.serve.api import run
from pmc.serve.server import PMCServer
from pmc.serve.registry import AdapterRegistry

# Prefer MLX (Apple Silicon, local GPU) when installed. Falls back to the
# deterministic MockEngine so the stack still boots on machines without MLX.
try:
    from pmc.serve.engine_mlx import MLXEngine, DEFAULT_MLX_BASE
    engine = MLXEngine(base_model=DEFAULT_MLX_BASE)
    print(f'[serve] engine=MLXEngine base={DEFAULT_MLX_BASE}', flush=True)
except Exception as e:
    from pmc.serve.engine import MockEngine
    engine = MockEngine(base_model='mock/base')
    print(f'[serve] engine=MockEngine (mlx unavailable: {e})', flush=True)

# Wire memory + identity into chat if OPENAI_API_KEY is set (for embeddings).
memory_provider = None
if os.environ.get('OPENAI_API_KEY'):
    try:
        from pmc.serve.memory_context import MemoryContextProvider
        from pmc.memory.embeddings import OpenAIEmbeddings
        from pmc.storage.paths import StoragePaths
        memory_provider = MemoryContextProvider(
            paths=StoragePaths(os.path.join(os.environ['PMC_DEV_ROOT'], 'storage')),
            embeddings=OpenAIEmbeddings(),
        )
        print('[serve] memory_provider=enabled (OpenAI embeddings)', flush=True)
    except Exception as e:
        print(f'[serve] memory_provider=disabled ({e})', flush=True)
else:
    print('[serve] memory_provider=disabled (set OPENAI_API_KEY to enable recall)', flush=True)

root = os.environ['PMC_DEV_ROOT']
registry = AdapterRegistry(os.path.join(root, 'registry'))
server = PMCServer(registry, engine, memory_provider=memory_provider)
run(
    server,
    storage_root=os.path.join(root, 'storage'),
    cors_origins=[
        'http://localhost:3000',
        'tauri://localhost',
        'http://tauri.localhost',
        'https://tauri.localhost',
    ],
    port=8000,
)
" &
BACKEND_PID=$!

echo -n "→ Waiting for backend..."
for i in {1..40}; do
    if curl -fs http://localhost:8000/healthz >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    echo -n "."
    sleep 0.5
    if [ "$i" -eq 40 ]; then
        echo
        echo "✗ Backend didn't come up in 20s. Check logs above."
        exit 1
    fi
done

# --- 2. Start Next.js dev server on :3000 ---
echo "→ Starting Next.js dev on :3000..."
(cd "$REPO_ROOT/web" && bun run dev) &
BUN_PID=$!

echo -n "→ Waiting for Next.js..."
for i in {1..60}; do
    if curl -fs http://localhost:3000 >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    echo -n "."
    sleep 0.5
    if [ "$i" -eq 60 ]; then
        echo
        echo "✗ Next.js didn't come up in 30s. Check logs above."
        exit 1
    fi
done

# --- 3. Launch Tauri (foreground) ---
# tauri.conf.json has no beforeDevCommand — we already started bun dev above.
echo "→ Building + launching Mac app window..."
cd "$REPO_ROOT/desktop"
exec cargo tauri dev
