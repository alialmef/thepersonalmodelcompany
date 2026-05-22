"""Production / dev entry point for the PMC backend.

    python -m pmc.serve

One source of truth for engine selection so prod (Railway) and dev
(scripts/dev.sh) can't drift. The dev script previously inlined this
logic in a heredoc; we now share it via `pmc.serve.__main__:main`.

Engine selection order:
  1. PMC_INFERENCE=together → TogetherEngine (hosted Kimi inference)
  2. PMC_INFERENCE=mock     → MockEngine (CI / smoke)
  3. TOGETHER_API_KEY set AND no Apple Silicon → TogetherEngine
  4. MLXEngine on Apple Silicon                → local Llama
  5. MockEngine fallback

The reason rule (3) checks for Apple Silicon: a hosted Linux box
can't run MLX, and a backend deployed to Railway with
TOGETHER_API_KEY set is almost certainly meant to serve hosted
inference. We default it to Together rather than silently falling
through to MockEngine.

Environment variables:
  PMC_INFERENCE         "together" | "mock" | unset (auto-detect)
  PMC_DEV_ROOT          Storage root (default: ~/.pmc-dev)
  PMC_PORT              HTTP port (default: 8000)
  PMC_CORS_ORIGINS      Comma-separated CORS allowlist
  TOGETHER_API_KEY      Required for TogetherEngine + Together training
  ANTHROPIC_API_KEY     Required for memory consolidation + supervisors
  OPENAI_API_KEY        Optional — enables the recall memory provider
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    # Production marketing site
    "https://thepersonalmodelcompany.com",
)


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}


def _together_base_model() -> str:
    """Resolve the base model TogetherEngine boots with.

    Order:
      1. PMC_BASE_MODEL env override (per-deploy config)
      2. FRONTIER tier default from the model registry → Kimi K2
      3. Hard fallback (Kimi K2 alias) if registry import fails

    The shipped default is the FRONTIER tier (Kimi K2). Without this
    the engine fell back to its module-level constant which is the
    8B Try tier — that's the cheap onboarding model, NOT what we
    want serving real users.
    """
    explicit = os.environ.get("PMC_BASE_MODEL", "").strip()
    if explicit:
        return explicit
    try:
        from pmc.schema.base_models import ModelTier, spec_for_tier
        # FRONTIER tier maps to the right Together model via the
        # alias table in pmc.train.together_trainer.resolve_together_model.
        spec = spec_for_tier(ModelTier.FRONTIER)
        from pmc.train.together_trainer import resolve_together_model
        return resolve_together_model(spec.hf_id)
    except Exception:
        return "moonshotai/Kimi-K2-Instruct-0905"


def _select_engine():
    """Pick the engine the server boots with. Logs which one and why
    so a future maintainer can grep the startup log."""
    forced = os.environ.get("PMC_INFERENCE", "").strip().lower()

    if forced == "together":
        from pmc.serve.engine_together import TogetherEngine
        base = _together_base_model()
        engine = TogetherEngine(base_model=base)
        print(f"[serve] engine=TogetherEngine base={base} (PMC_INFERENCE=together)", flush=True)
        return engine

    if forced == "mock":
        from pmc.serve.engine import MockEngine
        engine = MockEngine(base_model="mock/base")
        print("[serve] engine=MockEngine (PMC_INFERENCE=mock)", flush=True)
        return engine

    # Auto-detect. On Apple Silicon dev boxes, prefer MLX so devs get
    # fast local inference without API cost. On a Linux/x86 host
    # (Railway), MLX isn't even installable; if TOGETHER_API_KEY is
    # set, that's a strong signal we should use TogetherEngine.
    if os.environ.get("TOGETHER_API_KEY") and not _is_apple_silicon():
        from pmc.serve.engine_together import TogetherEngine
        base = _together_base_model()
        engine = TogetherEngine(base_model=base)
        print(f"[serve] engine=TogetherEngine base={base} (hosted Linux + TOGETHER_API_KEY)", flush=True)
        return engine

    if _is_apple_silicon():
        try:
            from pmc.serve.engine_mlx import MLXEngine, DEFAULT_MLX_BASE
            engine = MLXEngine(base_model=DEFAULT_MLX_BASE)
            print(f"[serve] engine=MLXEngine base={DEFAULT_MLX_BASE} (local dev)", flush=True)
            return engine
        except Exception as e:
            print(f"[serve] MLXEngine init failed: {e}", flush=True)

    # Last-resort fallback so the server still starts. /v1/runtime/
    # capabilities will tell the client inference is mocked.
    from pmc.serve.engine import MockEngine
    engine = MockEngine(base_model="mock/base")
    print("[serve] engine=MockEngine (fallback — no real engine available)", flush=True)
    return engine


def _select_memory_provider(storage_root: str):
    """Best-effort enable of the recall memory provider for inference
    context. Requires OPENAI_API_KEY for embeddings (or we'd silently
    no-op)."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[serve] memory_provider=disabled (set OPENAI_API_KEY to enable recall)", flush=True)
        return None
    try:
        from pmc.serve.memory_context import MemoryContextProvider
        from pmc.memory.embeddings import OpenAIEmbeddings
        from pmc.storage.paths import StoragePaths
        provider = MemoryContextProvider(
            paths=StoragePaths(os.path.join(storage_root, "storage")),
            embeddings=OpenAIEmbeddings(),
        )
        print("[serve] memory_provider=enabled (OpenAI embeddings)", flush=True)
        return provider
    except Exception as e:
        print(f"[serve] memory_provider=disabled ({e})", flush=True)
        return None


def _cors_origins() -> list[str]:
    raw = os.environ.get("PMC_CORS_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return list(DEFAULT_CORS_ORIGINS)


def main() -> int:
    storage_root = os.environ.get(
        "PMC_DEV_ROOT",
        str(Path.home() / ".pmc-dev"),
    )
    # PMC_PORT wins if set explicitly; otherwise honor Railway/Heroku's
    # convention of injecting $PORT. Default 8000 for local dev.
    port = int(os.environ.get("PMC_PORT") or os.environ.get("PORT") or "8000")

    from pmc.serve.api import run as run_server
    from pmc.serve.registry import AdapterRegistry
    from pmc.serve.server import PMCServer

    print(f"[serve] storage_root={storage_root}", flush=True)
    print(f"[serve] port={port}", flush=True)
    print(f"[serve] TOGETHER_API_KEY set: {bool(os.environ.get('TOGETHER_API_KEY'))}", flush=True)
    print(f"[serve] ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}", flush=True)

    engine = _select_engine()
    memory_provider = _select_memory_provider(storage_root)

    registry = AdapterRegistry(os.path.join(storage_root, "registry"))
    server = PMCServer(registry, engine, memory_provider=memory_provider)

    run_server(
        server,
        storage_root=os.path.join(storage_root, "storage"),
        cors_origins=_cors_origins(),
        port=port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
