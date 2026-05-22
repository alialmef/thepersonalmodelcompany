"""CLI entry to launch a training run.

Usage:
    python -m pmc.train together --user UID [--base MODEL] [--dataset VERSION]
    python -m pmc.train mlx      --user UID [--base MODEL] [--dataset VERSION]
    python -m pmc.train estimate --user UID

`estimate` is the dry run: shows you the dataset size and token count so
you can ballpark the Together cost before pressing the button.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pmc.schema.training import TrainingConfig
from pmc.storage import UserStore


DEFAULT_USER = "11c7ace3-f395-4353-8acb-d6f7a2ec6113"
DEFAULT_BASE_TOGETHER = "moonshotai/Kimi-K2-Instruct-0905"
DEFAULT_BASE_MLX = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def _storage_root() -> Path:
    return Path(os.environ.get("PMC_DEV_ROOT", str(Path.home() / ".pmc-dev"))) / "storage"


def _load_dataset(user_id: str, storage_root: Path, version: str | None):
    store = UserStore(storage_root)
    versions = store.list_dataset_versions(user_id)
    if not versions:
        raise SystemExit(f"No curated datasets for user {user_id} under {storage_root}")
    version = version or versions[-1]
    if version not in versions:
        raise SystemExit(f"Dataset version {version!r} not found. Available: {versions}")
    train = store.load_curated_dataset(user_id, version)
    try:
        holdout = store.load_holdout(user_id, version)
    except Exception:
        holdout = []
    return version, train, holdout


def _estimate_tokens(train) -> int:
    """Rough char-based token estimator: 1 token ≈ 4 chars of English."""
    total_chars = 0
    for c in train:
        for cand in c.candidates:
            for m in cand.messages:
                total_chars += len(m.content)
        for m in c.conversation.messages:
            total_chars += len(m.content)
    return total_chars // 4


def cmd_estimate(args) -> int:
    storage = _storage_root()
    version, train, holdout = _load_dataset(args.user, storage, args.dataset)
    if args.max_examples and args.max_examples < len(train):
        train = train[: args.max_examples]

    if args.max_tokens_per_example:
        # Re-estimate against the truncated shape we'd actually send.
        # 4 chars ≈ 1 token, so the per-example char cap is 4 * max_tokens.
        cap = args.max_tokens_per_example * 4
        clamped_chars = sum(min(cap,
            sum(len(m.content) for cand in c.candidates for m in cand.messages)
            + sum(len(m.content) for m in c.conversation.messages)
        ) for c in train)
        tok = clamped_chars // 4
        scope = f"capped @ {args.max_tokens_per_example} tok/ex"
    else:
        tok = _estimate_tokens(train)
        scope = "no truncation"

    print(f"Dataset version : {version}")
    print(f"Train examples  : {len(train)} ({scope})")
    print(f"Holdout examples: {len(holdout)}")
    print(f"~ Tokens (1 ep) : {tok:,}")
    print(f"Together cost   : ${tok * 3 / 1_000_000:.2f} — ${tok * 5 / 1_000_000:.2f} per epoch")
    print(f"Storage root    : {storage}")
    return 0


def cmd_together(args) -> int:
    storage = _storage_root()
    version, train, holdout = _load_dataset(args.user, storage, args.dataset)
    if args.max_examples and args.max_examples < len(train):
        train = train[: args.max_examples]
    if args.max_tokens_per_example:
        os.environ["PMC_TOGETHER_MAX_TOKENS_PER_EXAMPLE"] = str(args.max_tokens_per_example)

    base = args.base or DEFAULT_BASE_TOGETHER
    output_dir = storage / "users" / args.user / "trainings" / f"together-{version}"
    print("Launching Together fine-tune")
    print(f"  user      : {args.user}")
    print(f"  dataset   : {version} ({len(train)} train / {len(holdout)} eval)")
    print(f"  base      : {base}")
    print(f"  output    : {output_dir}")
    print()

    from pmc.train.together_trainer import together_train_fn

    cfg = TrainingConfig(
        user_id=args.user,
        base_model=base,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    def on_event(kind, data):
        print(f"  [{kind}] {json.dumps(data, default=str)[:300]}", flush=True)

    result = together_train_fn(
        cfg, train, output_dir,
        holdout if holdout else None,
        on_event=on_event,
    )
    print()
    print(f"Done. Adapter at: {result.adapter_dir}")
    print(f"  elapsed: {result.elapsed_seconds:.1f}s")
    print(f"  train loss: {result.final_train_loss}")
    print(f"  eval loss : {result.final_eval_loss}")
    return 0


def cmd_mlx(args) -> int:
    storage = _storage_root()
    version, train, holdout = _load_dataset(args.user, storage, args.dataset)

    base = args.base or DEFAULT_BASE_MLX
    output_dir = storage / "users" / args.user / "trainings" / f"mlx-{version}"
    print("Launching local MLX fine-tune (dev path)")
    print(f"  base      : {base}")
    print(f"  output    : {output_dir}")

    from pmc.train.mlx_trainer import mlx_train_fn
    cfg = TrainingConfig(
        user_id=args.user,
        base_model=base,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )

    def on_event(kind, data):
        print(f"  [{kind}] {json.dumps(data, default=str)[:200]}", flush=True)

    result = mlx_train_fn(cfg, train, output_dir, holdout if holdout else None, on_event=on_event)
    print(f"Done. Adapter at: {result.adapter_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m pmc.train")
    p.add_argument("--user", default=DEFAULT_USER)
    sub = p.add_subparsers(dest="cmd", required=True)

    est = sub.add_parser("estimate")
    est.add_argument("--dataset", default=None)
    est.add_argument("--max-tokens-per-example", type=int, default=0)
    est.add_argument("--max-examples", type=int, default=0)

    tog = sub.add_parser("together")
    tog.add_argument("--dataset", default=None)
    tog.add_argument("--base", default=None)
    tog.add_argument("--epochs", type=int, default=1)
    tog.add_argument("--lr", type=float, default=1e-4)
    tog.add_argument("--lora-r", type=int, default=16)
    tog.add_argument("--lora-alpha", type=int, default=32)
    tog.add_argument("--max-tokens-per-example", type=int, default=0,
                     help="Truncate conversation history so each example "
                          "fits N tokens. Default 0 = no truncation.")
    tog.add_argument("--max-examples", type=int, default=0,
                     help="Subsample to first N examples. Default 0 = all.")

    mlx = sub.add_parser("mlx")
    mlx.add_argument("--dataset", default=None)
    mlx.add_argument("--base", default=None)
    mlx.add_argument("--epochs", type=int, default=1)
    mlx.add_argument("--lr", type=float, default=1e-4)

    args = p.parse_args()
    handlers = {"estimate": cmd_estimate, "together": cmd_together, "mlx": cmd_mlx}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
