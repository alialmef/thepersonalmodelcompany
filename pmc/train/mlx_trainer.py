"""MLX-LM LoRA trainer — local fine-tuning on Apple Silicon.

This is the fast, free, runs-on-your-Mac training path. It uses Apple's
mlx-lm library which trains LoRA adapters on Apple Silicon GPUs via Metal.

Compared to Modal / Together fine-tuning:
- No API key, no credit, no network
- Llama 3.2 3B (4-bit MLX): ~15–30 min for ~1k examples on M4 Pro
- Llama 3.1 8B (4-bit MLX): ~45–90 min on M4 Pro 24GB
- Output: real safetensors adapter + adapter_config.json under `output_dir/`

We invoke mlx_lm.lora via subprocess for two reasons:
1. Stable API surface — the CLI is more stable than the internal Python API
2. Easy stdout/stderr streaming for the audit log + UI progress feed

The function signature matches `pmc.train.sft.run_sft` so the orchestrator
can swap it in transparently via the `train_fn` injection point.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from datetime import datetime

from pmc.schema.conversation import Completion
from pmc.schema.training import TrainingConfig
from pmc.train.config import SFTRunResult
from pmc.train.formatter import completion_to_messages


@dataclass(frozen=True)
class MLXTrainingResult:
    """Mirrors the relevant fields of pmc.train.sft.SFTRunResult."""

    adapter_dir: Path
    adapter_size_mb: float
    train_loss: float | None
    eval_loss: float | None
    train_examples: int
    elapsed_seconds: float


# Default base model — small enough to train fast on M-series, big enough to
# carry voice well. 4-bit MLX quant keeps memory bounded.
DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def write_mlx_dataset(
    completions: Iterable[Completion],
    out_dir: Path,
    *,
    eval_fraction: float = 0.1,
    min_eval: int = 1,
) -> tuple[int, int]:
    """Write train.jsonl + valid.jsonl in the format mlx_lm.lora expects.

    Each line: {"messages": [{"role": "user", "content": "..."},
                             {"role": "assistant", "content": "..."}, ...]}

    Returns (train_count, eval_count).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for c in completions:
        messages = completion_to_messages(c)
        if messages is None:
            continue
        # mlx_lm rejects examples without an assistant turn — formatter already
        # guarantees this, but assert defensively.
        if not any(m["role"] == "assistant" for m in messages):
            continue
        rows.append({"messages": messages})

    if not rows:
        raise ValueError("No usable training examples after formatting.")

    # Split — at least `min_eval` items go to validation if possible.
    n_eval = max(min_eval, int(round(len(rows) * eval_fraction)))
    n_eval = min(n_eval, max(1, len(rows) - 1))
    eval_rows = rows[-n_eval:]
    train_rows = rows[:-n_eval]
    if not train_rows:
        # Tiny dataset: train AND eval on everything. Better than a hard fail.
        train_rows = rows
        eval_rows = rows[-1:]

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", eval_rows)
    return len(train_rows), len(eval_rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_mlx_lora(
    training_config: TrainingConfig,
    completions: list[Completion],
    output_dir: Path,
    holdout: list[Completion] | None = None,
    *,
    model: str | None = None,
    iters: int | None = None,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    on_event: Callable[[str, dict[str, object]], None] | None = None,
) -> MLXTrainingResult:
    """Train a LoRA adapter via mlx_lm.lora. Streams metrics to `on_event`.

    `on_event(stage, data)` is invoked for every interpretable line from the
    mlx_lm stdout — used by the orchestrator to append audit events that the
    /train SSE picks up.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    n_train, n_eval = write_mlx_dataset(completions, data_dir)

    model_name = model or DEFAULT_MLX_MODEL

    # Heuristic iter count: ~1 epoch over the data, capped to a reasonable max
    # so a tiny dataset doesn't run forever and a big one doesn't OOM.
    if iters is None:
        iters = max(50, min(600, n_train * 2))

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_name,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(output_dir),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", "100",
        "--num-layers", "16",  # how many layers to LoRA-ify; -1 = all
    ]

    if on_event:
        on_event(
            "train",
            {
                "event": "mlx_train_starting",
                "model": model_name,
                "train_examples": n_train,
                "eval_examples": n_eval,
                "iters": iters,
                "batch_size": batch_size,
            },
        )

    start = time.time()
    final_train_loss: float | None = None
    final_val_loss: float | None = None

    # Run mlx_lm.lora and stream stdout line by line. mlx-lm logs human-
    # readable lines like "Iter 50: Train loss 1.234, Val loss 1.567" —
    # parse them into structured events for the UI.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            parsed = _parse_mlx_line(line)
            if parsed and on_event:
                if "train_loss" in parsed:
                    final_train_loss = float(parsed["train_loss"])
                if "val_loss" in parsed:
                    final_val_loss = float(parsed["val_loss"])
                on_event("train", {"event": "mlx_step", **parsed})
            elif on_event:
                on_event("train", {"event": "mlx_log", "line": line[:300]})
    finally:
        proc.stdout.close()
        proc.wait()

    if proc.returncode != 0:
        if on_event:
            on_event(
                "train",
                {"event": "mlx_train_failed", "returncode": proc.returncode},
            )
        raise RuntimeError(
            f"mlx_lm.lora exited with code {proc.returncode}"
        )

    elapsed = time.time() - start
    adapter_dir = output_dir
    adapter_size = _dir_size_mb(adapter_dir)

    if on_event:
        on_event(
            "train",
            {
                "event": "mlx_train_completed",
                "elapsed_seconds": round(elapsed, 1),
                "adapter_size_mb": round(adapter_size, 2),
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
            },
        )

    return MLXTrainingResult(
        adapter_dir=adapter_dir,
        adapter_size_mb=adapter_size,
        train_loss=final_train_loss,
        eval_loss=final_val_loss,
        train_examples=n_train,
        elapsed_seconds=elapsed,
    )


# Regex for "Iter 50: Train loss 1.234" / "Iter 100: Val loss 1.567" and
# combined forms like "Iter 50: Train loss 1.234, It/sec 2.5, Tokens/sec 312"
_MLX_ITER_RE = re.compile(
    r"Iter\s+(\d+):"                          # Iter N
    r".*?Train loss\s+([\d.]+)"               # Train loss X
    r"(?:.*?It/sec\s+([\d.]+))?"              # optional It/sec
    r"(?:.*?Tokens/sec\s+([\d.]+))?",         # optional Tokens/sec
)
_MLX_VAL_RE = re.compile(r"Iter\s+(\d+):.*?Val loss\s+([\d.]+)")
_MLX_SAVE_RE = re.compile(r"Saved (?:final )?adapter weights to (.+)")


def _parse_mlx_line(line: str) -> dict[str, object] | None:
    """Pull metrics out of an mlx_lm stdout line."""
    m = _MLX_VAL_RE.search(line)
    if m:
        return {"step": int(m.group(1)), "val_loss": float(m.group(2))}
    m = _MLX_ITER_RE.search(line)
    if m:
        out: dict[str, object] = {
            "step": int(m.group(1)),
            "train_loss": float(m.group(2)),
        }
        if m.group(3):
            out["it_per_sec"] = float(m.group(3))
        if m.group(4):
            out["tokens_per_sec"] = float(m.group(4))
        return out
    m = _MLX_SAVE_RE.search(line)
    if m:
        return {"saved_to": m.group(1).strip()}
    return None


def _dir_size_mb(directory: Path) -> float:
    total = 0
    for f in directory.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Orchestrator adapter — matches the train_fn signature expected by
# PMCPipeline (see pmc/orchestrator/pipeline.py::_default_train_fn).
# ---------------------------------------------------------------------------

def mlx_train_fn(
    training_config: TrainingConfig,
    completions: list[Completion],
    output_dir: Path,
    holdout: list[Completion] | None = None,
    *,
    on_event: Callable[[str, dict[str, object]], None] | None = None,
) -> SFTRunResult:
    """train_fn-compatible wrapper. Drop-in replacement for the default SFT
    trainer in the orchestrator pipeline. Returns the same SFTRunResult shape."""
    started = datetime.now()
    result = run_mlx_lora(
        training_config=training_config,
        completions=completions,
        output_dir=output_dir,
        holdout=holdout,
        on_event=on_event,
    )
    return SFTRunResult(
        user_id=getattr(training_config, "user_id", ""),
        base_model=getattr(training_config, "base_model", DEFAULT_MLX_MODEL),
        adapter_dir=result.adapter_dir,
        num_train_examples=result.train_examples,
        num_eval_examples=0,
        final_train_loss=result.train_loss,
        final_eval_loss=result.eval_loss,
        elapsed_seconds=result.elapsed_seconds,
        started_at=started,
        completed_at=datetime.now(),
        config=training_config,
    )
