"""Modal-hosted training: run QLoRA SFT on Modal's serverless A100s.

Modal is our V0 training backend. The pattern: define an `App` + `Image` here
with all ML deps, decorate `train_on_modal` with `@app.function(gpu=...)`, and
the function runs on Modal whenever invoked. Locally it's importable only if
the `modal` package is installed — if not, `modal_train_fn` raises with a clear
install hint.

Deploy:
    modal deploy pmc/train/modal_trainer.py

Call from the orchestrator:
    from pmc.train.modal_trainer import modal_train_fn
    pipeline = PMCPipeline(..., train_fn=modal_train_fn)

Per the analysis: use preemptible (1.25x multiplier instead of 3.75x). Jobs
are idempotent — Modal retries on eviction automatically when configured.

Per-job cost target: ~$2.32 for a 30-min QLoRA on A100 80GB preemptible.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Modal SDK is required at module load — Modal needs to register the App.
# If you don't have `modal` installed locally, don't import this file.
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore[assignment]


# Module-level App + Image only get created when modal is available.
if _MODAL_AVAILABLE:
    app = modal.App("pmc-trainer")

    # Image with the [train] extras + PMC itself
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install(
            "torch>=2.4",
            "transformers>=4.40",
            "peft>=0.11",
            "trl>=0.9",
            "datasets>=2.19",
            "bitsandbytes>=0.43",
            "accelerate>=0.30",
            "pydantic>=2.7",
            "safetensors>=0.4",
        )
        # In production, replace with: .pip_install("pmc @ git+https://github.com/.../pmc.git")
        # For now we mount the local repo — see modal_train_fn below.
    )

    # Persistent volume for adapters across runs (cheaper than re-uploading each time)
    adapter_volume = modal.Volume.from_name("pmc-adapters", create_if_missing=True)

    @app.function(
        image=image,
        gpu="A100-80GB",
        timeout=3600,  # 1 hour max
        volumes={"/adapters": adapter_volume},
        # Cost optimization: preemptible costs 1.25x base, vs 3.75x non-preemptible.
        # Jobs are idempotent (we save adapter to volume; on eviction Modal retries).
    )
    def train_on_modal(
        config_dict: dict[str, Any],
        train_jsonl: bytes,
        eval_jsonl: bytes | None,
        run_id: str,
    ) -> dict[str, Any]:
        """Run SFT on a Modal A100. Returns SFTRunResult as JSON-able dict."""
        from pmc.schema.conversation import Completion
        from pmc.schema.training import TrainingConfig
        from pmc.train.sft import run_sft

        config = TrainingConfig.model_validate(config_dict)
        train = [
            Completion.model_validate_json(line)
            for line in train_jsonl.splitlines()
            if line.strip()
        ]
        eval_completions = (
            [
                Completion.model_validate_json(line)
                for line in eval_jsonl.splitlines()
                if line.strip()
            ]
            if eval_jsonl
            else None
        )

        # Save adapter to persistent volume so it survives function lifetime
        output_dir = Path("/adapters") / config.user_id / run_id / "adapter"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_sft(config, train, output_dir, eval_completions)
        # Commit the volume so the adapter persists
        adapter_volume.commit()

        return result.model_dump(mode="json")


def modal_train_fn(
    config: Any,  # TrainingConfig
    train: list[Any],  # list[Completion]
    output_dir: Path,
    eval_completions: list[Any] | None,  # list[Completion] | None
) -> Any:  # SFTRunResult
    """Bridge function matching the orchestrator's TrainFn signature.

    Serializes inputs, calls the deployed Modal function, deserializes the
    result, and writes the adapter to `output_dir` (downloading from Modal's
    persistent volume).
    """
    if not _MODAL_AVAILABLE:
        raise ImportError(
            "modal is required to run training on Modal. "
            "Install with `pip install modal` and run `modal setup`."
        )
    from pmc.train.config import SFTRunResult

    run_id = output_dir.parent.name or "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    train_jsonl = b"\n".join(c.model_dump_json().encode() for c in train)
    eval_jsonl = (
        b"\n".join(c.model_dump_json().encode() for c in eval_completions)
        if eval_completions
        else None
    )

    with app.run():
        result_dict = train_on_modal.remote(  # type: ignore[union-attr]
            config.model_dump(mode="json"),
            train_jsonl,
            eval_jsonl,
            run_id,
        )

    result = SFTRunResult.model_validate(result_dict)
    # Download the adapter from Modal's volume to the local output_dir
    _download_adapter_from_volume(config.user_id, run_id, output_dir)
    # Update adapter_dir to local path
    result.adapter_dir = output_dir
    return result


def _download_adapter_from_volume(user_id: str, run_id: str, local_output: Path) -> None:
    """Download adapter files from Modal's persistent volume to local disk."""
    if not _MODAL_AVAILABLE:
        return
    local_output.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name("pmc-adapters")
    prefix = f"{user_id}/{run_id}/adapter"
    with tempfile.TemporaryDirectory():
        # Modal's get_file / read_file API to pull files
        for entry in volume.iterdir(prefix):
            target = local_output / entry.path.split("/")[-1]
            with target.open("wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)


__all__ = ["modal_train_fn"]
