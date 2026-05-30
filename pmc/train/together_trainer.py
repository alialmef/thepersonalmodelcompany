"""Together AI LoRA trainer — the production path.

Hands off to Together's fine-tuning API. Produces a real LoRA adapter
on top of a frontier base model (Kimi-K2-Instruct-0905 by default),
trained on H100s, downloaded back into the user's bundle dir so the
artifact stays portable + user-owned.

Local MLX is the dev path. This is the path that produces a model the
user will actually want to *use* — same data, vastly more capable base.

Cost model (May 2026 pricing):
  Kimi-K2 LoRA fine-tuning ≈ $3-5 / M training tokens
  Typical user (~5k curated examples × ~500 tokens × 1 epoch) ≈ 2.5M tok
  → ~$7.50-12.50 per personal training run

The Together API key must be present in the process environment as
TOGETHER_API_KEY (never on the user's Mac — set it on Railway).

Lifecycle:
  1. Format curated Completions into Together chat-JSONL (same shape MLX uses).
  2. Upload via Files API → file_id.
  3. POST /v1/fine-tunes with file_id + base + LoRA hyperparams → job_id.
  4. Poll job status; emit on_event() updates so the UI can stream them.
  5. On completion, download the adapter via the SDK's `download` and
     materialize as `<output_dir>/adapter_model.safetensors` plus an
     `adapter_config.json` describing what we trained.
  6. Return SFTRunResult.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from pmc.schema.conversation import Completion
from pmc.schema.training import TrainingConfig
from pmc.train.config import SFTRunResult
from pmc.train.formatter import completion_to_messages


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# What we ship as the default base. K2-Instruct-0905 is the
# best-documented Kimi LoRA target on Together as of May 2026.
DEFAULT_BASE = "moonshotai/Kimi-K2-Instruct-0905"

# Map PMC's internal model ids to the model names Together actually
# serves. Verified against /v1/models on 2026-05-27:
#   - moonshotai/Kimi-K2.6 is live, serverless (FP4), $1.20/$4.50 per M.
#   - The prior `Kimi-K2-Instruct-0905` returns 404 — Together rotated.
#   - Llama-3.1-8B-Reference is also gone; closest serverless host with
#     LoRA fine-tuning support is Llama-3.3-70B-Instruct-Turbo.
#   - Qwen3.6-27B was never on Together; closest peer is the 235B MoE.
TOGETHER_MODEL_ALIASES: dict[str, str] = {
    "moonshotai/Kimi-K2.6":             "moonshotai/Kimi-K2.6",
    "moonshotai/Kimi-K2":               "moonshotai/Kimi-K2.6",
    "kimi-k2.6":                         "moonshotai/Kimi-K2.6",
    "kimi-k2":                           "moonshotai/Kimi-K2.6",
    "kimi":                              "moonshotai/Kimi-K2.6",
    "frontier":                          "moonshotai/Kimi-K2.6",
    # Fallbacks — kept so test fixtures + the Personal/Try tier registry
    # entries don't 404. These are live + LoRA-fine-tunable on Together.
    "Qwen/Qwen3.6-27B":                 "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "qwen3.6-27b":                       "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "personal":                          "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "meta-llama/Llama-3.1-8B-Instruct":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "try":                               "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}


def resolve_together_model(name: str) -> str:
    """Return the Together-supported model id for a PMC-internal name."""
    if not name:
        return DEFAULT_BASE
    if name in TOGETHER_MODEL_ALIASES:
        return TOGETHER_MODEL_ALIASES[name]
    return name

# Hyperparams tuned for "voice fine-tuning" specifically — rank 16,
# alpha 32, single epoch, low LR. These preserve the base model's
# agentic / tool-use capability while shifting the voice.
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_N_EPOCHS = 1

# Terminal job statuses. Anything else means "still running".
TERMINAL_STATUSES = {"completed", "failed", "user_error", "cancelled", "error"}
ERROR_STATUSES    = {"failed", "user_error", "cancelled", "error"}


# ---------------------------------------------------------------------------
# Public entrypoint — matches the train_fn signature used by PMCPipeline
# (see `pmc.orchestrator.pipeline._default_train_fn`).
# ---------------------------------------------------------------------------


def together_train_fn(
    training_config: TrainingConfig,
    completions: list[Completion],
    output_dir: Path,
    holdout: list[Completion] | None = None,
    *,
    on_event: Callable[[str, dict[str, object]], None] | None = None,
) -> SFTRunResult:
    """Run a personal LoRA fine-tune on Together. Drop-in replacement
    for `mlx_train_fn` and the local PEFT fallback."""
    started_at = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TOGETHER_API_KEY is not set. The Together trainer needs an "
            "API key in the process environment. Set it on Railway for "
            "hosted runs, or in your local shell for dev."
        )

    requested_base = getattr(training_config, "base_model", "") or DEFAULT_BASE
    base_model = resolve_together_model(requested_base)
    if base_model != requested_base:
        _emit(on_event, "together_model_aliased", {
            "requested": requested_base, "resolved": base_model,
        })

    # 1. Format the data.
    #
    # Cost levers, applied in order:
    #   * max_examples — subsample to N best (we trust upstream curate
    #     to have already ranked-and-truncated; we just take the prefix)
    #   * max_tokens_per_example — truncate conversation history from the
    #     oldest turns until each example fits the budget. Final assistant
    #     turn is always preserved (that's the training signal).
    #
    # Why both: a 5k-example dataset at 7k tok/example = 35M tokens/epoch.
    # At $3-5/M that's $100-175 — too much for free tier. The legendary
    # recipe (1.5k examples × 2k tok × 3 epochs ≈ 9M tokens ≈ $30) is the
    # default; callers can override via TrainingConfig.
    max_tokens = (
        getattr(training_config, "max_tokens_per_example", None)
        or _env_int("PMC_TOGETHER_MAX_TOKENS_PER_EXAMPLE")
        or 0
    )
    max_examples = (
        getattr(training_config, "max_examples", None)
        or _env_int("PMC_TOGETHER_MAX_EXAMPLES")
        or 0
    )
    if max_examples and max_examples < len(completions):
        completions = list(completions)[:max_examples]

    train_jsonl = output_dir / "train.together.jsonl"
    eval_jsonl  = output_dir / "valid.together.jsonl" if holdout else None
    n_train, train_tokens = _write_messages_jsonl(completions, train_jsonl, max_tokens=max_tokens)
    n_eval, eval_tokens = (0, 0)
    if eval_jsonl and holdout:
        n_eval, eval_tokens = _write_messages_jsonl(holdout, eval_jsonl, max_tokens=max_tokens)

    _emit(on_event, "together_dataset_ready", {
        "train_examples": n_train,
        "eval_examples": n_eval,
        "approx_train_tokens": train_tokens,
        "approx_eval_tokens": eval_tokens,
        "approx_cost_usd_low":  round(train_tokens * 3 / 1_000_000, 2),
        "approx_cost_usd_high": round(train_tokens * 5 / 1_000_000, 2),
        "max_tokens_per_example": max_tokens or None,
        "train_file": str(train_jsonl),
    })

    # 2. Talk to Together.
    runner = _TogetherRun(
        api_key=api_key,
        base_model=base_model,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        n_epochs=getattr(training_config, "num_epochs", DEFAULT_N_EPOCHS),
        learning_rate=getattr(training_config, "learning_rate", DEFAULT_LEARNING_RATE),
        lora_r=getattr(training_config, "lora_r", DEFAULT_LORA_R),
        lora_alpha=getattr(training_config, "lora_alpha", DEFAULT_LORA_ALPHA),
        user_id=getattr(training_config, "user_id", ""),
        on_event=lambda kind, data: _emit(on_event, kind, data),
    )

    # Sample the base model now so the /training UI has the "before"
    # shot ready by the time the user lands on the screen. Cheap
    # (~$0.002) and gives the live-convergence view its first beat
    # without waiting for training to start.
    try:
        from pmc.train.checkpoint_sampler import sample_base
        baseline = sample_base(base_model, api_key=api_key)
        _emit(on_event, "checkpoint_sample", {
            "stage": "baseline",
            "response": baseline.response,
            "model": baseline.base_model,
        })
    except Exception as e:
        _emit(on_event, "checkpoint_sample_failed", {"stage": "baseline", "error": str(e)})

    job_id = runner.upload_and_launch()
    final = runner.wait_for_completion(job_id)

    # Sample the fine-tuned model — the "after" shot. The /training
    # screen now has both halves of the live-voice viz.
    output_model_name = final.get("output_name") or final.get("model") or final.get("fine_tuned_model")
    if output_model_name:
        try:
            from pmc.train.checkpoint_sampler import sample_final
            final_sample = sample_final(base_model, str(output_model_name), api_key=api_key)
            _emit(on_event, "checkpoint_sample", {
                "stage": "final",
                "response": final_sample.response,
                "model": final_sample.adapter_model,
            })
        except Exception as e:
            _emit(on_event, "checkpoint_sample_failed", {"stage": "final", "error": str(e)})

    # 3. Materialize the adapter into the bundle dir.
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    runner.download_adapter(job_id, adapter_dir)
    _write_adapter_config(adapter_dir, base_model=base_model,
                          lora_r=runner.lora_r, lora_alpha=runner.lora_alpha,
                          source="together")
    _record_remote_handle(adapter_dir, job_id=job_id, base_model=base_model,
                          output_model=output_model_name)

    completed_at = datetime.now()
    elapsed = (completed_at - started_at).total_seconds()

    return SFTRunResult(
        user_id=getattr(training_config, "user_id", ""),
        base_model=base_model,
        adapter_dir=adapter_dir,
        num_train_examples=n_train,
        num_eval_examples=n_eval,
        final_train_loss=_safe_float(final.get("train_loss")),
        final_eval_loss=_safe_float(final.get("eval_loss")),
        elapsed_seconds=elapsed,
        started_at=started_at,
        completed_at=completed_at,
        config=training_config,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _TogetherRun:
    api_key: str
    base_model: str
    train_jsonl: Path
    eval_jsonl: Optional[Path]
    n_epochs: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    user_id: str
    on_event: Callable[[str, dict[str, object]], None]

    def _client(self):
        try:
            from together import Together
        except ImportError as e:
            raise RuntimeError(
                "together package is required. Install with "
                "`uv pip install together`."
            ) from e
        return Together(api_key=self.api_key)

    # -- step 1: upload + launch ---------------------------------------------

    def upload_and_launch(self) -> str:
        client = self._client()

        self.on_event("together_upload_start", {"path": str(self.train_jsonl)})
        train_upload = client.files.upload(
            file=str(self.train_jsonl),
            purpose="fine-tune",
        )
        train_file_id = _get(train_upload, "id")
        self.on_event("together_upload_done", {"file_id": train_file_id})

        eval_file_id = None
        if self.eval_jsonl and self.eval_jsonl.exists():
            eval_upload = client.files.upload(file=str(self.eval_jsonl), purpose="fine-tune")
            eval_file_id = _get(eval_upload, "id")

        # Suffix tags the resulting custom model so we can find it later.
        suffix = f"pmc-{self.user_id[:8] or 'user'}-{int(time.time())}"

        create_kwargs = dict(
            training_file=train_file_id,
            model=self.base_model,
            n_epochs=int(self.n_epochs),
            learning_rate=float(self.learning_rate),
            lora=True,
            lora_r=int(self.lora_r),
            lora_alpha=int(self.lora_alpha),
            suffix=suffix,
        )
        if eval_file_id:
            create_kwargs["validation_file"] = eval_file_id

        self.on_event("together_create_job", {
            "base_model": self.base_model,
            "n_epochs": self.n_epochs,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "suffix": suffix,
        })
        job = client.fine_tuning.create(**create_kwargs)
        job_id = _get(job, "id")
        self.on_event("together_job_started", {"job_id": job_id, "status": _get(job, "status")})
        return job_id

    # -- step 2: poll until terminal ----------------------------------------

    def wait_for_completion(self, job_id: str, poll_seconds: int = 15) -> dict:
        client = self._client()
        last_status: Optional[str] = None
        last_emit = 0.0
        while True:
            job = client.fine_tuning.retrieve(job_id)
            status = (_get(job, "status") or "").lower()
            if status != last_status:
                self.on_event("together_status", {"job_id": job_id, "status": status})
                last_status = status
            # Heartbeat every minute even if status didn't change
            now = time.time()
            if now - last_emit > 60:
                self.on_event("together_heartbeat", {
                    "job_id": job_id,
                    "status": status,
                    "trained_tokens": _get(job, "trained_tokens"),
                })
                last_emit = now

            if status in TERMINAL_STATUSES:
                payload = _as_dict(job)
                if status in ERROR_STATUSES:
                    raise RuntimeError(
                        f"Together fine-tune failed with status={status}: "
                        f"{payload.get('events') or payload}"
                    )
                return payload

            time.sleep(poll_seconds)

    # -- step 3: pull weights -----------------------------------------------

    def download_adapter(self, job_id: str, adapter_dir: Path) -> None:
        client = self._client()
        # Together's SDK supports a `download` method that hands back
        # an archive. SDK shape: returns a path or a streamable file.
        tmp = adapter_dir / "together_download"
        tmp.mkdir(parents=True, exist_ok=True)
        archive_path = tmp / "adapter.tar.gz"

        self.on_event("together_download_start", {"job_id": job_id})

        # The SDK signature varies a touch across versions; try the
        # most common shapes.
        try:
            result = client.fine_tuning.download(id=job_id, output=str(archive_path))
        except TypeError:
            # Older SDKs used `download(job_id, output_dir=...)`.
            result = client.fine_tuning.download(job_id=job_id, output=str(archive_path))

        # If `result` is a file_like, persist; if it's already on disk, skip.
        if not archive_path.is_file() and hasattr(result, "read"):
            with archive_path.open("wb") as f:
                while True:
                    chunk = result.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        if not archive_path.is_file():
            # SDK wrote to wherever `output` pointed. Result might be
            # the actual path it wrote.
            if isinstance(result, str) and Path(result).is_file():
                archive_path = Path(result)
            else:
                raise RuntimeError(
                    f"Together SDK didn't produce a downloadable archive: result={result!r}"
                )

        # Extract the archive into adapter_dir.
        self.on_event("together_extract", {"archive": str(archive_path)})
        _extract_archive(archive_path, adapter_dir)

        # Clean up the temp archive.
        try:
            archive_path.unlink()
            tmp.rmdir()
        except OSError:
            pass

        # Sanity check that we got something LoRA-shaped.
        if not any(adapter_dir.glob("*.safetensors")):
            warning = "Together download didn't include a safetensors file"
            self.on_event("together_download_warning", {"warning": warning})
            logger.warning(warning)

        self.on_event("together_download_done", {"adapter_dir": str(adapter_dir)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_messages_jsonl(
    completions: Iterable[Completion],
    path: Path,
    *,
    max_tokens: int = 0,
) -> tuple[int, int]:
    """Write Completions as Together chat-JSONL. Returns (count, approx_tokens).

    Together's conversational fine-tuning format is exactly:
        {"messages": [{"role": "user", "content": "..."}, ...]}
    which is what `completion_to_messages` already produces. So we
    share the formatter with MLX — same dataset shape, both trainers.

    If `max_tokens > 0`, conversation history is truncated from the
    front (oldest turns first) until the example fits the budget. The
    final assistant turn is always preserved — that's the training
    signal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total_chars = 0
    with path.open("w", encoding="utf-8") as f:
        for c in completions:
            messages = completion_to_messages(c)
            if not messages:
                continue
            if not any(m.get("role") == "assistant" for m in messages):
                continue
            if max_tokens > 0:
                messages = _truncate_messages(messages, max_tokens)
                if not messages or not any(m.get("role") == "assistant" for m in messages):
                    continue
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n += 1
            total_chars += sum(len(m.get("content", "")) for m in messages)
    # 1 token ≈ 4 chars rough English approximation
    return n, total_chars // 4


def _truncate_messages(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """Drop history from the oldest user/assistant turns until the
    example fits the per-example token budget."""
    budget_chars = max(200, max_tokens * 4)
    # Always keep the final assistant turn; trim from the front.
    while messages and sum(len(m.get("content", "")) for m in messages) > budget_chars:
        # Never drop the last message (it's the assistant target).
        if len(messages) <= 1:
            break
        messages = messages[1:]
    return messages


def _write_adapter_config(adapter_dir: Path, *, base_model: str,
                           lora_r: int, lora_alpha: int, source: str) -> None:
    """Write the minimal adapter_config.json that downstream code expects.

    Mirrors what PEFT writes so the same loader path works whether the
    adapter came from PEFT, MLX, or Together."""
    cfg = {
        "base_model_name_or_path": base_model,
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        # PMC-specific extras
        "pmc_trainer": source,
        "pmc_format_version": 1,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))


def _record_remote_handle(adapter_dir: Path, *, job_id: str, base_model: str,
                           output_model: Optional[str]) -> None:
    """Save Together's identifiers alongside the adapter so the serve
    layer can hit Together-hosted inference if we want to (vs. always
    re-loading the LoRA against a local base)."""
    handle = {
        "provider": "together",
        "job_id": job_id,
        "base_model": base_model,
        "output_model": output_model,
        "recorded_at": datetime.now().isoformat(),
    }
    (adapter_dir / "remote.json").write_text(json.dumps(handle, indent=2))


def _extract_archive(archive_path: Path, into: Path) -> None:
    """Extract whatever shape Together hands us — tar.gz, tar, zip — into `into`."""
    p = str(archive_path)
    try:
        if p.endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                zf.extractall(into)
            return
        with tarfile.open(p, "r:*") as tf:
            tf.extractall(into)
        return
    except tarfile.ReadError:
        pass
    # Try raw bytes (small adapters sometimes ship as a single safetensors).
    with archive_path.open("rb") as f:
        head = f.read(8)
        f.seek(0)
        if head.startswith(b"\x80\x04"):  # pickle — bail
            raise RuntimeError("Together returned a pickle archive; refusing to load")
        # Otherwise just drop it in as-is.
        (into / archive_path.name).write_bytes(f.read())


def _get(obj, key: str):
    """Tolerate both pydantic models and plain dicts coming back from the SDK."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _emit(on_event, kind: str, data: dict[str, object]) -> None:
    if on_event:
        try:
            on_event(kind, data)
        except Exception:
            logger.exception("on_event callback raised; continuing")
