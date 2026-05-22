"""Tests for the Together fine-tuning trainer.

We don't hit the real API. We verify:
  * JSONL format matches Together's conversational schema
  * truncation drops oldest turns until under budget, never the
    final assistant turn
  * the adapter_config.json + remote.json land in the right place
  * pipeline routing flips to Together when TOGETHER_API_KEY is set
  * launch fails clearly when TOGETHER_API_KEY is missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
)
from pmc.schema.training import TrainingConfig
from pmc.train.together_trainer import (
    DEFAULT_BASE,
    _record_remote_handle,
    _truncate_messages,
    _write_adapter_config,
    _write_messages_jsonl,
    together_train_fn,
)


def _make_completion(user_msgs: list[str], assistant: str) -> Completion:
    msgs = [Message(role=Role.USER, content=m) for m in user_msgs]
    return Completion(
        conversation=Conversation(messages=msgs),
        candidates=[CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content=assistant)])],
    )


# ---------- formatter ----------


def test_jsonl_writes_messages_shape(tmp_path: Path):
    completions = [
        _make_completion(["hello"], "hi there"),
        _make_completion(["what time is it?"], "around noon"),
    ]
    path = tmp_path / "train.jsonl"
    n, tokens = _write_messages_jsonl(completions, path)
    assert n == 2
    assert tokens > 0
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert "messages" in row
        assert isinstance(row["messages"], list)
        assert any(m["role"] == "assistant" for m in row["messages"])


def test_jsonl_skips_examples_with_no_assistant(tmp_path: Path):
    # Construct a completion whose candidate has only user-role messages.
    c = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="orphan")]),
        candidates=[CompletionCandidate(messages=[Message(role=Role.USER, content="also orphan")])],
    )
    n, _ = _write_messages_jsonl([c], tmp_path / "out.jsonl")
    assert n == 0


# ---------- truncation ----------


def test_truncate_drops_oldest_first(tmp_path: Path):
    # 5 messages of 100 chars each + an assistant tail.
    msgs = [{"role": "user", "content": "X" * 100} for _ in range(5)]
    msgs.append({"role": "assistant", "content": "final response"})
    # Budget of ~50 tokens ≈ 200 chars — we should drop until under.
    truncated = _truncate_messages(msgs, max_tokens=50)
    assert truncated[-1]["role"] == "assistant"
    assert truncated[-1]["content"] == "final response"
    assert sum(len(m["content"]) for m in truncated) <= 50 * 4


def test_truncate_preserves_assistant_when_only_one_message():
    msgs = [{"role": "assistant", "content": "X" * 5000}]
    out = _truncate_messages(msgs, max_tokens=200)
    # Single-message case: don't strip past it; pipeline will skip if
    # still over-budget, but we don't lose the example outright.
    assert out and out[0]["role"] == "assistant"


def test_jsonl_truncation_reduces_tokens(tmp_path: Path):
    long_msgs = ["A" * 4000, "B" * 4000, "C" * 4000]  # ~3000 tokens combined
    c = _make_completion(long_msgs, "ok")
    raw_path = tmp_path / "raw.jsonl"
    cap_path = tmp_path / "cap.jsonl"
    n_raw, tokens_raw = _write_messages_jsonl([c], raw_path)
    n_cap, tokens_cap = _write_messages_jsonl([c], cap_path, max_tokens=500)
    assert n_raw == n_cap == 1
    assert tokens_cap < tokens_raw
    assert tokens_cap <= 500 + 100  # within budget, allow for the assistant


# ---------- adapter config ----------


def test_adapter_config_matches_peft_shape(tmp_path: Path):
    _write_adapter_config(tmp_path, base_model="moonshotai/Kimi-K2-Instruct-0905",
                          lora_r=16, lora_alpha=32, source="together")
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["base_model_name_or_path"] == "moonshotai/Kimi-K2-Instruct-0905"
    assert cfg["r"] == 16
    assert cfg["lora_alpha"] == 32
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["pmc_trainer"] == "together"


def test_remote_handle_records_fine_tuned_model(tmp_path: Path):
    _record_remote_handle(
        tmp_path,
        job_id="ftjob-1",
        base_model="moonshotai/Kimi-K2-Instruct-0905",
        output_model="ft:pmc-user",
    )
    remote = json.loads((tmp_path / "remote.json").read_text())
    assert remote["provider"] == "together"
    assert remote["job_id"] == "ftjob-1"
    assert remote["base_model"] == "moonshotai/Kimi-K2-Instruct-0905"
    assert remote["output_model"] == "ft:pmc-user"


# ---------- API key gating ----------


def test_together_train_fn_requires_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    cfg = TrainingConfig(user_id="u", base_model=DEFAULT_BASE)
    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
        together_train_fn(cfg, [_make_completion(["hi"], "hello")], tmp_path)


# ---------- pipeline routing ----------


def test_pipeline_picks_together_when_key_set(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "stub")
    monkeypatch.delenv("PMC_TRAINER", raising=False)
    # Patch the trainer fn we expect to be selected; the dispatch should
    # call it rather than mlx/sft.
    with patch("pmc.train.together_trainer.together_train_fn") as mock_train:
        mock_train.return_value = MagicMock()
        from pmc.orchestrator.pipeline import _default_train_fn
        cfg = TrainingConfig(user_id="u", base_model=DEFAULT_BASE)
        _default_train_fn(cfg, [], MagicMock(), None)
    assert mock_train.called


def test_pipeline_falls_back_to_local_when_no_together_key(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("PMC_TRAINER", "mlx")
    with patch("pmc.train.mlx_trainer.mlx_train_fn") as mock_mlx:
        mock_mlx.return_value = MagicMock()
        from pmc.orchestrator.pipeline import _default_train_fn
        cfg = TrainingConfig(user_id="u", base_model="mlx-something")
        _default_train_fn(cfg, [], MagicMock(), None)
    assert mock_mlx.called
