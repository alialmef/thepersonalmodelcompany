"""Tests for the train layer — formatter, plans, bundle, checkpoint helpers.

These tests cover everything that does NOT require torch/transformers/peft/trl
to be installed. The actual training runners (`run_sft`, `run_dpo`, `run_reward_model`)
are integration-tested separately when a GPU + ML deps are available.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pmc.schema.annotations import PreferenceAnnotation, QualityAnnotation, SourceAnnotation
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
    SourceType,
)
from pmc.schema.training import AdapterConfig, TrainingConfig
from pmc.schema.user import DataManifest, StyleProfile
from pmc.train import (
    ArtifactBundle,
    AuditEvent,
    BundleMetadata,
    DPOConfig,
    RewardModelConfig,
    build_inference_system_prompt,
    completion_to_dpo_pair,
    completion_to_messages,
    completions_to_messages,
    estimate_adapter_size_mb,
    estimate_steps,
    estimate_trainable_params,
)
from pmc.train.checkpoint import adapter_info, is_valid_adapter
from pmc.train.dpo import plan_dpo
from pmc.train.reward_model import plan_reward_model
from pmc.train.sft import plan_sft


# ---------- Helpers ----------


def _make_sft_completion(prompt: str, response: str) -> Completion:
    return Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content=prompt)]),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content=response)])
        ],
    )


def _make_dpo_completion(prompt: str, chosen: str, rejected: str) -> Completion:
    return Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content=prompt)]),
        candidates=[
            CompletionCandidate(
                messages=[Message(role=Role.ASSISTANT, content=chosen)],
                annotations=[PreferenceAnnotation(chosen=True)],
            ),
            CompletionCandidate(
                messages=[Message(role=Role.ASSISTANT, content=rejected)],
                annotations=[PreferenceAnnotation(chosen=False)],
            ),
        ],
    )


# ---------- Formatter ----------


def test_completion_to_messages_basic():
    c = _make_sft_completion("Hello", "Hi there!")
    msgs = completion_to_messages(c)
    assert msgs == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


def test_completion_to_messages_multi_turn_context():
    c = Completion(
        conversation=Conversation(
            messages=[
                Message(role=Role.USER, content="Q1"),
                Message(role=Role.ASSISTANT, content="A1"),
                Message(role=Role.USER, content="Q2"),
            ]
        ),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="A2")])
        ],
    )
    msgs = completion_to_messages(c)
    assert len(msgs) == 4
    assert msgs[-1]["content"] == "A2"
    assert msgs[-1]["role"] == "assistant"


def test_completion_to_messages_no_candidates_returns_none():
    c = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="hi")]),
        candidates=[],
    )
    assert completion_to_messages(c) is None


def test_completion_to_messages_empty_candidate_returns_none():
    c = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="hi")]),
        candidates=[CompletionCandidate(messages=[])],
    )
    assert completion_to_messages(c) is None


def test_completion_to_messages_whitespace_only_candidate_returns_none():
    c = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="hi")]),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="   ")])
        ],
    )
    assert completion_to_messages(c) is None


def test_completion_to_messages_drops_empty_context_messages():
    c = Completion(
        conversation=Conversation(
            messages=[
                Message(role=Role.USER, content=""),
                Message(role=Role.USER, content="real question"),
            ]
        ),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="answer")])
        ],
    )
    msgs = completion_to_messages(c)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "real question"


def test_completion_to_messages_standalone_writing():
    """A standalone user note (no context) should still produce messages with a user prompt
    if curate has synthesized one."""
    c = Completion(
        conversation=Conversation(
            messages=[Message(role=Role.USER, content="Continue in your own voice.")]
        ),
        candidates=[
            CompletionCandidate(
                messages=[Message(role=Role.ASSISTANT, content="A standalone thought.")]
            )
        ],
    )
    msgs = completion_to_messages(c)
    assert msgs is not None
    assert len(msgs) == 2


def test_completions_to_messages_filters_unusable():
    good = _make_sft_completion("Q", "A response that works")
    bad = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="Q")]),
        candidates=[],
    )
    out = completions_to_messages([good, bad])
    assert len(out) == 1


def test_completion_to_dpo_pair_uses_preference_annotations():
    c = _make_dpo_completion("Q", "good answer", "bad answer")
    pair = completion_to_dpo_pair(c)
    assert pair is not None
    assert pair["chosen"][0]["content"] == "good answer"
    assert pair["rejected"][0]["content"] == "bad answer"
    assert pair["prompt"][0]["content"] == "Q"


def test_completion_to_dpo_pair_falls_back_to_first_two():
    c = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="Q")]),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="first")]),
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content="second")]),
        ],
    )
    pair = completion_to_dpo_pair(c)
    assert pair is not None
    assert pair["chosen"][0]["content"] == "first"
    assert pair["rejected"][0]["content"] == "second"


def test_completion_to_dpo_pair_needs_two_candidates():
    c = _make_sft_completion("Q", "A")
    assert completion_to_dpo_pair(c) is None


def test_build_inference_system_prompt():
    assert build_inference_system_prompt(None, None) is None
    prompt = build_inference_system_prompt("Alex", "warm and direct")
    assert prompt is not None
    assert "Alex" in prompt
    assert "warm and direct" in prompt


# ---------- Plan estimation ----------


def test_estimate_steps_with_gradient_accumulation():
    effective, total = estimate_steps(
        num_examples=1000,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=3,
    )
    assert effective == 16
    assert total == 63 * 3  # ceil(1000/16) * 3


def test_estimate_steps_handles_zero_examples():
    effective, total = estimate_steps(0, 4, 4, 3)
    assert total == 0
    assert effective == 16


def test_estimate_trainable_params_scales_with_rank():
    cfg = AdapterConfig(rank=16)
    small = estimate_trainable_params(cfg)
    cfg2 = AdapterConfig(rank=64)
    big = estimate_trainable_params(cfg2)
    assert big == 4 * small


def test_estimate_adapter_size_mb_reasonable():
    params = estimate_trainable_params(AdapterConfig(rank=32))
    mb = estimate_adapter_size_mb(params)
    assert 1 < mb < 100  # in the expected ballpark for 8B QLoRA rank-32


def test_plan_sft_returns_warnings_for_small_dataset():
    config = TrainingConfig(user_id="u1")
    completions = [_make_sft_completion(f"Q{i}", f"A{i}") for i in range(10)]
    plan = plan_sft(config, completions)
    assert plan.num_train_examples == 10
    assert plan.estimated_steps > 0
    assert any("overfit" in w for w in plan.warnings)


def test_plan_sft_skips_unusable_completions():
    config = TrainingConfig(user_id="u1")
    good = [_make_sft_completion(f"Q{i}", f"Answer number {i} here") for i in range(5)]
    bad = [
        Completion(
            conversation=Conversation(messages=[Message(role=Role.USER, content="Q")]),
            candidates=[],
        )
    ]
    plan = plan_sft(config, good + bad)
    assert plan.num_train_examples == 5


def test_plan_sft_includes_eval_count():
    config = TrainingConfig(user_id="u1")
    train = [_make_sft_completion(f"Q{i}", f"A{i}") for i in range(60)]
    held_out = [_make_sft_completion(f"H{i}", f"HA{i}") for i in range(10)]
    plan = plan_sft(config, train, held_out)
    assert plan.num_eval_examples == 10


def test_plan_dpo():
    sft = TrainingConfig(user_id="u1")
    dpo_cfg = DPOConfig()
    pairs = [_make_dpo_completion(f"Q{i}", f"good{i}", f"bad{i}") for i in range(50)]
    plan = plan_dpo(sft, dpo_cfg, pairs)
    assert plan.job_type == "dpo"
    assert plan.num_train_examples == 50
    assert any("pairs recommended" in w for w in plan.warnings)


def test_plan_reward_model():
    config = RewardModelConfig()
    pairs = [_make_dpo_completion(f"Q{i}", "g", "b") for i in range(20)]
    plan = plan_reward_model(config, pairs)
    assert plan.job_type == "reward"
    assert plan.num_train_examples == 20
    assert any("pairs for meaningful" in w for w in plan.warnings)


# ---------- Checkpoint helpers ----------


def test_adapter_info_reads_config(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "Qwen/Qwen3-8B",
        })
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fake adapter weights")
    info = adapter_info(tmp_path)
    assert info.rank == 32
    assert info.alpha == 64
    assert info.target_modules == ["q_proj", "v_proj"]
    assert info.base_model_name == "Qwen/Qwen3-8B"
    assert info.size_bytes > 0


def test_adapter_info_missing_config_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        adapter_info(tmp_path)


def test_is_valid_adapter(tmp_path: Path):
    assert is_valid_adapter(tmp_path) is False

    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert is_valid_adapter(tmp_path) is True


# ---------- ArtifactBundle ----------


def _fake_adapter(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "Qwen/Qwen3-8B",
        })
    )
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    return path


def test_artifact_bundle_write_creates_full_layout(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "src_adapter")
    bundle = ArtifactBundle(
        metadata=BundleMetadata(
            user_id="user-1",
            user_name="Alex",
            user_email="alex@example.com",
            base_model="Qwen/Qwen3-8B",
            job_type="sft",
        ),
        adapter_dir=adapter,
        style_profile=StyleProfile(
            formality=0.4,
            verbosity=0.6,
            tone_tags=["warm", "direct"],
            description="warm and direct",
        ),
        training_manifest=DataManifest(
            training_run_id=__import__("uuid").uuid4(),
            dataset_version="v1",
            num_examples=1500,
        ),
        eval_report={"style_match": 0.78, "factual": 0.85},
        audit_log=[
            AuditEvent(stage="train", event="sft_started", data={"steps": 300}),
            AuditEvent(stage="train", event="sft_completed", data={"loss": 1.23}),
        ],
    )
    out = bundle.write(tmp_path / "out")

    for fname in ["bundle.json", "style_profile.json", "training_manifest.json",
                  "eval_report.json", "audit_log.json", "README.md"]:
        assert (out / fname).is_file(), f"missing {fname}"

    assert (out / "adapter" / "adapter_config.json").is_file()
    assert (out / "adapter" / "adapter_model.safetensors").is_file()

    metadata = json.loads((out / "bundle.json").read_text())
    assert metadata["adapter_checksum"] is not None
    assert metadata["user_email"] == "alex@example.com"


def test_artifact_bundle_roundtrip(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter_src")
    style = StyleProfile(
        formality=0.5,
        verbosity=0.7,
        tone_tags=["humorous"],
        common_phrases=["honestly", "you know"],
        sentence_length_avg=12.3,
    )
    original = ArtifactBundle(
        metadata=BundleMetadata(
            user_id="user-2", base_model="Qwen/Qwen3-8B", job_type="sft"
        ),
        adapter_dir=adapter,
        style_profile=style,
        eval_report={"x": 1.0},
    )
    original.append_audit("eval", "ran_style_eval", {"score": 0.81})
    out = original.write(tmp_path / "bundle")

    loaded = ArtifactBundle.load(out)
    assert loaded.metadata.user_id == "user-2"
    assert loaded.style_profile is not None
    assert loaded.style_profile.tone_tags == ["humorous"]
    assert "honestly" in loaded.style_profile.common_phrases
    assert loaded.eval_report["x"] == 1.0
    assert len(loaded.audit_log) == 1
    assert loaded.audit_log[0].event == "ran_style_eval"
    assert loaded.audit_log[0].data["score"] == 0.81


def test_artifact_bundle_readme_includes_base_model(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    bundle = ArtifactBundle(
        metadata=BundleMetadata(
            user_id="u3", base_model="meta-llama/Llama-3.1-8B-Instruct", job_type="sft"
        ),
        adapter_dir=adapter,
    )
    out = bundle.write(tmp_path / "out")
    readme = (out / "README.md").read_text()
    assert "Llama-3.1-8B-Instruct" in readme
    assert "PeftModel.from_pretrained" in readme
    assert "merge_adapter_into_base" in readme


def test_artifact_bundle_to_zip(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    bundle = ArtifactBundle(
        metadata=BundleMetadata(user_id="u4", base_model="Qwen/Qwen3-8B", job_type="sft"),
        adapter_dir=adapter,
    )
    zip_path = bundle.to_zip(tmp_path / "out.zip")
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "bundle.json" in names
    assert "README.md" in names
    assert any("adapter/" in n for n in names)


def test_audit_log_appends_in_order():
    bundle = ArtifactBundle(
        metadata=BundleMetadata(user_id="u", base_model="x", job_type="sft"),
        adapter_dir=Path("/nonexistent"),
    )
    bundle.append_audit("ingest", "loaded", {"items": 100})
    bundle.append_audit("curate", "filtered", {"kept": 80})
    bundle.append_audit("train", "started")
    assert [e.event for e in bundle.audit_log] == ["loaded", "filtered", "started"]
    assert [e.stage for e in bundle.audit_log] == ["ingest", "curate", "train"]


def test_quality_annotations_dont_leak_into_messages():
    """Annotations on messages are metadata — they should not appear in the chat format."""
    c = Completion(
        conversation=Conversation(messages=[Message(role=Role.USER, content="Hello")]),
        candidates=[
            CompletionCandidate(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content="Response",
                        annotations=[
                            SourceAnnotation(source_type="email", source_id="abc"),
                            QualityAnnotation(overall=0.9),
                        ],
                    )
                ],
            )
        ],
    )
    msgs = completion_to_messages(c)
    assert msgs == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Response"},
    ]
    # No extra keys leaking through
    for m in msgs:
        assert set(m.keys()) == {"role", "content"}
