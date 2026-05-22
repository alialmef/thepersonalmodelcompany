"""Tests for the serve layer — schema, registry, mock engine, server, export, API.

All tests run without torch/vLLM. The FastAPI tests are skipped automatically
if fastapi or httpx aren't installed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pmc.serve import (
    AdapterRecord,
    AdapterRegistry,
    ChatCompletionRequest,
    ChatMessage,
    MockEngine,
    PMCServer,
    export_adapter_only,
    export_bundle,
)
from pmc.train.bundle import ArtifactBundle, BundleMetadata


# ---------- Helpers ----------


def _fake_adapter(path: Path, base_model: str = "Qwen/Qwen3-8B") -> Path:
    """Create a minimal adapter directory that passes is_valid_adapter()."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": base_model,
        })
    )
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 2048)
    return path


def _persist_bundle(adapter_path: Path, bundle_dir: Path, user_id: str = "user-1") -> Path:
    """Write a complete ArtifactBundle to disk for tests that need one."""
    bundle = ArtifactBundle(
        metadata=BundleMetadata(
            user_id=user_id, user_name="Alex", base_model="Qwen/Qwen3-8B", job_type="sft"
        ),
        adapter_dir=adapter_path,
    )
    return bundle.write(bundle_dir)


# ---------- Schema ----------


def test_chat_request_minimum():
    req = ChatCompletionRequest(
        model="user-1",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.max_tokens == 512
    assert req.temperature == 0.7
    assert req.stream is False


def test_chat_request_accepts_openai_fields():
    req = ChatCompletionRequest(
        model="user-1",
        messages=[ChatMessage(role="user", content="hi")],
        user="alternate-user-id",
        top_p=0.9,
        stop=["\n"],
        n=1,
        stream=False,
    )
    assert req.user == "alternate-user-id"
    assert req.stop == ["\n"]


# ---------- Registry ----------


def test_registry_register_and_lookup(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    registry = AdapterRegistry(tmp_path / "reg")
    record = registry.register("user-1", adapter, base_model="Qwen/Qwen3-8B")
    assert record.user_id == "user-1"
    assert record.rank == 32
    assert record.adapter_size_mb > 0
    assert registry.get("user-1") is not None
    assert "user-1" in registry
    assert len(registry) == 1


def test_registry_rejects_invalid_adapter(tmp_path: Path):
    (tmp_path / "fake").mkdir()
    registry = AdapterRegistry(tmp_path / "reg")
    with pytest.raises(ValueError):
        registry.register("user-1", tmp_path / "fake", base_model="x")


def test_registry_require_raises_on_missing(tmp_path: Path):
    registry = AdapterRegistry(tmp_path / "reg")
    with pytest.raises(KeyError):
        registry.require("nobody")


def test_registry_persists_across_instances(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    reg_root = tmp_path / "reg"
    r1 = AdapterRegistry(reg_root)
    r1.register("user-1", adapter, base_model="Qwen/Qwen3-8B")
    r1.mark_served("user-1")

    r2 = AdapterRegistry(reg_root)
    record = r2.get("user-1")
    assert record is not None
    assert record.user_id == "user-1"
    assert record.request_count == 1


def test_registry_register_bundle(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter_src")
    bundle_dir = _persist_bundle(adapter, tmp_path / "bundle", user_id="user-bundle")
    registry = AdapterRegistry(tmp_path / "reg")
    record = registry.register_bundle(bundle_dir)
    assert record.user_id == "user-bundle"
    assert record.bundle_dir is not None
    assert Path(record.bundle_dir).is_dir()


def test_registry_register_bundle_reads_together_remote_handle(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter_remote", base_model="moonshotai/Kimi-K2-Instruct-0905")
    (adapter / "remote.json").write_text(
        json.dumps({
            "provider": "together",
            "job_id": "ftjob-1",
            "base_model": "moonshotai/Kimi-K2-Instruct-0905",
            "output_model": "ft:pmc-alex",
        })
    )
    bundle_dir = _persist_bundle(adapter, tmp_path / "bundle-remote", user_id="alex")
    registry = AdapterRegistry(tmp_path / "reg-remote")
    record = registry.register_bundle(bundle_dir)

    assert record.metadata["provider"] == "together"
    assert record.metadata["together_job_id"] == "ftjob-1"
    assert record.metadata["together_output_model"] == "ft:pmc-alex"


def test_registry_unregister(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register("u", adapter, base_model="x")
    assert registry.unregister("u") is True
    assert registry.unregister("u") is False
    assert "u" not in registry


def test_registry_unregister_with_delete_files(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter_to_delete")
    bundle_dir = _persist_bundle(adapter, tmp_path / "bundle_to_delete")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register_bundle(bundle_dir)
    registry.unregister("user-1", delete_files=True)
    assert not (tmp_path / "bundle_to_delete").exists()
    # adapter dir was copied into the bundle, original may still exist


def test_registry_list_users_sorted(tmp_path: Path):
    a1 = _fake_adapter(tmp_path / "a1")
    a2 = _fake_adapter(tmp_path / "a2")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register("zebra", a1, base_model="x")
    registry.register("alpha", a2, base_model="x")
    assert registry.list_users() == ["alpha", "zebra"]


def test_registry_mark_served_tracks_count(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register("u", adapter, base_model="x")
    for _ in range(3):
        registry.mark_served("u")
    record = registry.require("u")
    assert record.request_count == 3
    assert record.last_served_at is not None


# ---------- MockEngine ----------


def test_mock_engine_matches_substring():
    engine = MockEngine(
        responses={"weather": "It's sunny."},
        default="I don't know",
    )
    record = AdapterRecord(user_id="u", adapter_dir="/x", base_model="mock/base")
    text, usage = engine.chat(
        record,
        messages=[{"role": "user", "content": "How is the weather today?"}],
    )
    assert text == "It's sunny."
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert "u" in engine.warm_users


def test_mock_engine_warm_evict():
    engine = MockEngine()
    record = AdapterRecord(user_id="u", adapter_dir="/x", base_model="mock/base")
    engine.warm(record)
    assert "u" in engine.warm_users
    assert engine.evict("u") is True
    assert engine.evict("u") is False


# ---------- PMCServer ----------


def _make_server(tmp_path: Path, *, user_id: str = "user-1") -> tuple[PMCServer, AdapterRegistry]:
    adapter = _fake_adapter(tmp_path / f"adapter-{user_id}")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register(user_id, adapter, base_model="mock/base")
    engine = MockEngine(base_model="mock/base", default="A mock model response.")
    return PMCServer(registry, engine), registry


def test_server_chat_basic(tmp_path: Path):
    server, _ = _make_server(tmp_path)
    response = server.chat(
        ChatCompletionRequest(
            model="user-1",
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "A mock model response."
    assert response.usage.total_tokens > 0
    assert response.model == "user-1"


def test_server_chat_rejects_unknown_user(tmp_path: Path):
    server, _ = _make_server(tmp_path)
    with pytest.raises(KeyError):
        server.chat(
            ChatCompletionRequest(
                model="unknown",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )


def test_server_chat_rejects_base_model_mismatch(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register("user-x", adapter, base_model="qwen/some-model")
    server = PMCServer(registry, MockEngine(base_model="different/model"))
    with pytest.raises(ValueError) as exc:
        server.chat(
            ChatCompletionRequest(
                model="user-x",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
    assert "base" in str(exc.value).lower()


def test_server_chat_user_field_overrides_model(tmp_path: Path):
    server, _ = _make_server(tmp_path, user_id="real-user")
    response = server.chat(
        ChatCompletionRequest(
            model="ignored",
            user="real-user",
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    assert response.model == "real-user"


def test_server_chat_marks_served(tmp_path: Path):
    server, registry = _make_server(tmp_path)
    server.chat(
        ChatCompletionRequest(
            model="user-1",
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    record = registry.require("user-1")
    assert record.request_count == 1


def test_server_list_models(tmp_path: Path):
    server, _ = _make_server(tmp_path, user_id="u1")
    server.registry.register("u2", _fake_adapter(tmp_path / "a2"), base_model="mock/base")
    listing = server.list_models()
    assert listing.object == "list"
    ids = [m.id for m in listing.data]
    assert ids == ["u1", "u2"]
    assert all(m.adapter_size_mb is not None for m in listing.data)


def test_server_get_model(tmp_path: Path):
    server, _ = _make_server(tmp_path)
    info = server.get_model("user-1")
    assert info.id == "user-1"
    assert info.base_model == "mock/base"


def test_server_delete_model_evicts_engine(tmp_path: Path):
    server, _ = _make_server(tmp_path)
    server.chat(
        ChatCompletionRequest(
            model="user-1",
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    assert "user-1" in server.engine.warm_users  # type: ignore[attr-defined]
    assert server.delete_model("user-1") is True
    assert "user-1" not in server.engine.warm_users  # type: ignore[attr-defined]
    assert "user-1" not in server.registry


def test_server_finish_reason_length_when_at_limit(tmp_path: Path):
    """If the response token count meets max_tokens, finish_reason should be 'length'."""
    adapter = _fake_adapter(tmp_path / "adapter")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register("u", adapter, base_model="mock/base")
    # MockEngine reports tokens as len(response) // 4, so we need a long response.
    long_response = "x" * 4000
    engine = MockEngine(base_model="mock/base", default=long_response)
    server = PMCServer(registry, engine)
    response = server.chat(
        ChatCompletionRequest(
            model="u",
            max_tokens=50,
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    assert response.choices[0].finish_reason == "length"


# ---------- Export ----------


def test_export_adapter_only_creates_zip(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    record = AdapterRecord(
        user_id="u",
        adapter_dir=str(adapter),
        base_model="Qwen/Qwen3-8B",
    )
    zip_path = export_adapter_only(record, tmp_path / "out.zip")
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any("adapter_config.json" in n for n in names)
    assert any("adapter_model.safetensors" in n for n in names)


def test_export_bundle_uses_persisted_bundle(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter_src")
    bundle_dir = _persist_bundle(adapter, tmp_path / "bundle", user_id="user-export")
    record = AdapterRecord(
        user_id="user-export",
        adapter_dir=str(adapter),
        base_model="Qwen/Qwen3-8B",
        bundle_dir=str(bundle_dir),
    )
    zip_path = export_bundle(record, tmp_path / "bundle.zip")
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "README.md" in names
    assert "bundle.json" in names


def test_export_bundle_falls_back_when_no_persisted_bundle(tmp_path: Path):
    """If no bundle_dir is set, export should synthesize a minimal one."""
    adapter = _fake_adapter(tmp_path / "adapter")
    record = AdapterRecord(
        user_id="u",
        adapter_dir=str(adapter),
        base_model="Qwen/Qwen3-8B",
    )
    zip_path = export_bundle(record, tmp_path / "bundle.zip")
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        assert "bundle.json" in zf.namelist()


def test_export_adapter_only_missing_dir(tmp_path: Path):
    record = AdapterRecord(
        user_id="u",
        adapter_dir=str(tmp_path / "nonexistent"),
        base_model="x",
    )
    with pytest.raises(FileNotFoundError):
        export_adapter_only(record, tmp_path / "out.zip")


# ---------- FastAPI integration ----------


fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")  # required by TestClient
from fastapi.testclient import TestClient  # noqa: E402

from pmc.serve.api import create_app  # noqa: E402


def _client(tmp_path: Path) -> tuple[TestClient, PMCServer]:
    server, _ = _make_server(tmp_path)
    app = create_app(server)
    return TestClient(app), server


def test_api_healthz(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["base_model"] == "mock/base"


def test_api_runtime_capabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PMC_TRAINER", "together")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    client, _ = _client(tmp_path)
    r = client.get("/v1/runtime/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["training"]["provider"] == "together"
    assert body["training"]["available"] is False
    assert body["training"]["together_key_present"] is False
    assert body["inference"]["provider"] == "mock"


def test_api_list_models(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "user-1"


def test_api_chat_completions(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "user-1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_api_chat_unknown_user_404(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "nobody",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 404


def test_api_chat_stream_returns_sse(tmp_path: Path):
    """Streaming requests should return SSE events in OpenAI's format."""
    client, _ = _client(tmp_path)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "user-1",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes()).decode("utf-8")

    # SSE events are `data: <json>\n\n` lines, ending with `data: [DONE]`
    events = [
        line[len("data: ") :]
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    # First chunk should announce role=assistant
    first = json.loads(events[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    # At least one chunk in the middle should carry content
    middle = [json.loads(e) for e in events[1:-2]]
    assert any(c["choices"][0]["delta"].get("content") for c in middle)
    # Last meaningful chunk has a finish_reason
    last = json.loads(events[-2])
    assert last["choices"][0]["finish_reason"] in {"stop", "length"}


def test_api_chat_stream_uses_engine_acceptance_hook(tmp_path: Path):
    """Hosted engines can accept registered remote models whose base differs."""

    class FlexibleMockEngine(MockEngine):
        def allows_base_model(
            self,
            base_model: str,
            record: AdapterRecord | None = None,
        ) -> bool:
            return bool(record and record.metadata.get("provider") == "together")

    adapter = _fake_adapter(tmp_path / "adapter-remote")
    registry = AdapterRegistry(tmp_path / "reg-remote-stream")
    record = registry.register(
        "remote-user",
        adapter,
        base_model="moonshotai/Kimi-K2-Instruct-0905",
    )
    record.metadata["provider"] = "together"
    server = PMCServer(
        registry,
        FlexibleMockEngine(base_model="mock/base", default="remote stream ok"),
    )
    client = TestClient(create_app(server))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "remote-user",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode("utf-8")
    events = [
        line[len("data: ") :]
        for line in body.split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    text = "".join(
        json.loads(event)["choices"][0]["delta"].get("content") or ""
        for event in events
    )
    assert text == "remote stream ok"


def test_api_chat_stream_unknown_user_404(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "nobody",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 404


def test_mock_engine_chat_stream_yields_chunks():
    """MockEngine.chat_stream should yield non-empty pieces that reconstruct the response."""
    from pmc.serve.engine import MockEngine
    engine = MockEngine(default="Hello there friend")
    record = AdapterRecord(user_id="u", adapter_dir="/x", base_model="mock/base")
    chunks = list(engine.chat_stream(record, [{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello there friend"
    assert len(chunks) > 1  # actually broke it up


def test_server_chat_stream_marks_served(tmp_path: Path):
    server, registry = _make_server(tmp_path)
    chunks = list(
        server.chat_stream(
            ChatCompletionRequest(
                model="user-1",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
    )
    # First chunk role, middle chunks content, last chunk finish_reason
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason in {"stop", "length"}
    assert registry.require("user-1").request_count == 1


def test_server_chat_stream_base_model_mismatch_raises(tmp_path: Path):
    adapter = _fake_adapter(tmp_path / "adapter")
    registry = AdapterRegistry(tmp_path / "reg")
    registry.register("user-x", adapter, base_model="qwen/something")
    server = PMCServer(registry, MockEngine(base_model="different/model"))
    with pytest.raises(ValueError):
        list(server.chat_stream(
            ChatCompletionRequest(
                model="user-x",
                messages=[ChatMessage(role="user", content="hi")],
            )
        ))


def test_api_export_returns_zip(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.get("/v1/models/user-1/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "user-1" in r.headers.get("content-disposition", "")


def test_api_export_adapter_only(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.get("/v1/models/user-1/export?adapter_only=true")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"


def test_api_delete_model(tmp_path: Path):
    client, server = _client(tmp_path)
    r = client.delete("/v1/models/user-1")
    assert r.status_code == 200
    assert r.json()["deleted"] == "user-1"
    assert "user-1" not in server.registry


def test_api_delete_unknown_404(tmp_path: Path):
    client, _ = _client(tmp_path)
    r = client.delete("/v1/models/nobody")
    assert r.status_code == 404


# ---------- web-app endpoints (storage_root mounted) ----------


def _storage_client(tmp_path: Path) -> tuple[TestClient, PMCServer, Path]:
    """Build a TestClient with storage_root enabled (web-app endpoints mounted)."""
    server, _ = _make_server(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    app = create_app(server, storage_root=storage)
    return TestClient(app), server, storage


def test_api_healthz_reports_storage_enabled(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.get("/healthz")
    body = r.json()
    assert body["storage_enabled"] is True


def test_api_healthz_without_storage(tmp_path: Path):
    client, _ = _client(tmp_path)  # no storage_root
    body = client.get("/healthz").json()
    assert body["storage_enabled"] is False


def test_api_upload_text_source(tmp_path: Path):
    client, _, storage = _storage_client(tmp_path)
    content = b"A personal note about Tuesday's plan. Will reschedule the demo to Wednesday."
    r = client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", content, "text/markdown")},
        data={"kind": "text"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["raw_items_ingested"] == 1
    assert body["kind"] == "text"
    assert body["total_raw_items"] == 1
    # Persisted on disk under storage/
    assert (storage / "users" / "alex" / "raw").is_dir()


def test_api_upload_mbox_requires_user_emails(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("inbox.mbox", b"", "application/mbox")},
        data={"kind": "email_mbox"},  # missing user_emails
    )
    assert r.status_code == 500 or r.status_code == 400


def test_api_upload_rejects_unknown_kind(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("x.bin", b"", "application/octet-stream")},
        data={"kind": "gibberish"},
    )
    assert r.status_code == 400


def test_api_upload_rejects_raw_kind(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("x.txt", b"hi", "text/plain")},
        data={"kind": "raw"},
    )
    assert r.status_code == 400


def test_api_user_status(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    # Upload one source first
    client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A real personal thought.", "text/markdown")},
        data={"kind": "text"},
    )
    r = client.get("/v1/users/alex/status")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "alex"
    assert body["raw_item_count"] == 1
    assert any("text" in s for s in body["raw_sources"])


def test_api_user_status_empty_user_returns_skeleton(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.get("/v1/users/ghost/status")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "ghost"
    assert body["raw_item_count"] == 0
    assert body["raw_sources"] == []


def test_api_delete_source(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    upload = client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A note.", "text/markdown")},
        data={"kind": "text"},
    ).json()
    source_id = upload["source_id"]

    r = client.delete(f"/v1/users/alex/sources/{source_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted_source"] == source_id
    assert body["items_removed"] == 1

    # Status confirms source is gone
    status = client.get("/v1/users/alex/status").json()
    assert status["raw_item_count"] == 0


def test_api_delete_unknown_source_404(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.delete("/v1/users/alex/sources/nonexistent")
    assert r.status_code == 404


# ---------- Batch items upload (native desktop ingestion path) ----------


def test_api_items_upload_ingests_raw_items(tmp_path: Path):
    """The desktop app pushes pre-parsed RawItems via JSON. Endpoint persists them."""
    client, _, _ = _storage_client(tmp_path)
    body = {
        "kind": "imessage",
        "source_id": "imessage-2026-05-18",
        "items": [
            {
                "source_type": "imessage",
                "source_id": "imessage:1",
                "content": "Hey, you free Thursday?",
                "timestamp": "2026-05-18T10:30:00+00:00",
                "thread_id": "chat-abc",
                "author_identifier": "+15551234567",
                "is_user": False,
                "metadata": {"chat_name": "Family"},
            },
            {
                "source_type": "imessage",
                "source_id": "imessage:2",
                "content": "Sure, around 6?",
                "timestamp": "2026-05-18T10:31:00+00:00",
                "thread_id": "chat-abc",
                "author_identifier": "+15551234567",
                "is_user": True,
                "metadata": {},
            },
        ],
    }
    r = client.post("/v1/users/alex/sources/items", json=body)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["ingested"] == 2
    assert result["source_id"] == "imessage-2026-05-18"
    assert result["kind"] == "imessage"
    assert result["total_raw_items"] == 2

    status = client.get("/v1/users/alex/status").json()
    assert status["raw_item_count"] == 2
    assert "imessage-2026-05-18" in status["raw_sources"]


def test_api_items_upload_rejects_missing_source_id(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post("/v1/users/alex/sources/items", json={
        "kind": "imessage",
        "items": [{"source_type": "imessage", "source_id": "x", "content": "hi"}],
    })
    assert r.status_code == 400
    assert "source_id" in r.json()["detail"]


def test_api_items_upload_rejects_empty_items(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post("/v1/users/alex/sources/items", json={
        "kind": "imessage",
        "source_id": "x",
        "items": [],
    })
    assert r.status_code == 400


def test_api_items_upload_rejects_malformed_item(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post("/v1/users/alex/sources/items", json={
        "kind": "imessage",
        "source_id": "x",
        "items": [{"missing": "required_fields"}],
    })
    assert r.status_code == 400
    assert "index 0" in r.json()["detail"]


def test_api_items_upload_writes_audit_event(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    client.post("/v1/users/alex/sources/items", json={
        "kind": "imessage",
        "source_id": "imessage-test",
        "items": [{
            "source_type": "imessage",
            "source_id": "imessage:1",
            "content": "test",
            "is_user": True,
        }],
    })
    status = client.get("/v1/users/alex/status").json()
    events = status["recent_events"]
    assert any(
        e["stage"] == "ingest" and e["event"] == "items_pushed_via_native"
        for e in events
    )


def test_api_storage_endpoints_not_mounted_without_storage_root(tmp_path: Path):
    client, _ = _client(tmp_path)  # no storage_root
    r = client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("x.txt", b"hi", "text/plain")},
        data={"kind": "text"},
    )
    assert r.status_code == 404


def test_api_cors_headers(tmp_path: Path):
    server, _ = _make_server(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    app = create_app(
        server,
        storage_root=storage,
        cors_origins=["http://localhost:3000"],
    )
    client = TestClient(app)
    r = client.options(
        "/v1/users/alex/sources/upload",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ---------- Verification endpoints ----------


def test_api_eval_prompts_build_private_probes(tmp_path: Path):
    client, _, storage = _storage_client(tmp_path)
    r = client.get("/v1/users/alex/eval/prompts")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["prompts"]) == 5
    assert body["prompts"][0]["id"].startswith("probe-")
    assert body["prompts"][0]["kind"] == "voice"
    assert "candidates" in body["prompts"][0]
    assert (storage / "users" / "alex" / "verification" / "probes.jsonl").is_file()


def test_api_eval_judgment_records_training_signal(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    prompts = client.get("/v1/users/alex/eval/prompts").json()["prompts"]
    first = prompts[0]
    candidate_id = first["candidates"][0]["id"]

    r = client.post(
        "/v1/users/alex/eval/judgments",
        json={
            "probeId": first["id"],
            "verdict": "edit",
            "chosenCandidateId": candidate_id,
            "editedText": "yeah thursday works",
            "dimension": "voice",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["judgment"]["verdict"] == "edit"
    assert body["trust_report"]["total_judgments"] == 1

    signal = client.get("/v1/users/alex/verification/training-signal").json()
    assert signal["preference_completions"] == 1


def test_api_action_trace_updates_trust_report(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.post(
        "/v1/users/alex/verification/action-traces",
        json={
            "surface": "mail",
            "operation": "draft_reply",
            "proposed_text": "Sure, Thursday works.",
            "decision": "approved",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trace"]["decision"] == "approved"
    assert body["trust_report"]["total_action_traces"] == 1

    signal = client.get("/v1/users/alex/verification/training-signal").json()
    assert signal["action_sft_completions"] == 1


def test_api_action_proposal_review_creates_trace(tmp_path: Path):
    client, _, storage = _storage_client(tmp_path)
    create = client.post(
        "/v1/users/alex/actions/proposals",
        json={
            "surface": "messages",
            "operation": "draft_reply",
            "prompt": "Reply to Maya",
            "proposed_text": "Sure, Thursday works.",
            "proposed_payload": {"recipient": "Maya"},
        },
    )
    assert create.status_code == 200, create.text
    proposal = create.json()["proposal"]
    assert proposal["id"].startswith("prop-")
    assert proposal["risk_level"] == "low"

    review = client.post(
        f"/v1/users/alex/actions/proposals/{proposal['id']}/review",
        json={
            "decision": "edited",
            "edited_text": "yeah thursday works",
            "final_payload": {"recipient": "Maya"},
        },
    )
    assert review.status_code == 200, review.text
    body = review.json()
    assert body["proposal"]["status"] == "edited"
    assert body["trace"]["proposal_id"] == proposal["id"]
    assert body["trace"]["edited_text"] == "yeah thursday works"
    assert (storage / "users" / "alex" / "verification" / "action_proposals.jsonl").is_file()

    signal = client.get("/v1/users/alex/verification/training-signal").json()
    assert signal["action_sft_completions"] == 1


def test_api_action_proposal_risk_ladder(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    create = client.post(
        "/v1/users/alex/actions/proposals",
        json={
            "surface": "mail",
            "operation": "send_email",
            "proposed_text": "Sending this would be high risk.",
            "proposed_payload": {"to": "x@example.com"},
        },
    )
    assert create.status_code == 200, create.text
    body = create.json()
    assert body["proposal"]["risk_level"] == "high"
    assert body["review"]["requires_confirmation"] is True
    assert body["review"]["execution_allowed"] is False


def test_api_action_runtime_executes_and_undos_local_file(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    target = tmp_path / "laptop" / "note.md"
    create = client.post(
        "/v1/users/alex/actions/proposals",
        json={
            "surface": "files",
            "operation": "write_text",
            "risk_level": "medium",
            "proposed_text": "write a note",
            "proposed_payload": {
                "path": str(target),
                "content": "local world note",
            },
        },
    )
    assert create.status_code == 200, create.text
    proposal = create.json()["proposal"]

    simulated = client.post(f"/v1/users/alex/actions/proposals/{proposal['id']}/simulate")
    assert simulated.status_code == 200, simulated.text
    assert simulated.json()["receipt"]["ok"] is True
    assert target.exists() is False

    blocked = client.post(f"/v1/users/alex/actions/proposals/{proposal['id']}/execute")
    assert blocked.status_code == 400

    review = client.post(
        f"/v1/users/alex/actions/proposals/{proposal['id']}/review",
        json={"decision": "approved"},
    )
    assert review.status_code == 200, review.text

    executed = client.post(f"/v1/users/alex/actions/proposals/{proposal['id']}/execute")
    assert executed.status_code == 200, executed.text
    body = executed.json()
    assert body["proposal"]["status"] == "executed"
    assert target.read_text(encoding="utf-8") == "local world note"

    undone = client.post(
        f"/v1/users/alex/actions/proposals/{proposal['id']}/undo",
        json={"undo_token": body["receipt"]["undo_token"]},
    )
    assert undone.status_code == 200, undone.text
    assert target.exists() is False


def test_api_world_scan_indexes_requested_root(tmp_path: Path):
    client, _, storage = _storage_client(tmp_path)
    root = tmp_path / "laptop"
    root.mkdir()
    (root / "project.md").write_text("personal project context", encoding="utf-8")

    scan = client.post(
        "/v1/users/alex/world/scan",
        json={
            "roots": [str(root)],
            "full_disk": False,
            "max_files": 25,
        },
    )
    assert scan.status_code == 200, scan.text
    assert scan.json()["scan"]["files_indexed"] == 1

    files = client.get("/v1/users/alex/world/files", params={"query": "project"})
    assert files.status_code == 200, files.text
    body = files.json()
    assert body["files"][0]["name"] == "project.md"
    assert (storage / "users" / "alex" / "world" / "files.jsonl").is_file()


def test_api_promote_requires_private_verification(tmp_path: Path):
    client, _, storage = _storage_client(tmp_path)
    adapter = _fake_adapter(tmp_path / "adapter-promote")
    _persist_bundle(
        adapter,
        storage / "users" / "alex" / "bundles" / "run-verified",
        user_id="alex",
    )

    blocked = client.post("/v1/users/alex/runs/run-verified/promote")
    assert blocked.status_code == 409

    prompts = client.get("/v1/users/alex/eval/prompts").json()["prompts"]
    for prompt in prompts[:3]:
        r = client.post(
            "/v1/users/alex/eval/judgments",
            json={
                "probeId": prompt["id"],
                "verdict": "approve",
                "chosenCandidateId": prompt["candidates"][0]["id"],
                "dimension": "voice",
            },
        )
        assert r.status_code == 200, r.text

    promoted = client.post("/v1/users/alex/runs/run-verified/promote")
    assert promoted.status_code == 200, promoted.text
    body = promoted.json()
    assert body["active"]["run_id"] == "run-verified"
    assert body["trust_report"]["readiness"] == "voice"


# ---------- Pipeline runs (Act 3 backend: job submission + SSE) ----------


def test_api_submit_run_returns_job_id(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A personal reflection on the project.", "text/markdown")},
        data={"kind": "text"},
    )
    r = client.post(
        "/v1/users/alex/runs",
        json={"dry_run": True, "skip_eval": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == "alex"
    assert body["job_id"].startswith("job-")
    assert body["status"] in {"queued", "running", "completed"}


def test_api_get_run_status(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A reflection.", "text/markdown")},
        data={"kind": "text"},
    )
    job = client.post(
        "/v1/users/alex/runs",
        json={"dry_run": True, "skip_eval": True},
    ).json()
    import time
    body = None
    for _ in range(50):
        r = client.get(f"/v1/users/alex/runs/{job['job_id']}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert body is not None
    assert body["user_id"] == "alex"
    assert body["status"] in {"completed", "failed"}


def test_api_get_unknown_run_404(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.get("/v1/users/alex/runs/job-does-not-exist")
    assert r.status_code == 404


def test_api_run_events_stream_emits_audit_events(tmp_path: Path):
    """SSE stream yields the audit events captured during the pipeline run."""
    client, _, _ = _storage_client(tmp_path)
    # Upload enough varied content that curate keeps the minimum examples
    for i, body_bytes in enumerate([
        b"Reflection on shipping: cost of waiting is felt every week.",
        b"Notes on the team conversation about ownership versus outcomes.",
        b"Decided to push the trip back a month. Depth over breadth this season.",
        b"Realized I keep deferring the dental appointment. Booked it for Tuesday.",
        b"Saturday lesson: blocking by category beats blocking by project for my style.",
        b"Long walk through the park. The structural fix is more important than cost.",
        b"Coffee chat with Rana about onboarding flow. Bleeding on the third screen.",
        b"Spent the morning rereading pricing notes. Tiered makes more sense than per-seat.",
        b"Hiring update: passed on the second-round candidate. Cultural mismatch.",
        b"Travel plan for May is set. Two weeks because three burns me out.",
        b"Notes on the demo: cut the third slide. Open with one specific thing.",
        b"Workout block worked. Eight weeks was enough to feel different.",
        b"Thinking about partnership pitch: lead with the integration story.",
        b"Reread the old journal entries. Anxieties real but predictions wrong.",
        b"Reading habits this month: short essays beat full books right now.",
    ]):
        client.post(
            "/v1/users/alex/sources/upload",
            files={"file": (f"note-{i}.md", body_bytes, "text/markdown")},
            data={"kind": "text"},
        )

    job = client.post(
        "/v1/users/alex/runs",
        json={"dry_run": True, "skip_eval": True, "skip_deploy": True},
    ).json()

    with client.stream("GET", f"/v1/users/alex/runs/{job['job_id']}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes()).decode("utf-8")

    events = [
        line[len("data: "):]
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"

    parsed = []
    for e in events[:-1]:
        try:
            parsed.append(json.loads(e))
        except json.JSONDecodeError:
            pass

    stages_seen = {p.get("stage") for p in parsed if isinstance(p, dict) and "stage" in p}
    assert "ingest" in stages_seen or "curate" in stages_seen or "train" in stages_seen
    assert any(p.get("event") == "job_finished" for p in parsed)


def test_api_run_events_unknown_job_404(tmp_path: Path):
    client, _, _ = _storage_client(tmp_path)
    r = client.get("/v1/users/alex/runs/nope/events")
    assert r.status_code == 404


# ---------- Pricing (founder-aware) ----------


def test_api_pricing_returns_all_tiers_with_founder_freebie(tmp_path: Path):
    """Initial state: 100 founder slots → Try tier shows $0, others list price."""
    client, _, _ = _storage_client(tmp_path)
    r = client.get("/v1/pricing")
    assert r.status_code == 200
    body = r.json()
    assert "tiers" in body
    assert body["founder"]["try_tier_free"] is True
    assert body["founder"]["slots_remaining"] == 100
    assert body["founder"]["slots_total"] == 100

    by_tier = {t["tier"]: t for t in body["tiers"]}
    assert by_tier["try"]["is_free_now"] is True
    assert by_tier["try"]["effective_price_usd_per_month"] == 0.0
    assert by_tier["try"]["list_price_usd_per_month"] == 19.0
    assert by_tier["personal"]["is_free_now"] is False
    assert by_tier["personal"]["effective_price_usd_per_month"] == 79.0
    assert by_tier["frontier"]["effective_price_usd_per_month"] == 299.0


def test_api_pricing_flips_when_founder_slots_exhausted(tmp_path: Path):
    """When the counter hits zero, Try tier price flips to list ($19/mo)."""
    from pmc.storage.founders import FounderTracker
    tracker = FounderTracker(tmp_path / "storage")
    for i in range(100):
        tracker.grant_if_available(f"early-{i}")

    client, _, _ = _storage_client(tmp_path)
    body = client.get("/v1/pricing").json()
    assert body["founder"]["try_tier_free"] is False
    assert body["founder"]["slots_remaining"] == 0
    by_tier = {t["tier"]: t for t in body["tiers"]}
    assert by_tier["try"]["is_free_now"] is False
    assert by_tier["try"]["effective_price_usd_per_month"] == 19.0


def test_api_run_grants_founder_status_on_first_try_training(tmp_path: Path):
    """Submitting a non-dry-run on the Try base grants founder status."""
    client, _, _ = _storage_client(tmp_path)
    client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A reflection.", "text/markdown")},
        data={"kind": "text"},
    )
    r = client.post(
        "/v1/users/alex/runs",
        json={
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "skip_eval": True,
            "skip_deploy": True,
        },
    )
    body = r.json()
    assert body["founder"]["is_founder"] is True
    assert body["founder"]["training_free"] is True

    pricing = client.get("/v1/pricing").json()
    assert pricing["founder"]["slots_remaining"] == 99


def test_api_run_dry_run_does_not_consume_founder_slot(tmp_path: Path):
    """Dry runs should not burn a founder grant."""
    client, _, _ = _storage_client(tmp_path)
    client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A reflection.", "text/markdown")},
        data={"kind": "text"},
    )
    client.post(
        "/v1/users/alex/runs",
        json={
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "dry_run": True,
        },
    )
    pricing = client.get("/v1/pricing").json()
    assert pricing["founder"]["slots_remaining"] == 100


def test_api_run_personal_tier_does_not_consume_founder_slot(tmp_path: Path):
    """Founder freebie is Try-tier only. Training on Personal doesn't burn a slot."""
    client, _, _ = _storage_client(tmp_path)
    client.post(
        "/v1/users/alex/sources/upload",
        files={"file": ("note.md", b"A reflection.", "text/markdown")},
        data={"kind": "text"},
    )
    client.post(
        "/v1/users/alex/runs",
        json={
            "base_model": "Qwen/Qwen3.6-27B",  # Personal tier
            "skip_eval": True,
            "skip_deploy": True,
        },
    )
    pricing = client.get("/v1/pricing").json()
    assert pricing["founder"]["slots_remaining"] == 100


def test_api_run_no_data_completes_with_no_data_status(tmp_path: Path):
    """Submitting a run with no ingested data should complete with result.status = no_data."""
    client, _, _ = _storage_client(tmp_path)
    job = client.post(
        "/v1/users/ghost/runs",
        json={"dry_run": True},
    ).json()
    import time
    body = None
    for _ in range(50):
        body = client.get(f"/v1/users/ghost/runs/{job['job_id']}").json()
        if body["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert body is not None
    assert body["result"] is not None
    assert body["result"]["status"] == "no_data"
