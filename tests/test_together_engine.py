"""Tests for the Together AI engine adapter.

Together's API is OpenAI-compatible. We mock the OpenAI client to verify our
adapter wires the right model, messages, and LoRA reference, and that
streaming chunks pass through correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pmc.serve.engine_together import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_URL,
    TogetherEngine,
    set_together_adapter_id,
)
from pmc.serve.registry import AdapterRecord


def _record(user_id: str = "alex", with_adapter_id: bool = True) -> AdapterRecord:
    record = AdapterRecord(
        user_id=user_id,
        adapter_dir="/tmp/adapter",
        base_model=DEFAULT_BASE_MODEL,
    )
    if with_adapter_id:
        set_together_adapter_id(record, "alex-adapter-v1")
    return record


def _mock_openai_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 8) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def _mock_stream_chunks(texts: list[str]) -> list[MagicMock]:
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)
    # Final empty chunk (end of stream)
    end_chunk = MagicMock()
    end_chunk.choices = [MagicMock()]
    end_chunk.choices[0].delta.content = None
    chunks.append(end_chunk)
    return chunks


def test_defaults():
    assert "together" in DEFAULT_BASE_URL.lower()
    assert "Llama" in DEFAULT_BASE_MODEL or "llama" in DEFAULT_BASE_MODEL.lower()


def test_engine_requires_api_key():
    engine = TogetherEngine(api_key="")
    with pytest.raises(RuntimeError, match="API key"):
        engine._get_client()


def test_engine_uses_env_var_if_no_key_passed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "env-key")
    engine = TogetherEngine()
    assert engine._api_key == "env-key"


def test_chat_routes_lora_via_extra_body():
    """The adapter ID must end up in extra_body so Together routes to the right LoRA."""
    engine = TogetherEngine(api_key="test-key")
    record = _record()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_openai_response("Hello world")
    engine._client = fake_client

    text, usage = engine.chat(
        record=record,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=50,
        temperature=0.5,
    )
    assert text == "Hello world"
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 8

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_BASE_MODEL
    assert call_kwargs["max_tokens"] == 50
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["stream"] is False
    assert call_kwargs["extra_body"]["lora"] == "alex-adapter-v1"


def test_chat_without_adapter_id_skips_lora():
    """A record with no together_adapter_id should not pass a lora field."""
    engine = TogetherEngine(api_key="k")
    record = _record(with_adapter_id=False)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_openai_response("hi")
    engine._client = fake_client

    engine.chat(record=record, messages=[{"role": "user", "content": "x"}])
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in call_kwargs or "lora" not in call_kwargs.get("extra_body", {})


def test_chat_stream_yields_content_chunks():
    engine = TogetherEngine(api_key="k")
    record = _record()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(
        _mock_stream_chunks(["Hel", "lo ", "world"])
    )
    engine._client = fake_client

    chunks = list(engine.chat_stream(record, [{"role": "user", "content": "Hi"}]))
    assert chunks == ["Hel", "lo ", "world"]

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True
    assert call_kwargs["extra_body"]["lora"] == "alex-adapter-v1"


def test_chat_stream_skips_empty_deltas():
    """Some chunks have no content (role-only, etc.). Don't yield empty strings."""
    engine = TogetherEngine(api_key="k")
    record = _record()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(
        _mock_stream_chunks(["Hi", ""])  # second has empty content
    )
    engine._client = fake_client

    chunks = list(engine.chat_stream(record, [{"role": "user", "content": "x"}]))
    assert chunks == ["Hi"]


def test_chat_passes_stop_sequences():
    engine = TogetherEngine(api_key="k")
    record = _record()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_openai_response("ok")
    engine._client = fake_client

    engine.chat(record=record, messages=[{"role": "user", "content": "hi"}], stop=["\n"])
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stop"] == ["\n"]


def test_set_together_adapter_id_helper():
    record = AdapterRecord(user_id="u", adapter_dir="/x", base_model=DEFAULT_BASE_MODEL)
    assert record.metadata == {}
    set_together_adapter_id(record, "adapter-xyz")
    assert record.metadata["together_adapter_id"] == "adapter-xyz"


def test_upload_adapter_not_yet_implemented():
    engine = TogetherEngine(api_key="k")
    with pytest.raises(NotImplementedError):
        engine.upload_adapter("/some/path", "name")


def test_engine_implements_protocol():
    """TogetherEngine should be usable wherever InferenceEngine is expected."""
    from pmc.serve.engine import InferenceEngine
    engine = TogetherEngine(api_key="k")
    # runtime_checkable protocol check
    assert isinstance(engine, InferenceEngine)


def test_engine_lazy_imports_openai():
    """If openai isn't importable, _get_client gives a clear error."""
    engine = TogetherEngine(api_key="k")
    with patch.dict("sys.modules", {"openai": None}):
        # Force re-import attempt
        engine._client = None
        with pytest.raises((ImportError, AttributeError)):
            engine._get_client()
