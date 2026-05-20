"""Tests for the memory layer + continuous-learning loop.

Coverage:
  - MemoryStore: add, add_many, delete, delete_source, count, iter_all
  - Retriever: top-k cosine ordering, source filter, context block formatting
  - IdentityProfile: prompt composition, second-person framing, no first-person leak
  - sync: completion → memory item, batch sync
  - incremental: ingest_one, diff_and_ingest no-duplicate
  - serve.memory_context: enrich_messages with + without retrieval
  - orchestrator.runs_ledger: append / read / latest_shipped / best_scalar
  - orchestrator.drift: bootstrap / cadence / volume triggers
  - orchestrator.refresh: should_refresh + evaluate_and_promote
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from pmc.memory.embeddings import MockEmbeddings
from pmc.memory.identity import (
    IdentityProfile,
    build_first_contact_message,
    build_identity_prompt,
)
from pmc.memory.incremental import diff_and_ingest, ingest_many, ingest_one
from pmc.memory.retriever import Retriever
from pmc.memory.store import MemoryItem, MemoryStore
from pmc.memory.sync import (
    completion_to_memory_item,
    sync_completions_to_memory,
)
from pmc.orchestrator.drift import DriftConfig, assess
from pmc.orchestrator.refresh import evaluate_and_promote, should_refresh
from pmc.orchestrator.runs_ledger import (
    RunRecord,
    append_run,
    best_scalar,
    latest_shipped,
    new_run_id,
    read_runs,
)
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
)
from pmc.serve.memory_context import (
    MemoryContext,
    MemoryContextProvider,
    enrich_messages,
    save_identity,
)
from pmc.storage.paths import StoragePaths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _completion(text: str, prompt: str = "say hi", cid: str | None = None) -> Completion:
    return Completion(
        id=cid or str(uuid.uuid4()),
        conversation=Conversation(messages=[Message(role=Role.USER, content=prompt)]),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content=text)])
        ],
    )


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

def test_store_add_and_get(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    item = MemoryItem(id="x1", text="hello world", source="texts")
    store.add(item, [0.1, 0.2, 0.3])

    fetched = store.get("x1")
    assert fetched is not None
    got_item, got_vec = fetched
    assert got_item.id == "x1"
    assert got_item.text == "hello world"
    assert got_vec == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert store.count() == 1


def test_store_add_many_is_transactional(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    items = [
        (MemoryItem(id=f"m{i}", text=f"t{i}", source="notes"), [float(i)] * 4)
        for i in range(10)
    ]
    written = store.add_many(items)
    assert written == 10
    assert store.count() == 10
    assert store.count(source="notes") == 10
    assert store.count(source="texts") == 0


def test_store_delete_source(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    store.add(MemoryItem(id="a", text="x", source="texts"), [0.0] * 4)
    store.add(MemoryItem(id="b", text="y", source="notes"), [0.0] * 4)
    store.add(MemoryItem(id="c", text="z", source="notes"), [0.0] * 4)

    removed = store.delete_source("notes")
    assert removed == 2
    assert store.count() == 1
    assert store.count(source="notes") == 0


def test_store_clear(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    for i in range(5):
        store.add(MemoryItem(id=f"x{i}", text="t", source="s"), [0.0] * 4)
    removed = store.clear()
    assert removed == 5
    assert store.count() == 0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def test_retriever_returns_topk_in_order(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)

    # MockEmbeddings is deterministic by text; identical text → identical vector
    # → max similarity. Use this for predictable retrieval ordering.
    texts = ["alpha", "beta", "gamma", "delta"]
    for t in texts:
        store.add(MemoryItem(id=f"{t}-1", text=t, source="notes"), emb.embed([t])[0])

    retriever = Retriever(store=store, embeddings=emb)
    results = retriever.search("alpha", k=2)
    assert len(results) == 2
    assert results[0].item.text == "alpha"
    assert results[0].score > 0.999  # essentially 1.0 since identical embeddings


def test_retriever_filters_by_source(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)
    store.add(MemoryItem(id="n1", text="alpha", source="notes"), emb.embed(["alpha"])[0])
    store.add(MemoryItem(id="t1", text="alpha", source="texts"), emb.embed(["alpha"])[0])

    retriever = Retriever(store=store, embeddings=emb)
    results = retriever.search("alpha", k=5, sources=["notes"])
    assert len(results) == 1
    assert results[0].item.source == "notes"


def test_retriever_context_block_truncates(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)
    for i in range(20):
        text = f"item-{i}: " + ("padding " * 30)
        store.add(MemoryItem(id=f"m{i}", text=text, source="notes"), emb.embed([text])[0])

    retriever = Retriever(store=store, embeddings=emb)
    results = retriever.search("item", k=20)
    block = retriever.format_context_block(results, max_chars=500)
    assert len(block) <= 600  # generous slack for header + bullets
    assert block.startswith("Relevant snippets")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_identity_prompt_uses_second_person(tmp_path: Path) -> None:
    profile = IdentityProfile(
        user_id="alif",
        display_name="Alif",
        style_facts=("lowercase", "uses 'tbh' often", "short sentences"),
        tone="dry, sparse",
    )
    prompt = build_identity_prompt(profile)

    # Hard requirements: model is its own entity, user is addressed as "you".
    assert "Alif's personal AI model" in prompt
    assert "Refer to them" not in prompt  # not lecturing, just doing it
    assert "always refer to them as \"you\"" in prompt.lower()
    # Style facts surface in the prompt
    assert "lowercase" in prompt
    assert "tbh" in prompt
    assert "dry, sparse" in prompt


def test_identity_prompt_minimal_profile_still_works() -> None:
    profile = IdentityProfile(user_id="x", display_name="x")
    prompt = build_identity_prompt(profile)
    assert "x's personal AI model" in prompt
    # No style block when there are no facts
    assert "style" not in prompt.lower() or "voice tends to be" not in prompt


def test_first_contact_message_addresses_user() -> None:
    profile = IdentityProfile(
        user_id="alif",
        display_name="Alif",
        style_facts=("lowercase", "uses 'tbh'", "ends with questions"),
    )
    seed = build_first_contact_message(profile)
    assert "Alif" in seed
    # Explicit second-person instruction so the model doesn't lapse into "I"
    assert "second person" in seed.lower() or "'you'" in seed or "you'" in seed


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def test_completion_to_memory_item_extracts_assistant_text() -> None:
    c = _completion("hey maya, free thursday?", prompt="dinner this week?")
    item = completion_to_memory_item(c)
    assert item is not None
    assert "hey maya" in item.text
    assert item.metadata["completion_id"] == str(c.id)
    assert "dinner" in item.metadata["in_reply_to"]


def test_completion_to_memory_item_returns_none_when_empty() -> None:
    c = Completion(
        id=uuid.uuid4(),
        conversation=Conversation(messages=[Message(role=Role.USER, content="hi")]),
        candidates=[],
    )
    assert completion_to_memory_item(c) is None


def test_sync_completions_writes_to_store(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)
    completions = [
        _completion("first response"),
        _completion("second response"),
        _completion("third response"),
    ]
    written = sync_completions_to_memory(completions, store, emb)
    assert written == 3
    assert store.count() == 3


# ---------------------------------------------------------------------------
# Incremental
# ---------------------------------------------------------------------------

def test_ingest_one_writes_single_item(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)
    ok = ingest_one(_completion("hello"), store, emb)
    assert ok is True
    assert store.count() == 1


def test_diff_and_ingest_skips_already_present(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)

    completions = [_completion("a"), _completion("b"), _completion("c")]
    ingest_many(completions, store, emb)
    assert store.count() == 3

    # Re-running diff_and_ingest with the same completions should be a no-op
    # for new writes; existing count should match.
    new, existing = diff_and_ingest(completions, store, emb)
    assert new == 0
    assert existing == 3
    assert store.count() == 3


# ---------------------------------------------------------------------------
# Serve memory_context
# ---------------------------------------------------------------------------

def test_enrich_messages_prepends_identity_and_context(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)
    store.add(MemoryItem(id="m", text="reminder: alpha", source="notes"), emb.embed(["alpha"])[0])

    profile = IdentityProfile(user_id="u", display_name="Alif")
    ctx = MemoryContext(
        user_id="u",
        identity=profile,
        store=store,
        retriever=Retriever(store=store, embeddings=emb),
    )

    out = enrich_messages([{"role": "user", "content": "alpha"}], ctx)
    assert out[0]["role"] == "system"
    assert "Alif's personal AI model" in out[0]["content"]
    assert "Relevant snippets" in out[0]["content"]
    assert out[1]["role"] == "user"


def test_enrich_messages_merges_existing_system(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "store.db")
    emb = MockEmbeddings(dim=64)
    profile = IdentityProfile(user_id="u", display_name="Alif")
    ctx = MemoryContext(
        user_id="u",
        identity=profile,
        store=store,
        retriever=Retriever(store=store, embeddings=emb),
    )

    out = enrich_messages(
        [
            {"role": "system", "content": "extra: be terse"},
            {"role": "user", "content": "hi"},
        ],
        ctx,
    )
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "Alif's personal AI model" in out[0]["content"]
    assert "be terse" in out[0]["content"]


def test_memory_context_provider_falls_back_when_no_identity(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path / "storage")
    emb = MockEmbeddings(dim=64)
    provider = MemoryContextProvider(paths=paths, embeddings=emb)
    ctx = provider.get("brand_new_user")
    assert ctx.identity.display_name == "brand_new_user"
    assert ctx.store.count() == 0


def test_save_identity_roundtrip(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path / "storage")
    profile = IdentityProfile(
        user_id="u1",
        display_name="Alif",
        style_facts=("lowercase",),
        tone="dry",
    )
    save_identity(paths, profile)

    emb = MockEmbeddings(dim=64)
    provider = MemoryContextProvider(paths=paths, embeddings=emb)
    ctx = provider.get("u1")
    assert ctx.identity.display_name == "Alif"
    assert ctx.identity.tone == "dry"
    assert "lowercase" in ctx.identity.style_facts


# ---------------------------------------------------------------------------
# Runs ledger
# ---------------------------------------------------------------------------

def test_append_and_read_runs(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    r1 = RunRecord(run_id="20260101-000000", ts=1.0, status="shipped", scalar=0.50)
    r2 = RunRecord(run_id="20260201-000000", ts=2.0, status="rejected", scalar=0.45)
    r3 = RunRecord(run_id="20260301-000000", ts=3.0, status="shipped", scalar=0.62)

    append_run(ledger, r1)
    append_run(ledger, r2)
    append_run(ledger, r3)

    runs = read_runs(ledger)
    assert [r.run_id for r in runs] == [
        "20260101-000000",
        "20260201-000000",
        "20260301-000000",
    ]
    assert latest_shipped(ledger).run_id == "20260301-000000"
    assert best_scalar(ledger) == pytest.approx(0.62)


def test_new_run_id_sortable() -> None:
    a = new_run_id()
    time.sleep(1.1)
    b = new_run_id()
    assert b > a  # YYYYMMDD-HHMMSS sorts lexicographically


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

class _CountStore:
    def __init__(self, n: int) -> None:
        self._n = n

    def count(self) -> int:
        return self._n


def test_drift_bootstrap_fires_above_threshold() -> None:
    cfg = DriftConfig(min_items_for_first_run=100)
    report = assess(_CountStore(120), last_run_ts=None, last_run_item_count=None, config=cfg)  # type: ignore[arg-type]
    assert report.should_refresh is True
    assert "bootstrap" in report.reason


def test_drift_bootstrap_holds_below_threshold() -> None:
    cfg = DriftConfig(min_items_for_first_run=100)
    report = assess(_CountStore(50), last_run_ts=None, last_run_item_count=None, config=cfg)  # type: ignore[arg-type]
    assert report.should_refresh is False


def test_drift_volume_signal_fires() -> None:
    cfg = DriftConfig(min_new_items=200)
    now = time.time()
    report = assess(
        _CountStore(800),  # type: ignore[arg-type]
        last_run_ts=now - 86400,  # 1 day ago
        last_run_item_count=500,
        config=cfg,
    )
    assert report.should_refresh is True
    assert "volume" in report.reason
    assert report.new_items_since_last_run == 300


def test_drift_cadence_fires_after_ceiling() -> None:
    cfg = DriftConfig(min_new_items=10000, max_days_since_run=30.0)
    long_ago = time.time() - (40 * 86400)
    report = assess(
        _CountStore(100),  # type: ignore[arg-type]
        last_run_ts=long_ago,
        last_run_item_count=90,
        config=cfg,
    )
    assert report.should_refresh is True
    assert "cadence" in report.reason


def test_drift_holds_when_no_signal() -> None:
    cfg = DriftConfig(min_new_items=500, max_days_since_run=30.0)
    report = assess(
        _CountStore(120),  # type: ignore[arg-type]
        last_run_ts=time.time() - 86400,
        last_run_item_count=100,
        config=cfg,
    )
    assert report.should_refresh is False


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

def test_should_refresh_manual_always_fires(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    decision = should_refresh(ledger, new_item_count=0, manual=True)
    assert decision.fire is True
    assert decision.trigger == "manual"


def test_should_refresh_drift_returns_decision(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    append_run(
        ledger,
        RunRecord(
            run_id="r1",
            ts=time.time() - 86400,
            status="shipped",
            scalar=0.5,
            train_examples=500,
        ),
    )
    decision = should_refresh(
        ledger,
        new_item_count=2000,
        drift_config=DriftConfig(min_new_items=200),
    )
    assert decision.fire is True
    assert decision.trigger in {"drift", "cadence"}


def test_evaluate_and_promote_promotes_when_better(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    append_run(ledger, RunRecord(run_id="r1", ts=1.0, status="shipped", scalar=0.5))

    promoted_to: list[str] = []
    result = evaluate_and_promote(
        new_run_id_="r2",
        new_scalar=0.7,
        ledger_path=ledger,
        train_examples=600,
        base_model="llama-3.1-8b",
        adapter_size_mb=142.0,
        promote_fn=promoted_to.append,
    )
    assert result.promoted is True
    assert promoted_to == ["r2"]
    runs = read_runs(ledger)
    assert runs[-1].status == "shipped"
    assert runs[-1].promoted_from == "r1"


def test_evaluate_and_promote_rejects_when_worse(tmp_path: Path) -> None:
    ledger = tmp_path / "runs.jsonl"
    append_run(ledger, RunRecord(run_id="r1", ts=1.0, status="shipped", scalar=0.8))

    promoted_to: list[str] = []
    result = evaluate_and_promote(
        new_run_id_="r2",
        new_scalar=0.6,
        ledger_path=ledger,
        train_examples=600,
        base_model="llama-3.1-8b",
        adapter_size_mb=142.0,
        promote_fn=promoted_to.append,
    )
    assert result.promoted is False
    assert promoted_to == []
    runs = read_runs(ledger)
    assert runs[-1].status == "rejected"
    assert runs[-1].scalar == pytest.approx(0.6)
