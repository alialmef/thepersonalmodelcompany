"""Unit tests for the BYOM agent layer — crypto + provider registry +
AuthStore provider-config CRUD. Provider HTTP calls are covered by
integration tests with live keys; this file is offline-only.
"""

from __future__ import annotations

import os

import pytest

from pmc.agent import crypto
from pmc.agent.providers.registry import (
    KNOWN_PROVIDERS,
    get_provider,
    is_known_provider,
)
from pmc.auth.store import AuthStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encrypt_secret(monkeypatch):
    """Bind a deterministic encryption secret + clear the fernet cache so
    tests don't bleed key state into each other."""
    monkeypatch.setenv("PMC_KEY_ENCRYPTION_SECRET", "test-secret-must-be-stable")
    crypto._fernet.cache_clear()
    yield
    crypto._fernet.cache_clear()


@pytest.fixture
def store(tmp_path):
    return AuthStore(tmp_path)


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------


def test_crypto_round_trip(encrypt_secret):
    plain = "sk-ant-fake-key-12345"
    ct = crypto.encrypt(plain)
    assert ct != plain
    assert crypto.decrypt(ct) == plain


def test_crypto_different_ciphertexts_for_same_plaintext(encrypt_secret):
    """Fernet uses a random IV, so two encryptions of the same input
    must produce different ciphertexts (defends against equality
    inference of stored keys)."""
    a = crypto.encrypt("same-plaintext")
    b = crypto.encrypt("same-plaintext")
    assert a != b
    assert crypto.decrypt(a) == crypto.decrypt(b) == "same-plaintext"


def test_crypto_raises_when_secret_missing(monkeypatch):
    monkeypatch.delenv("PMC_KEY_ENCRYPTION_SECRET", raising=False)
    crypto._fernet.cache_clear()
    assert not crypto.is_configured()
    with pytest.raises(crypto.EncryptionNotConfigured):
        crypto.encrypt("anything")


def test_crypto_decrypt_with_wrong_secret_fails(encrypt_secret):
    ct = crypto.encrypt("secret-payload")
    # Rotate the env secret; previously-encrypted blobs must no longer decrypt
    crypto._fernet.cache_clear()
    os.environ["PMC_KEY_ENCRYPTION_SECRET"] = "different-secret"
    try:
        with pytest.raises(Exception):
            crypto.decrypt(ct)
    finally:
        crypto._fernet.cache_clear()


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def test_known_providers_metadata_complete():
    ids = {p["id"] for p in KNOWN_PROVIDERS}
    assert ids == {"anthropic", "openai", "google", "openrouter"}
    for p in KNOWN_PROVIDERS:
        assert p["label"]
        assert isinstance(p["default_models"], list)
        assert p["default_models"], f"{p['id']} has no default models"


def test_is_known_provider():
    assert is_known_provider("anthropic")
    assert is_known_provider("openrouter")
    assert not is_known_provider("foobar")


def test_get_provider_returns_instance_for_each_id():
    for pid in ("anthropic", "openai", "google", "openrouter"):
        p = get_provider(pid)
        assert p is not None
        assert p.name == pid


def test_get_provider_unknown_returns_none():
    assert get_provider("nonexistent") is None


# ---------------------------------------------------------------------------
# AuthStore provider config CRUD
# ---------------------------------------------------------------------------


def test_set_get_clear_provider_config(store):
    acct = store.get_or_create_account("alice@example.com")
    assert store.get_provider_config(acct.id) is None

    store.set_provider_config(
        acct.id, provider="anthropic", model="claude-sonnet-4-6",
        api_key_ciphertext="ENC:xxxx",
    )
    cfg = store.get_provider_config(acct.id)
    assert cfg is not None
    assert cfg["provider"] == "anthropic"
    assert cfg["model"] == "claude-sonnet-4-6"
    assert cfg["api_key_ciphertext"] == "ENC:xxxx"
    assert cfg["updated_at"] is not None

    # Idempotent upsert — second call replaces, not duplicates
    store.set_provider_config(
        acct.id, provider="openai", model="gpt-5", api_key_ciphertext="ENC:yyyy",
    )
    cfg = store.get_provider_config(acct.id)
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-5"

    store.clear_provider_config(acct.id)
    assert store.get_provider_config(acct.id) is None


def test_provider_config_is_per_account(store):
    a = store.get_or_create_account("a@example.com")
    b = store.get_or_create_account("b@example.com")
    store.set_provider_config(
        a.id, provider="anthropic", model="claude-opus-4-7",
        api_key_ciphertext="ENC:aaa",
    )
    store.set_provider_config(
        b.id, provider="openai", model="gpt-5", api_key_ciphertext="ENC:bbb",
    )
    assert store.get_provider_config(a.id)["provider"] == "anthropic"
    assert store.get_provider_config(b.id)["provider"] == "openai"


def test_account_deletion_cascades_provider_config(store):
    """Deleting an account via the FK cascade should drop its provider
    config so a stale key doesn't outlive the account."""
    a = store.get_or_create_account("doomed@example.com")
    store.set_provider_config(
        a.id, provider="anthropic", model="claude-haiku-4-5-20251001",
        api_key_ciphertext="ENC:zzz",
    )
    # Delete via raw SQL to mimic the cascade path
    with store._connect() as conn:
        sql = "DELETE FROM accounts WHERE id = %s"
        if store.kind == "sqlite":
            sql = sql.replace("%s", "?")
        conn.execute(sql, (a.id,))
    assert store.get_provider_config(a.id) is None
