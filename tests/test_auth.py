"""Unit tests for the auth layer — store, tokens, router.

We use an in-memory-ish SQLite (a tmp file) and the FastAPI TestClient.
No real network. The email path is short-circuited by the "console"
fallback (no RESEND_API_KEY in test env).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pmc.auth import auth_router
from pmc.auth.store import AuthStore
from pmc.auth.tokens import (
    CODE_LEN,
    hash_token,
    new_code,
    new_session_token,
    normalize_code,
)


# ---------- token primitives ----------


def test_new_code_shape():
    code = new_code()
    # 3 chars, dash, 3 chars
    assert len(code) == 2 * CODE_LEN // 2 + 1
    assert code[3] == "-"
    # ambiguous chars (O/0/1/I/L) never appear
    for c in code.replace("-", ""):
        assert c not in {"O", "0", "1", "I", "L"}


def test_normalize_code():
    assert normalize_code("abc def") == "ABCDEF"
    assert normalize_code("ABC-DEF") == "ABCDEF"
    assert normalize_code(" a-b-c ") == "ABC"
    assert normalize_code("") == ""


def test_session_token_uniqueness():
    seen = {new_session_token() for _ in range(100)}
    assert len(seen) == 100  # no collisions


def test_hash_token_stable():
    t = new_session_token()
    assert hash_token(t) == hash_token(t)
    assert hash_token(t) != hash_token(new_session_token())


# ---------- store ----------


@pytest.fixture
def store(tmp_path: Path) -> AuthStore:
    # Force SQLite by clearing DATABASE_URL in this test session.
    os.environ.pop("DATABASE_URL", None)
    return AuthStore(storage_root=tmp_path)


def test_get_or_create_account_is_idempotent(store: AuthStore):
    a1 = store.get_or_create_account("ALI@example.com")
    a2 = store.get_or_create_account("ali@example.com")
    assert a1.id == a2.id
    assert a1.email == "ali@example.com"


def test_get_or_create_account_rejects_invalid(store: AuthStore):
    with pytest.raises(ValueError):
        store.get_or_create_account("not-an-email")


def test_login_code_consume_one_shot(store: AuthStore):
    acct = store.get_or_create_account("a@b.com")
    code = new_code()
    store.set_login_code(acct.id, code)
    # Correct on first try.
    assert store.consume_login_code(acct.id, code) is True
    # Replays fail.
    assert store.consume_login_code(acct.id, code) is False


def test_login_code_wrong_code_doesnt_consume(store: AuthStore):
    acct = store.get_or_create_account("a@b.com")
    store.set_login_code(acct.id, "ABCDEF")
    assert store.consume_login_code(acct.id, "ZZZZZZ") is False
    # The correct code still works after the failure.
    assert store.consume_login_code(acct.id, "ABCDEF") is True


def test_session_lifecycle(store: AuthStore):
    acct = store.get_or_create_account("a@b.com")
    token = new_session_token()
    th = hash_token(token)
    store.create_session(acct.id, th)
    sess = store.get_session(th)
    assert sess is not None
    assert sess.account_id == acct.id
    store.delete_session(th)
    assert store.get_session(th) is None


def test_claim_pmc_user_first_wins(store: AuthStore):
    a1 = store.get_or_create_account("a@b.com")
    a2 = store.get_or_create_account("c@d.com")
    assert store.claim_pmc_user(a1.id, "anon-1") is True
    # Same account re-claiming is a no-op success.
    assert store.claim_pmc_user(a1.id, "anon-1") is True
    # Different account trying to take it is rejected.
    assert store.claim_pmc_user(a2.id, "anon-1") is False
    # a1 keeps the binding.
    assert store.list_pmc_users(a1.id) == ["anon-1"]
    assert store.list_pmc_users(a2.id) == []


# ---------- router ----------


@pytest.fixture
def sent_codes(monkeypatch) -> list[tuple[str, str]]:
    """Intercept email sends so tests can read the codes back."""
    captured: list[tuple[str, str]] = []

    def fake_send(email: str, code: str, **kwargs):
        captured.append((email, code))
        from pmc.auth.email import EmailResult
        return EmailResult(delivered=True, method="test")

    monkeypatch.setattr("pmc.auth.router.send_login_code", fake_send)
    return captured


@pytest.fixture
def app(store: AuthStore) -> FastAPI:
    app = FastAPI()
    app.state.auth_store = store
    app.include_router(auth_router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _last_code_for(captured, email: str) -> str:
    for e, c in reversed(captured):
        if e == email:
            return c
    raise AssertionError(f"no code captured for {email}")


def test_email_request_creates_account_and_emits_code(
    client: TestClient, store: AuthStore, sent_codes
):
    resp = client.post("/v1/auth/email", json={"email": "new@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True

    # Account got created
    with store._connect() as conn:
        row = conn.execute(
            "SELECT id FROM accounts WHERE email = ?", ("new@example.com",)
        ).fetchone()
    assert row is not None

    # Code was dispatched (intercepted by sent_codes fixture).
    assert _last_code_for(sent_codes, "new@example.com")


def test_email_request_rejects_invalid(client: TestClient, sent_codes):
    resp = client.post("/v1/auth/email", json={"email": "not-email"})
    assert resp.status_code == 400


def test_exchange_happy_path(client: TestClient, sent_codes):
    client.post("/v1/auth/email", json={"email": "ali@example.com"})
    code = _last_code_for(sent_codes, "ali@example.com")

    resp = client.post("/v1/auth/exchange", json={
        "email": "ali@example.com",
        "code": code,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_token"]
    assert body["account"]["email"] == "ali@example.com"
    assert body["pmc_user_ids"] == []


def test_exchange_wrong_code_returns_401(client: TestClient, sent_codes):
    client.post("/v1/auth/email", json={"email": "x@y.com"})
    resp = client.post("/v1/auth/exchange", json={
        "email": "x@y.com",
        "code": "WRO-NGG",
    })
    assert resp.status_code == 401


def test_exchange_unknown_email_returns_401(client: TestClient, sent_codes):
    resp = client.post("/v1/auth/exchange", json={
        "email": "never-registered@example.com",
        "code": "ABC-DEF",
    })
    assert resp.status_code == 401


def test_me_requires_session(client: TestClient):
    resp = client.get("/v1/auth/me")
    assert resp.status_code == 401


def test_full_flow_email_exchange_me(client: TestClient, sent_codes):
    client.post("/v1/auth/email", json={"email": "founder@example.com"})
    code = _last_code_for(sent_codes, "founder@example.com")

    ex = client.post("/v1/auth/exchange", json={
        "email": "founder@example.com",
        "code": code,
    })
    token = ex.json()["session_token"]

    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["account"]["email"] == "founder@example.com"


def test_claim_binds_pmc_user_id(client: TestClient, sent_codes):
    client.post("/v1/auth/email", json={"email": "claimer@example.com"})
    code = _last_code_for(sent_codes, "claimer@example.com")
    ex = client.post("/v1/auth/exchange", json={
        "email": "claimer@example.com",
        "code": code,
    })
    token = ex.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/v1/auth/claim",
        json={"pmc_user_id": "anon-abc"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert "anon-abc" in body["pmc_user_ids"]


def test_signout_invalidates_session(client: TestClient, sent_codes):
    client.post("/v1/auth/email", json={"email": "user@example.com"})
    code = _last_code_for(sent_codes, "user@example.com")
    ex = client.post("/v1/auth/exchange", json={"email": "user@example.com", "code": code})
    token = ex.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Works before signout.
    assert client.get("/v1/auth/me", headers=headers).status_code == 200
    # Sign out.
    so = client.post("/v1/auth/signout", headers=headers)
    assert so.status_code == 200
    # Token now invalid.
    assert client.get("/v1/auth/me", headers=headers).status_code == 401
