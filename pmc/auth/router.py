"""HTTP surface for the auth flow.

Endpoints:

  POST /v1/auth/email      — { email } → issues a one-time code,
                              emails it. 200 even when the email
                              doesn't exist yet (we create on demand),
                              so the response doesn't leak account
                              existence.

  POST /v1/auth/exchange   — { email, code } → on success, returns
                              { session_token, account, pmc_user_ids }.
                              On failure, 401.

  GET  /v1/me              — requires session. Returns the current
                              account + claimed pmc_user_ids.

  POST /v1/auth/claim      — requires session, body { pmc_user_id }.
                              Binds the anonymous id to the account.
                              Returns the updated list.

  POST /v1/auth/signout    — invalidates the presented session.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from pmc.auth.email import send_login_code
from pmc.auth.middleware import AuthSession, require_session
from pmc.auth.store import Account, AuthStore
from pmc.auth.tokens import (
    hash_token,
    new_code,
    new_session_token,
    normalize_code,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EmailRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class EmailResponse(BaseModel):
    ok: bool
    detail: str
    email_method: Optional[str] = None  # 'resend' | 'console' (dev)


class ExchangeRequest(BaseModel):
    email: str
    code: str


class ExchangeResponse(BaseModel):
    session_token: str
    account: dict
    pmc_user_ids: list[str]


class ClaimRequest(BaseModel):
    pmc_user_id: str = Field(..., min_length=1, max_length=128)


class ClaimResponse(BaseModel):
    ok: bool
    pmc_user_ids: list[str]


class MeResponse(BaseModel):
    account: dict
    pmc_user_ids: list[str]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


auth_router = APIRouter(prefix="/v1/auth", tags=["auth"])


# Loose RFC-5322-ish — we don't need a parser, just to reject obvious
# garbage at the API boundary. Real validation happens at the email
# delivery layer.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _get_store(request: Request) -> AuthStore:
    store = getattr(request.app.state, "auth_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="auth backend not configured",
        )
    return store


@auth_router.post("/email", response_model=EmailResponse)
def request_email_code(req: EmailRequest, request: Request) -> EmailResponse:
    """Send a one-time login code to `email`. Creates the account
    lazily if it doesn't exist yet — auth and sign-up are the same
    surface, which keeps the UX one step instead of two.

    Always returns 200 on a syntactically valid email so we don't
    leak whether the address is already an account.
    """
    store = _get_store(request)
    email = req.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="invalid email")

    account = store.get_or_create_account(email)
    code = new_code()
    store.set_login_code(account.id, code, ttl_minutes=15)
    result = send_login_code(email, code)

    if not result.delivered:
        # Email provider failed (Resend down, etc). Surface the
        # error rather than silently dropping — better to tell the
        # user "try again" than have them watch a code that won't
        # arrive.
        log.warning("[auth] code send failed: %s", result.detail)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="couldn't send email — try again in a moment",
        )

    return EmailResponse(
        ok=True,
        detail="code sent" if result.method == "resend" else "code sent (dev console)",
        email_method=result.method,
    )


@auth_router.post("/exchange", response_model=ExchangeResponse)
def exchange_code(req: ExchangeRequest, request: Request) -> ExchangeResponse:
    """Trade a (email, code) pair for a session token.

    Constant-time wrong-code behavior: we always look up by email,
    always normalize the code, and only return success when the
    stored row matches. Two 401 paths are intentionally
    indistinguishable so an attacker can't tell whether the email
    or the code was wrong.
    """
    store = _get_store(request)
    email = req.email.strip().lower()
    code = normalize_code(req.code)
    if not _EMAIL_RE.match(email) or not code:
        raise HTTPException(status_code=400, detail="invalid request")

    # Find the account — but don't lazy-create here; if the email
    # was never registered, the exchange must fail.
    account = _account_by_email(store, email)
    if account is None or not store.consume_login_code(account.id, code):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="code didn't match or has expired",
        )

    token = new_session_token()
    device_label = request.headers.get("user-agent", "")[:120]
    store.create_session(account.id, hash_token(token), ttl_days=30, device_label=device_label)

    return ExchangeResponse(
        session_token=token,
        account=_account_dict(account),
        pmc_user_ids=store.list_pmc_users(account.id),
    )


@auth_router.get("/me", response_model=MeResponse)
def whoami(auth: AuthSession = Depends(require_session), request: Request = None) -> MeResponse:
    store = _get_store(request)
    return MeResponse(
        account=_account_dict(auth.account),
        pmc_user_ids=store.list_pmc_users(auth.account.id),
    )


@auth_router.post("/claim", response_model=ClaimResponse)
def claim_pmc_user(
    req: ClaimRequest,
    auth: AuthSession = Depends(require_session),
    request: Request = None,
) -> ClaimResponse:
    """Bind an anonymous pmc_user_id to the authenticated account.

    Conservative: if the pmc_user_id is already bound to a different
    account, we DO NOT take it over. That'd be a route to data theft
    by guessing UUIDs. The endpoint returns ok=False with the
    account's current list unchanged.
    """
    store = _get_store(request)
    ok = store.claim_pmc_user(auth.account.id, req.pmc_user_id.strip())
    return ClaimResponse(
        ok=ok,
        pmc_user_ids=store.list_pmc_users(auth.account.id),
    )


@auth_router.post("/signout")
def sign_out(auth: AuthSession = Depends(require_session), request: Request = None) -> dict:
    store = _get_store(request)
    store.delete_session(auth.session.token_hash)
    return {"ok": True}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _account_dict(account: Account) -> dict:
    cpe = account.subscription_current_period_end
    return {
        "id": account.id,
        "email": account.email,
        "created_at": account.created_at.isoformat(),
        "subscription": {
            "is_subscribed": account.is_subscribed(),
            "status": account.subscription_status,
            "tier": account.subscription_tier,
            "current_period_end": cpe.isoformat() if cpe else None,
        },
    }


def _account_by_email(store: AuthStore, email: str) -> Optional[Account]:
    """Look up by email without lazy-creating. Goes through the store's
    private connection so the result carries the full billing column
    set (the `accounts` row includes subscription state)."""
    from pmc.auth.store import _ACCOUNT_SELECT, _row_to_account
    with store._connect() as conn:  # type: ignore[attr-defined]
        sql = _ACCOUNT_SELECT + " WHERE email = %s"
        if store.kind == "sqlite":
            sql = sql.replace("%s", "?")
        row = conn.execute(sql, (email,)).fetchone()
        if not row:
            return None
        return _row_to_account(row, store.kind)
