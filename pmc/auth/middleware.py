"""FastAPI dependency for session-bound endpoints.

Two flavors:

  * `require_session(...)` — gates an endpoint behind a valid session.
    No session / expired session / unknown token → 401.

  * `optional_session(...)` — returns the session if present, None
    otherwise. Used during the V1 transition where anonymous flows
    still work side-by-side.

The auth store is loaded from the FastAPI app's state. `create_app`
in `pmc/serve/api.py` is responsible for attaching it so the
dependencies can pick it up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from pmc.auth.store import Account, AuthStore, Session
from pmc.auth.tokens import hash_token


@dataclass
class AuthSession:
    account: Account
    session: Session


def _store(request: Request) -> Optional[AuthStore]:
    return getattr(request.app.state, "auth_store", None)


def _extract_token(request: Request) -> Optional[str]:
    """Pull a session token out of the request. Supports both:
      - Authorization: Bearer <token>     (Mac app / curl)
      - X-PMC-Session: <token>            (defensive — works around
        any middleware that strips Authorization)
    """
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return request.headers.get("x-pmc-session") or None


def optional_session(request: Request) -> Optional[AuthSession]:
    """Return AuthSession if a valid token is presented, else None.
    Touches the session's last_seen_at."""
    store = _store(request)
    if store is None:
        return None
    token = _extract_token(request)
    if not token:
        return None
    th = hash_token(token)
    session = store.get_session(th)
    if session is None:
        return None
    account = store.get_account_by_id(session.account_id)
    if account is None:
        return None
    store.touch_session(th)
    return AuthSession(account=account, session=session)


def require_session(request: Request) -> AuthSession:
    sess = optional_session(request)
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not signed in",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return sess
