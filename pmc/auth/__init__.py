"""Email-anchored accounts for PMC.

The Mac app used to identify a user by a UUID stored in localStorage —
if the user wiped their browser data, their model was effectively
gone. This module adds real accounts backed by email so identity
survives device changes and the founder-tier counter can be keyed to
something durable.

V1 design choices (Ali, May 2026):
  * Email-only — no passwords. Magic link sent via Resend.
  * Code-paste flow rather than deep links. The user gets a 6-char
    code, types it back into the app. Simpler than registering a
    pmc-app:// URL scheme; we can layer deep links + Apple Sign In
    on later (V2).
  * Opaque session tokens (256-bit random), stored hashed in the
    accounts DB. No JWT signing keys to rotate.
  * SQLite by default at <PMC_DEV_ROOT>/auth.db. Postgres if
    DATABASE_URL is set in env (Railway's Postgres add-on does this
    automatically).
  * Anonymous-UUID migration: on first sign-in, the existing
    pmc_user_id in localStorage is claimed by the new account via
    POST /v1/auth/claim. Disk layout doesn't move — the mapping
    table just records which account owns each pmc_user_id.

The auth router mounts at `/v1/auth/*`. A FastAPI dependency
`require_session` gates the user-scoped endpoints in `pmc/serve/api.py`.
"""

from pmc.auth.router import auth_router
from pmc.auth.middleware import (
    AuthSession,
    require_session,
    optional_session,
)
from pmc.auth.store import AuthStore, Account, Session

__all__ = [
    "auth_router",
    "AuthSession",
    "AuthStore",
    "Account",
    "Session",
    "require_session",
    "optional_session",
]
