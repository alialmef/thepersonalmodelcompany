"""Account / session persistence.

Backed by SQLite by default (one file at `<storage_root>/auth.db`).
If `DATABASE_URL` is set in env, we connect to Postgres instead —
Railway's Postgres add-on injects this automatically.

The schema is identical between backends; we use a tiny SQL dialect
abstraction (`%s` placeholders translated to `?` for SQLite) so we
don't have to drag in SQLAlchemy. Three tables:

  accounts(id, email, created_at)
  account_sessions(token_hash, account_id, created_at, expires_at, last_seen_at)
  account_pmc_users(account_id, pmc_user_id, claimed_at)
  account_login_codes(account_id, code, created_at, expires_at, consumed_at)

The login-codes table is opportunistic — there's at most one valid
row per account at a time. We delete on consume and let the rest
expire-and-sweep on a slow cadence.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Account:
    id: str
    email: str
    created_at: datetime
    stripe_customer_id: Optional[str] = None
    subscription_status: Optional[str] = None
    subscription_tier: Optional[str] = None
    subscription_current_period_end: Optional[datetime] = None

    def is_subscribed(self) -> bool:
        """Active or trialing subscription that hasn't lapsed. Bills run
        only when this is True (or the user is a founder)."""
        if self.subscription_status not in ("active", "trialing"):
            return False
        # If we know the period end and it's in the past, treat as lapsed.
        # Stripe normally flips status to past_due / canceled itself, but
        # protect against missed webhook events.
        if self.subscription_current_period_end is not None:
            if self.subscription_current_period_end < _utcnow():
                return False
        return True


@dataclass
class Session:
    token_hash: str
    account_id: str
    created_at: datetime
    expires_at: datetime
    last_seen_at: datetime


# ---------------------------------------------------------------------------
# Connection setup
# ---------------------------------------------------------------------------


def _backend_kind() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url.startswith("postgres://") or url.startswith("postgresql://"):
        return "postgres"
    return "sqlite"


def _sqlite_path(storage_root: Path) -> Path:
    return Path(storage_root) / "auth.db"


def _translate_sql(sql: str, kind: str) -> str:
    """Crude dialect adaptation. We write SQL with `%s` placeholders
    (Postgres's psycopg style); convert to `?` for SQLite."""
    if kind == "sqlite":
        return re.sub(r"%s", "?", sql)
    return sql


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL,
    stripe_customer_id TEXT,
    subscription_status TEXT,
    subscription_tier TEXT,
    subscription_current_period_end TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_sessions (
    token_hash TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP NOT NULL,
    device_label TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_account ON account_sessions(account_id);

CREATE TABLE IF NOT EXISTS account_pmc_users (
    account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    pmc_user_id TEXT NOT NULL UNIQUE,
    claimed_at TIMESTAMP NOT NULL,
    PRIMARY KEY (account_id, pmc_user_id)
);
CREATE INDEX IF NOT EXISTS idx_pmc_users_pmc ON account_pmc_users(pmc_user_id);

CREATE TABLE IF NOT EXISTS account_login_codes (
    account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    code TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    consumed_at TIMESTAMP,
    PRIMARY KEY (account_id)
);

CREATE TABLE IF NOT EXISTS account_provider_configs (
    account_id TEXT PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    api_key_ciphertext TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
"""


class AuthStore:
    """Thread-safe wrapper. Holds a single connection per thread for
    SQLite (each thread needs its own) and a single shared connection
    for Postgres."""

    def __init__(self, storage_root: Path | str):
        self.storage_root = Path(storage_root)
        self.kind = _backend_kind()
        self._sqlite_lock = threading.RLock()
        self._sqlite_path: Optional[Path] = None
        self._pg_conn = None
        if self.kind == "sqlite":
            self._sqlite_path = _sqlite_path(self.storage_root)
            self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ---- backend connections ----

    @contextmanager
    def _connect(self) -> Iterator:
        if self.kind == "sqlite":
            with self._sqlite_lock:
                conn = sqlite3.connect(str(self._sqlite_path), isolation_level=None)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA foreign_keys=ON")
                    yield conn
                finally:
                    conn.close()
        else:
            import psycopg
            conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
            try:
                yield conn
            finally:
                conn.close()

    def _ensure_schema(self) -> None:
        statements = [s for s in _SCHEMA.split(";\n") if s.strip()]
        with self._connect() as conn:
            for stmt in statements:
                conn.execute(_translate_sql(stmt + ";", self.kind))
            self._migrate_accounts_for_billing(conn)

    def _migrate_accounts_for_billing(self, conn) -> None:
        """Add billing columns to existing `accounts` tables.

        CREATE TABLE IF NOT EXISTS won't add new columns to a pre-existing
        table, so run idempotent ALTER TABLE ADD COLUMN for each billing
        field. Both sqlite and Postgres treat duplicate-column errors as
        non-fatal; swallow them so re-runs work.
        """
        cols = [
            ("stripe_customer_id", "TEXT"),
            ("subscription_status", "TEXT"),
            ("subscription_tier", "TEXT"),
            ("subscription_current_period_end", "TIMESTAMP"),
        ]
        for name, sql_type in cols:
            try:
                conn.execute(f"ALTER TABLE accounts ADD COLUMN {name} {sql_type};")
            except Exception:
                # Already present — both backends raise; safe to ignore.
                pass

    # ---- account operations ----

    def get_or_create_account(self, email: str) -> Account:
        """Idempotent on email. Returns the existing row if present,
        otherwise inserts a new account."""
        email_lc = email.strip().lower()
        if not email_lc or "@" not in email_lc:
            raise ValueError("invalid email")
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(_ACCOUNT_SELECT + " WHERE email = %s", self.kind),
                (email_lc,),
            ).fetchone()
            if row:
                return _row_to_account(row, self.kind)
            new_id = _short_id()
            now = _utcnow()
            conn.execute(
                _translate_sql(
                    "INSERT INTO accounts (id, email, created_at) VALUES (%s, %s, %s)",
                    self.kind,
                ),
                (new_id, email_lc, now),
            )
            return Account(id=new_id, email=email_lc, created_at=now)

    def get_account_by_id(self, account_id: str) -> Optional[Account]:
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(_ACCOUNT_SELECT + " WHERE id = %s", self.kind),
                (account_id,),
            ).fetchone()
            return _row_to_account(row, self.kind) if row else None

    def get_account_by_stripe_customer(self, customer_id: str) -> Optional[Account]:
        """Reverse-lookup for webhook handlers — Stripe events carry the
        customer id, not our internal account id."""
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(
                    _ACCOUNT_SELECT + " WHERE stripe_customer_id = %s", self.kind
                ),
                (customer_id,),
            ).fetchone()
            return _row_to_account(row, self.kind) if row else None

    # ---- billing operations ----

    def set_stripe_customer_id(self, account_id: str, customer_id: str) -> None:
        """Persist Stripe's customer id on the account. Idempotent —
        overwrites any prior value (the same email/account should never
        hold two customers, but if Stripe were to issue one we'd want the
        latest)."""
        with self._connect() as conn:
            conn.execute(
                _translate_sql(
                    "UPDATE accounts SET stripe_customer_id = %s WHERE id = %s",
                    self.kind,
                ),
                (customer_id, account_id),
            )

    # ---- provider config (BYOM) operations ----

    def set_provider_config(
        self,
        account_id: str,
        *,
        provider: str,
        model: str,
        api_key_ciphertext: str,
    ) -> None:
        """Store/replace the user's provider+model+encrypted-key choice.

        Idempotent — overwrites any prior config for this account.
        Plaintext keys must never reach this layer; the caller is
        responsible for encrypting via pmc.agent.crypto before calling.
        """
        now = _utcnow()
        with self._connect() as conn:
            # UPSERT — sqlite's INSERT OR REPLACE works for both backends
            # via DELETE-then-INSERT to keep dialect overhead minimal.
            conn.execute(
                _translate_sql(
                    "DELETE FROM account_provider_configs WHERE account_id = %s",
                    self.kind,
                ),
                (account_id,),
            )
            conn.execute(
                _translate_sql(
                    "INSERT INTO account_provider_configs "
                    "(account_id, provider, model, api_key_ciphertext, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    self.kind,
                ),
                (account_id, provider, model, api_key_ciphertext, now),
            )

    def get_provider_config(self, account_id: str) -> Optional[dict]:
        """Return the stored (provider, model, ciphertext, updated_at)
        as a dict, or None if the user hasn't configured a provider yet.
        Caller decrypts the ciphertext just-in-time per request."""
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(
                    "SELECT provider, model, api_key_ciphertext, updated_at "
                    "FROM account_provider_configs WHERE account_id = %s",
                    self.kind,
                ),
                (account_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "provider": _get(row, 0, "provider"),
                "model": _get(row, 1, "model"),
                "api_key_ciphertext": _get(row, 2, "api_key_ciphertext"),
                "updated_at": _coerce_dt(_get(row, 3, "updated_at")),
            }

    def clear_provider_config(self, account_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                _translate_sql(
                    "DELETE FROM account_provider_configs WHERE account_id = %s",
                    self.kind,
                ),
                (account_id,),
            )

    def update_subscription_state(
        self,
        account_id: str,
        *,
        status: Optional[str],
        tier: Optional[str],
        current_period_end: Optional[datetime],
    ) -> None:
        """Write subscription state from a Stripe webhook. Pass None to
        clear (canceled / deleted subscription)."""
        with self._connect() as conn:
            conn.execute(
                _translate_sql(
                    "UPDATE accounts SET subscription_status = %s, "
                    "subscription_tier = %s, "
                    "subscription_current_period_end = %s WHERE id = %s",
                    self.kind,
                ),
                (status, tier, current_period_end, account_id),
            )

    # ---- login code operations ----

    def set_login_code(self, account_id: str, code: str, ttl_minutes: int = 15) -> None:
        """Store the normalized form (no dashes, uppercase) so the
        wire format ("ABC-DEF") and stored form match what the user
        types back (case-insensitive, dash-optional)."""
        from pmc.auth.tokens import normalize_code
        normalized = normalize_code(code)
        now = _utcnow()
        expires = now + timedelta(minutes=ttl_minutes)
        with self._connect() as conn:
            # Upsert — clobber any prior pending code so a user
            # rapidly hitting "send" can't accumulate active codes.
            conn.execute(
                _translate_sql("DELETE FROM account_login_codes WHERE account_id = %s", self.kind),
                (account_id,),
            )
            conn.execute(
                _translate_sql(
                    "INSERT INTO account_login_codes (account_id, code, created_at, expires_at) "
                    "VALUES (%s, %s, %s, %s)",
                    self.kind,
                ),
                (account_id, normalized, now, expires),
            )

    def consume_login_code(self, account_id: str, code: str) -> bool:
        """One-shot consumption: returns True if the code was valid +
        unexpired + not already used. Deletes the row on success so
        the code can't be replayed. Accepts the code in any case /
        with-or-without dashes — both ends normalize before comparing."""
        from pmc.auth.tokens import normalize_code
        code = normalize_code(code)
        now = _utcnow()
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(
                    "SELECT code, expires_at, consumed_at FROM account_login_codes "
                    "WHERE account_id = %s",
                    self.kind,
                ),
                (account_id,),
            ).fetchone()
            if not row:
                return False
            stored = row["code"] if hasattr(row, "__getitem__") else row[0]
            expires = _coerce_dt(row["expires_at"] if hasattr(row, "__getitem__") else row[1])
            consumed = row["consumed_at"] if hasattr(row, "__getitem__") else row[2]
            if consumed is not None:
                return False
            if expires < now:
                return False
            # Constant-time compare. Both have been normalized.
            from pmc.auth.tokens import constant_time_eq
            if not constant_time_eq(stored, code):
                return False
            conn.execute(
                _translate_sql("DELETE FROM account_login_codes WHERE account_id = %s", self.kind),
                (account_id,),
            )
            return True

    # ---- session operations ----

    def create_session(
        self,
        account_id: str,
        token_hash: str,
        ttl_days: int = 30,
        device_label: Optional[str] = None,
    ) -> Session:
        now = _utcnow()
        expires = now + timedelta(days=ttl_days)
        with self._connect() as conn:
            conn.execute(
                _translate_sql(
                    "INSERT INTO account_sessions "
                    "(token_hash, account_id, created_at, expires_at, last_seen_at, device_label) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    self.kind,
                ),
                (token_hash, account_id, now, expires, now, device_label),
            )
        return Session(
            token_hash=token_hash,
            account_id=account_id,
            created_at=now,
            expires_at=expires,
            last_seen_at=now,
        )

    def get_session(self, token_hash: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(
                    "SELECT token_hash, account_id, created_at, expires_at, last_seen_at "
                    "FROM account_sessions WHERE token_hash = %s",
                    self.kind,
                ),
                (token_hash,),
            ).fetchone()
            if not row:
                return None
            sess = Session(
                token_hash=_get(row, 0, "token_hash"),
                account_id=_get(row, 1, "account_id"),
                created_at=_coerce_dt(_get(row, 2, "created_at")),
                expires_at=_coerce_dt(_get(row, 3, "expires_at")),
                last_seen_at=_coerce_dt(_get(row, 4, "last_seen_at")),
            )
            if sess.expires_at < _utcnow():
                return None
            return sess

    def touch_session(self, token_hash: str) -> None:
        """Update last_seen_at. Best-effort — failure is non-fatal."""
        try:
            with self._connect() as conn:
                conn.execute(
                    _translate_sql(
                        "UPDATE account_sessions SET last_seen_at = %s WHERE token_hash = %s",
                        self.kind,
                    ),
                    (_utcnow(), token_hash),
                )
        except Exception:
            pass

    def delete_session(self, token_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                _translate_sql("DELETE FROM account_sessions WHERE token_hash = %s", self.kind),
                (token_hash,),
            )

    # ---- pmc_user_id binding ----

    def claim_pmc_user(self, account_id: str, pmc_user_id: str) -> bool:
        """Bind an anonymous pmc_user_id to an account. Returns True
        if the bind succeeded; False if that pmc_user_id is already
        claimed by a different account (in which case we leave the
        existing binding alone — the user can't take over someone
        else's data by guessing UUIDs)."""
        now = _utcnow()
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(
                    "SELECT account_id FROM account_pmc_users WHERE pmc_user_id = %s",
                    self.kind,
                ),
                (pmc_user_id,),
            ).fetchone()
            if row:
                existing = _get(row, 0, "account_id")
                return existing == account_id
            conn.execute(
                _translate_sql(
                    "INSERT INTO account_pmc_users (account_id, pmc_user_id, claimed_at) "
                    "VALUES (%s, %s, %s)",
                    self.kind,
                ),
                (account_id, pmc_user_id, now),
            )
            return True

    def list_pmc_users(self, account_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                _translate_sql(
                    "SELECT pmc_user_id FROM account_pmc_users WHERE account_id = %s "
                    "ORDER BY claimed_at",
                    self.kind,
                ),
                (account_id,),
            ).fetchall()
            return [_get(r, 0, "pmc_user_id") for r in rows]

    def get_account_for_pmc_user(self, pmc_user_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                _translate_sql(
                    "SELECT account_id FROM account_pmc_users WHERE pmc_user_id = %s",
                    self.kind,
                ),
                (pmc_user_id,),
            ).fetchone()
            return _get(row, 0, "account_id") if row else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _short_id() -> str:
    """Account id. Short, opaque, no PII. ~80 bits of entropy."""
    import secrets
    return secrets.token_urlsafe(12)


def _coerce_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return _utcnow()
    return _utcnow()


def _get(row, idx, key):
    """Read either tuple-style (Postgres) or row-style (SQLite)."""
    if row is None:
        return None
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        try:
            return row[idx]
        except (KeyError, IndexError, TypeError):
            return None


def _row_to_account(row, kind: str) -> Account:
    period_end = _get(row, 6, "subscription_current_period_end")
    return Account(
        id=_get(row, 0, "id"),
        email=_get(row, 1, "email"),
        created_at=_coerce_dt(_get(row, 2, "created_at")),
        stripe_customer_id=_get(row, 3, "stripe_customer_id"),
        subscription_status=_get(row, 4, "subscription_status"),
        subscription_tier=_get(row, 5, "subscription_tier"),
        subscription_current_period_end=_coerce_dt(period_end) if period_end else None,
    )


_ACCOUNT_SELECT = (
    "SELECT id, email, created_at, stripe_customer_id, "
    "subscription_status, subscription_tier, subscription_current_period_end "
    "FROM accounts"
)
