"""Token primitives for the auth flow.

Two distinct tokens:

  * **Magic-link code** — 6 characters, human-typable. The user sees
    it in the email, types it back into the app. Lives ~15 minutes,
    one-time use. We don't try to make it unguessable on a single
    attempt; we rely on per-account rate limits + short expiry +
    the email channel being out-of-band.

  * **Session token** — 32 bytes of CSPRNG-derived randomness,
    base64url-encoded. The Mac app stores this and presents it in
    the Authorization header. We persist its SHA-256 hash, never
    the token itself. ~30 days lifetime, refreshed on use.
"""

from __future__ import annotations

import hashlib
import secrets


# Code char set excludes ambiguous glyphs (0/O, 1/I/L). Easier to
# read in an email, faster to type, no support tickets about "is
# that an O or a zero."
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LEN = 6


def new_code() -> str:
    """Generate a one-time login code formatted as ABC-DEF."""
    chars = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))
    return f"{chars[:3]}-{chars[3:]}"


def normalize_code(raw: str) -> str:
    """Strip dashes / whitespace and uppercase so 'abc def' == 'ABC-DEF'."""
    if not raw:
        return ""
    return "".join(c for c in raw.upper() if c.isalnum())


def new_session_token() -> str:
    """Opaque session token. The plaintext lives on the client; the
    server only stores the hash."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 hex of the token. Used to look up sessions without
    storing the plaintext."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_eq(a: str, b: str) -> bool:
    """Compare two strings without leaking timing. Used when we have
    a plaintext code and a stored plaintext code (we don't hash codes
    because they're short-lived + one-use)."""
    return secrets.compare_digest(a, b)
