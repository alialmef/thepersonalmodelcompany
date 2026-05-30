"""At-rest encryption for user-supplied API keys.

User-supplied API keys (Anthropic / OpenAI / Google / OpenRouter) get
stored in Postgres in encrypted form. The key wrapping the user keys
is derived from a single server-side secret (`PMC_KEY_ENCRYPTION_SECRET`)
so the encrypted blobs at rest cannot be decrypted with just DB access.

Implementation: Fernet (AES-128-CBC + HMAC-SHA256, both authenticated).
The wrapping key is derived via PBKDF2-HMAC-SHA256 from the env secret
so operators can pick any reasonably-strong passphrase rather than
needing to manage a raw 32-byte key.

If `PMC_KEY_ENCRYPTION_SECRET` isn't set the helpers raise — we'd rather
the Settings API loudly fail to store a key than silently store it in
plaintext. Operators set the secret once on the deploy; rotating it
invalidates every stored key (users re-paste).
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache


class EncryptionNotConfigured(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _fernet():
    try:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]
        from cryptography.hazmat.primitives import hashes  # type: ignore[import-untyped]
        from cryptography.hazmat.primitives.kdf.pbkdf2 import (  # type: ignore[import-untyped]
            PBKDF2HMAC,
        )
    except ImportError as e:
        raise EncryptionNotConfigured(
            "cryptography package not installed",
        ) from e

    secret = os.environ.get("PMC_KEY_ENCRYPTION_SECRET", "").strip()
    if not secret:
        raise EncryptionNotConfigured(
            "PMC_KEY_ENCRYPTION_SECRET not set — refusing to store user API keys",
        )
    # Static salt is acceptable here: there's exactly one PMC-wide
    # wrapping key, derived once per deploy. The salt's job (preventing
    # rainbow-table attacks on a user-controllable password) doesn't
    # apply to an operator-controlled deploy secret.
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"pmc-byom-keys-v1",
        iterations=600_000,
    )
    raw = kdf.derive(secret.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(raw))


def encrypt(plaintext: str) -> str:
    """Encrypt a user API key. Returns a URL-safe base64 ciphertext."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str) -> str:
    """Decrypt a previously-encrypted user API key."""
    return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")


def is_configured() -> bool:
    """Cheap probe for the Settings API to surface a clear 503 when
    encryption isn't set up on the deploy."""
    try:
        _fernet()
        return True
    except EncryptionNotConfigured:
        return False
