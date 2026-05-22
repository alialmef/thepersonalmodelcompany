"""Magic-link delivery via Resend.

If `RESEND_API_KEY` isn't set in the env, we degrade to a *console
sender* that prints the code to stdout. This is fine in dev (you grab
the code from the server log) and gives us a way to ship the auth
flow without blocking on signing up with Resend. In production the
key should be set on Railway.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional


log = logging.getLogger(__name__)


DEFAULT_FROM_ADDRESS = "PMC <signin@thepersonalmodelcompany.com>"
DEFAULT_REPLY_TO = "ali@thepersonalmodelcompany.com"


@dataclass
class EmailResult:
    delivered: bool
    method: str            # 'resend' | 'console' | 'noop'
    detail: Optional[str] = None


def send_login_code(
    email: str,
    code: str,
    *,
    api_key: Optional[str] = None,
    from_address: str = DEFAULT_FROM_ADDRESS,
    reply_to: Optional[str] = DEFAULT_REPLY_TO,
) -> EmailResult:
    """Deliver the login code. Returns EmailResult so the router can
    surface failures honestly without leaking which addresses exist."""
    key = api_key or os.environ.get("RESEND_API_KEY")
    if not key:
        # Dev fallback — log so we can grab the code from server output.
        log.warning("[auth] RESEND_API_KEY not set — logging code to stdout")
        print(f"[auth] login code for {email}: {code}", flush=True)
        return EmailResult(delivered=True, method="console")

    try:
        import resend
        resend.api_key = key
        resend.Emails.send({
            "from":     from_address,
            "to":       [email],
            "reply_to": reply_to,
            "subject":  "Your sign-in code",
            "text":     _plain_text(code),
            "html":     _html(code),
        })
        return EmailResult(delivered=True, method="resend")
    except ImportError:
        log.warning("[auth] resend package not installed — logging code")
        print(f"[auth] login code for {email}: {code}", flush=True)
        return EmailResult(delivered=True, method="console",
                           detail="resend package missing")
    except Exception as e:
        log.exception("[auth] Resend send failed: %s", e)
        return EmailResult(delivered=False, method="resend", detail=str(e))


def _plain_text(code: str) -> str:
    # Body intentionally short. The brand voice is institutional and
    # declarative — this isn't a marketing email.
    return (
        f"Your sign-in code:\n\n"
        f"    {code}\n\n"
        f"It works once and expires in 15 minutes.\n"
        f"\n"
        f"If you didn't ask for this, ignore the email.\n"
        f"\n"
        f"The Personal Model Company\n"
    )


def _html(code: str) -> str:
    return f"""\
<!doctype html>
<html>
  <body style="font-family:ui-sans-serif,system-ui,-apple-system; line-height:1.55; color:#1a1a1a; padding:32px;">
    <p style="margin:0 0 24px;">Your sign-in code:</p>
    <p style="font-family:ui-monospace,SFMono-Regular,monospace; font-size:22px; letter-spacing:0.15em; margin:0 0 32px;">{code}</p>
    <p style="color:#6b6b6b; margin:0 0 8px;">It works once and expires in 15 minutes.</p>
    <p style="color:#6b6b6b; margin:0 0 32px;">If you didn't ask for this, ignore the email.</p>
    <p style="color:#aaa; margin:0; font-size:13px;">The Personal Model Company</p>
  </body>
</html>
"""
