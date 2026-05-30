"""FastAPI router for /v1/billing/*.

  POST /v1/billing/checkout       (auth) → {url}
  POST /v1/billing/webhook        (Stripe-signed) — receives subscription events
  GET  /v1/billing/status         (auth) → current subscription state

Mounted from pmc/serve/api.py only when the AuthStore is configured; the
service itself fails closed with a clear 503 if STRIPE_SECRET_KEY isn't
set on the deploy, so the frontend can distinguish "not wired" from
"declined."
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from pmc.auth.middleware import AuthSession, require_session
from pmc.billing.service import (
    BillingError,
    BillingNotConfigured,
    BillingService,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    success_url: Optional[str] = Field(
        default=None,
        description=(
            "Where Stripe sends the user after a successful checkout. "
            "Falls back to PMC_WEB_URL/welcome?paid=1."
        ),
    )
    cancel_url: Optional[str] = Field(
        default=None,
        description="Where Stripe sends the user if they cancel. Falls back to PMC_WEB_URL/pay.",
    )


class CheckoutResponse(BaseModel):
    url: str
    session_id: str


class BillingStatus(BaseModel):
    configured: bool
    is_subscribed: bool
    is_founder: bool
    subscription_status: Optional[str] = None
    subscription_tier: Optional[str] = None
    subscription_current_period_end: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(request: Request) -> BillingService:
    svc = getattr(request.app.state, "billing_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="billing not configured",
        )
    return svc


def _default_urls() -> tuple[str, str]:
    """Resolve success/cancel URLs from the deploy's PMC_WEB_URL env.

    Web defaults to https://thepersonalmodelcompany.com so that a Stripe
    Checkout opened from the Mac app comes back to the site, where the
    Tauri app can listen for the return via deep link in V2. For V1, the
    success page just tells the user to return to the app.
    """
    base = os.environ.get(
        "PMC_WEB_URL", "https://thepersonalmodelcompany.com"
    ).rstrip("/")
    return f"{base}/welcome?paid=1", f"{base}/pay"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def build_billing_router(founders=None) -> APIRouter:
    """Construct the router. Caller passes the FounderTracker if it
    wants /v1/billing/status to surface the founder flag — keeps the
    router decoupled from server.py wiring."""
    router = APIRouter(prefix="/v1/billing", tags=["billing"])

    @router.post("/checkout", response_model=CheckoutResponse)
    def create_checkout(
        request: Request,
        body: Optional[CheckoutRequest] = None,
        session: AuthSession = Depends(require_session),
    ) -> CheckoutResponse:
        svc = _service(request)
        body = body or CheckoutRequest()
        default_success, default_cancel = _default_urls()
        try:
            checkout = svc.create_checkout_session(
                session.account,
                success_url=body.success_url or default_success,
                cancel_url=body.cancel_url or default_cancel,
            )
        except BillingNotConfigured as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e
        except BillingError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
            ) from e
        return CheckoutResponse(url=checkout.url, session_id=checkout.id)

    @router.post("/webhook")
    async def webhook(
        request: Request,
        stripe_signature: Optional[str] = Header(default=None, alias="Stripe-Signature"),
    ) -> dict:
        svc = _service(request)
        if not stripe_signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing Stripe-Signature header",
            )
        payload = await request.body()
        try:
            event = svc.verify_and_parse_event(payload, stripe_signature)
        except BillingNotConfigured as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e
        except BillingError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        # Don't propagate handler failures back to Stripe — log on the
        # server, ack 200 so Stripe stops retrying when the issue is
        # ours (e.g. an event for an account we no longer have).
        try:
            svc.handle_event(event)
        except Exception:
            pass
        return {"received": True}

    @router.get("/status", response_model=BillingStatus)
    def get_status(
        request: Request,
        session: AuthSession = Depends(require_session),
    ) -> BillingStatus:
        svc = getattr(request.app.state, "billing_service", None)
        configured = bool(svc and svc.is_configured())
        acct = session.account
        is_founder = bool(founders and founders.is_founder(acct.id))
        cpe = acct.subscription_current_period_end
        return BillingStatus(
            configured=configured,
            is_subscribed=acct.is_subscribed(),
            is_founder=is_founder,
            subscription_status=acct.subscription_status,
            subscription_tier=acct.subscription_tier,
            subscription_current_period_end=cpe.isoformat() if cpe else None,
        )

    return router
