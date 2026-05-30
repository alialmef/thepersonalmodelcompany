"""PMC billing — Stripe subscription glue.

One product, one price: the Frontier subscription. The user signs in,
hits /pay, and Stripe Checkout collects payment to PMC's Stripe account.
Webhooks update the `accounts` row, and `POST /v1/users/{id}/runs` is
gated on `account.is_subscribed() OR is_founder`.

Module shape:
  service.py   — BillingService: ensure_customer, create_checkout_session,
                 handle_webhook. All Stripe SDK calls live here.
  router.py    — FastAPI routes mounted at /v1/billing/*.

Configuration via env (read at request time, not import time, so tests
can monkeypatch without restarting):
  STRIPE_SECRET_KEY              required to take payment
  STRIPE_WEBHOOK_SIGNING_SECRET  required to verify webhooks
  STRIPE_PRICE_ID                the Frontier monthly price (price_...)
  PMC_WEB_URL                    where Stripe Checkout returns the user
                                 (defaults to PMC's production domain)
"""

from pmc.billing.service import (
    BillingError,
    BillingService,
    BillingNotConfigured,
    CheckoutSession,
    SubscriptionState,
)

__all__ = [
    "BillingError",
    "BillingNotConfigured",
    "BillingService",
    "CheckoutSession",
    "SubscriptionState",
]
