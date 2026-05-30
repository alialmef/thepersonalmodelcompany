"""Stripe wrapper. The single file that imports the stripe SDK.

The rest of PMC talks to billing through `BillingService` so we never
sprinkle stripe.* calls across handlers, tests can swap a fake without
touching network, and the SDK import is lazy (Railway boots faster when
billing isn't exercised at startup).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from pmc.auth.store import Account, AuthStore


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BillingError(RuntimeError):
    """Base class for billing failures (network, signature, schema)."""


class BillingNotConfigured(BillingError):
    """Raised when an env var the caller needs isn't set. The router
    converts this into a clear 503 so the frontend can tell the user the
    payment system isn't wired yet — different from "the payment failed."
    """


# ---------------------------------------------------------------------------
# Plain dataclasses returned to callers — no stripe types leak out
# ---------------------------------------------------------------------------


@dataclass
class CheckoutSession:
    id: str
    url: str


@dataclass
class SubscriptionState:
    """What we extract from a Stripe subscription event. None of these
    are required individually — the webhook handler decides what to
    persist based on the event type."""

    account_id: str
    status: str
    tier: str
    current_period_end: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


# The single product we sell. The price id maps to "Frontier monthly".
# Kept as a constant so frontend code can ask `GET /v1/billing/status`
# and learn the tier label without re-deriving it from a Stripe lookup.
DEFAULT_TIER = "frontier"


class BillingService:
    """Owns all Stripe SDK calls. Construct once at app startup, pass
    the AuthStore so we can persist customer ids + subscription state.
    """

    def __init__(self, auth_store: AuthStore) -> None:
        self.auth_store = auth_store
        self._stripe = None  # lazy

    # ---- configuration --------------------------------------------------

    def _require_secret_key(self) -> str:
        key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
        if not key:
            raise BillingNotConfigured("STRIPE_SECRET_KEY not set")
        return key

    def _require_price_id(self) -> str:
        price = os.environ.get("STRIPE_PRICE_ID", "").strip()
        if not price:
            raise BillingNotConfigured("STRIPE_PRICE_ID not set")
        return price

    def _require_webhook_secret(self) -> str:
        secret = os.environ.get("STRIPE_WEBHOOK_SIGNING_SECRET", "").strip()
        if not secret:
            raise BillingNotConfigured(
                "STRIPE_WEBHOOK_SIGNING_SECRET not set",
            )
        return secret

    def _client(self):
        if self._stripe is None:
            try:
                import stripe  # type: ignore[import-untyped]
            except ImportError as e:
                raise BillingNotConfigured(
                    "stripe package not installed",
                ) from e
            stripe.api_key = self._require_secret_key()
            self._stripe = stripe
        return self._stripe

    def is_configured(self) -> bool:
        """Cheap probe used by /v1/auth/me + /v1/billing/status so the
        UI can show "Payment unavailable" if the operator hasn't set
        STRIPE_SECRET_KEY yet, without throwing on every request."""
        return bool(
            os.environ.get("STRIPE_SECRET_KEY", "").strip()
            and os.environ.get("STRIPE_PRICE_ID", "").strip()
        )

    # ---- customers ------------------------------------------------------

    def ensure_customer(self, account: Account) -> str:
        """Return the Stripe customer id for this account, creating one
        if needed. Persists the id back onto the account row so future
        webhooks can resolve account from customer."""
        if account.stripe_customer_id:
            return account.stripe_customer_id

        stripe = self._client()
        customer = stripe.Customer.create(
            email=account.email,
            metadata={"pmc_account_id": account.id},
        )
        self.auth_store.set_stripe_customer_id(account.id, customer.id)
        return customer.id

    # ---- checkout -------------------------------------------------------

    def create_checkout_session(
        self,
        account: Account,
        *,
        success_url: str,
        cancel_url: str,
    ) -> CheckoutSession:
        """Create a Stripe Checkout session for the Frontier subscription.

        The pmc_account_id is set in subscription metadata so future
        webhooks can resolve back to our account row without relying on
        the customer-id lookup (belt and suspenders — both are reliable).
        """
        stripe = self._client()
        price_id = self._require_price_id()
        customer_id = self.ensure_customer(account)

        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            subscription_data={
                "metadata": {
                    "pmc_account_id": account.id,
                    "pmc_tier": DEFAULT_TIER,
                },
            },
            metadata={"pmc_account_id": account.id},
            allow_promotion_codes=True,
        )
        return CheckoutSession(id=session.id, url=session.url)

    # ---- webhook --------------------------------------------------------

    def verify_and_parse_event(self, payload: bytes, signature: str) -> Any:
        """Stripe signature check. Returns the parsed event dict."""
        stripe = self._client()
        secret = self._require_webhook_secret()
        try:
            event = stripe.Webhook.construct_event(payload, signature, secret)
        except Exception as e:  # SignatureVerificationError, ValueError, …
            raise BillingError(f"webhook signature invalid: {e}") from e
        return event

    def handle_event(self, event: Any) -> Optional[SubscriptionState]:
        """Update account state from a Stripe event. Returns the state we
        wrote (for logging + tests), or None if the event wasn't one we
        care about.

        Events we handle:
          customer.subscription.created   — new sub
          customer.subscription.updated   — renewal / status change
          customer.subscription.deleted   — canceled
        Other events are silently acked so Stripe stops retrying.
        """
        kind = event.get("type", "")
        if not kind.startswith("customer.subscription."):
            return None

        obj = event.get("data", {}).get("object", {}) or {}
        state = self._state_from_subscription(obj)
        if state is None:
            return None

        # If subscription was deleted, mark canceled regardless of the
        # status field (Stripe sometimes carries 'active' on the final
        # event payload for deleted subs).
        if kind == "customer.subscription.deleted":
            self.auth_store.update_subscription_state(
                state.account_id,
                status="canceled",
                tier=state.tier,
                current_period_end=state.current_period_end,
            )
            return SubscriptionState(
                account_id=state.account_id,
                status="canceled",
                tier=state.tier,
                current_period_end=state.current_period_end,
            )

        self.auth_store.update_subscription_state(
            state.account_id,
            status=state.status,
            tier=state.tier,
            current_period_end=state.current_period_end,
        )
        return state

    # ---- helpers --------------------------------------------------------

    def _state_from_subscription(self, sub: dict) -> Optional[SubscriptionState]:
        """Pull what we persist out of a Stripe subscription object.

        Tries metadata.pmc_account_id first (set when we created the
        checkout session), falls back to looking up by customer id.
        """
        metadata = sub.get("metadata") or {}
        account_id = metadata.get("pmc_account_id")
        tier = metadata.get("pmc_tier", DEFAULT_TIER)

        if not account_id:
            customer_id = sub.get("customer")
            if not customer_id:
                return None
            account = self.auth_store.get_account_by_stripe_customer(customer_id)
            if account is None:
                return None
            account_id = account.id

        status = sub.get("status") or "incomplete"
        cpe_unix = sub.get("current_period_end")
        cpe = (
            datetime.fromtimestamp(cpe_unix, tz=timezone.utc)
            if isinstance(cpe_unix, (int, float))
            else None
        )
        return SubscriptionState(
            account_id=account_id,
            status=status,
            tier=tier,
            current_period_end=cpe,
        )
