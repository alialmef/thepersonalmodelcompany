"use client";

/**
 * Client-side helpers for the PMC backend's /v1/auth/* endpoints.
 *
 * This is the Mac-app path. It's distinct from web/lib/auth.ts —
 * that one is for the Next.js marketing site's server-rendered
 * magic-link flow (Postgres, cookies). This one talks straight to
 * the FastAPI backend over HTTP and stashes the session token in
 * localStorage (V1; Keychain upgrade is V2).
 *
 * Endpoints:
 *   POST /v1/auth/email     → request a one-time code
 *   POST /v1/auth/exchange  → trade (email, code) for a session token
 *   POST /v1/auth/claim     → bind an anonymous pmcUserId to the account
 *   POST /v1/auth/signout   → invalidate the current session
 *   GET  /v1/auth/me        → current account
 */

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

const TOKEN_KEY = "pmc.sessionToken";
const ACCOUNT_KEY = "pmc.account";
const ANON_KEY = "pmc.localUserId";

export interface Account {
  id: string;
  email: string;
  created_at: string;
}

export interface ExchangeResult {
  session_token: string;
  account: Account;
  pmc_user_ids: string[];
}

// ---------------------------------------------------------------------------
// Token + account storage
// ---------------------------------------------------------------------------

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getStoredAccount(): Account | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(ACCOUNT_KEY);
    return raw ? (JSON.parse(raw) as Account) : null;
  } catch {
    return null;
  }
}

export function storeSession(token: string, account: Account): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
    window.localStorage.setItem(ACCOUNT_KEY, JSON.stringify(account));
  } catch {
    /* ignore quota errors */
  }
}

export function clearSession(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(ACCOUNT_KEY);
  } catch {
    /* ignore */
  }
}

export function getAnonymousPmcUserId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(ANON_KEY);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// HTTP calls
// ---------------------------------------------------------------------------

export async function requestEmailCode(email: string): Promise<void> {
  const res = await fetch(`${PMC_API_URL}/v1/auth/email`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `HTTP ${res.status}`);
  }
}

export async function exchangeCode(
  email: string,
  code: string,
): Promise<ExchangeResult> {
  const res = await fetch(`${PMC_API_URL}/v1/auth/exchange`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, code }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as ExchangeResult;
}

/** Bind the localStorage anonymous pmcUserId onto the signed-in
 * account. Returns false silently if there's nothing to claim or
 * the id is already owned by another account. */
export async function claimAnonymousIfAny(token: string): Promise<boolean> {
  const anon = getAnonymousPmcUserId();
  if (!anon) return false;
  try {
    const res = await fetch(`${PMC_API_URL}/v1/auth/claim`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ pmc_user_id: anon }),
    });
    if (!res.ok) return false;
    const body = (await res.json()) as { ok: boolean };
    return body.ok;
  } catch {
    return false;
  }
}

export async function signOut(token: string | null): Promise<void> {
  if (token) {
    try {
      await fetch(`${PMC_API_URL}/v1/auth/signout`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
      });
    } catch {
      /* best-effort */
    }
  }
  clearSession();
}

export async function fetchMe(): Promise<Account | null> {
  const token = getStoredToken();
  if (!token) return null;
  try {
    const res = await fetch(`${PMC_API_URL}/v1/auth/me`, {
      headers: { authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!res.ok) {
      if (res.status === 401) clearSession();
      return null;
    }
    const body = (await res.json()) as { account: Account };
    return body.account;
  } catch {
    return null;
  }
}

/** Wrap a fetch init with an Authorization header when a session
 * token is present. Callers that already pass headers should still
 * use this — we merge correctly. */
export function withAuth(init: RequestInit = {}): RequestInit {
  const token = getStoredToken();
  if (!token) return init;
  return {
    ...init,
    headers: {
      ...(init.headers ?? {}),
      authorization: `Bearer ${token}`,
    },
  };
}
