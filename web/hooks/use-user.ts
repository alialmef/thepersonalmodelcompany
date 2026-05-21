"use client";

import { useEffect, useState } from "react";

/**
 * `useUser()` — fetches the current signed-in user from /api/auth/me.
 *
 * Replaces every reference to DEMO_USER_ID in client components. The hook
 * returns:
 *   - { user: null, status: "loading" }    — initial fetch in flight
 *   - { user: User,  status: "authed" }    — signed in
 *   - { user: null, status: "anon" }       — not signed in (the middleware
 *                                            should have redirected us already
 *                                            for protected routes; this is the
 *                                            transient state during HMR)
 *
 * Cached at the module level so re-mounts within the same session don't
 * re-fetch. If we ever need cross-tab invalidation, add a BroadcastChannel.
 */

export interface SessionUser {
  id: string;
  email: string;
  pmcUserId: string;
}

type Status = "loading" | "authed" | "anon";

let CACHED: SessionUser | null = null;
let CACHED_FETCHED = false;
let inFlight: Promise<void> | null = null;

/**
 * The Mac app has no server-rendered session (the static export strips
 * /api/auth/*), so /api/auth/me will 404. In that case we fall back to a
 * stable, locally-stored anonymous identity. The backend creates users
 * lazily on first POST, so an arbitrary ID is fine.
 */
function localAnonymousUser(): SessionUser {
  if (typeof window === "undefined") {
    return { id: "anon", email: "anon@local", pmcUserId: "anon" };
  }
  const KEY = "pmc.localUserId";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `local-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
    window.localStorage.setItem(KEY, id);
  }
  return { id, email: `${id}@local`, pmcUserId: id };
}

function isTauriRuntime(): boolean {
  if (typeof window === "undefined") return false;
  const w = window as unknown as Record<string, unknown>;
  return !!(
    w.__TAURI_INTERNALS__ ||
    w.__TAURI__ ||
    w.__TAURI_METADATA__ ||
    w.isTauri
  );
}

async function fetchUser(): Promise<void> {
  try {
    const res = await fetch("/api/auth/me", { cache: "no-store" });
    if (res.ok) {
      CACHED = (await res.json()) as SessionUser;
    } else if (isTauriRuntime()) {
      CACHED = localAnonymousUser();
    } else {
      CACHED = null;
    }
  } catch {
    CACHED = isTauriRuntime() ? localAnonymousUser() : null;
  } finally {
    CACHED_FETCHED = true;
  }
}

export function useUser(): { user: SessionUser | null; status: Status } {
  const [, force] = useState(0);
  const [status, setStatus] = useState<Status>(
    CACHED_FETCHED ? (CACHED ? "authed" : "anon") : "loading",
  );

  useEffect(() => {
    if (CACHED_FETCHED) return;
    if (!inFlight) {
      inFlight = fetchUser().finally(() => {
        inFlight = null;
      });
    }
    inFlight.then(() => {
      setStatus(CACHED ? "authed" : "anon");
      force((n) => n + 1);
    });
  }, []);

  return { user: CACHED, status };
}

/** Invalidate the cached user — call after sign-out to force re-fetch. */
export function clearUserCache(): void {
  CACHED = null;
  CACHED_FETCHED = false;
}
