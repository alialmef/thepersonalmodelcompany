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

async function fetchUser(): Promise<void> {
  try {
    const res = await fetch("/api/auth/me", { cache: "no-store" });
    if (res.ok) {
      CACHED = (await res.json()) as SessionUser;
    } else {
      CACHED = null;
    }
  } catch {
    CACHED = null;
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
