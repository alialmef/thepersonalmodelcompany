"use client";

import { useEffect, useState } from "react";

import {
  fetchMe,
  getAnonymousPmcUserId,
  getStoredAccount,
  getStoredToken,
  signOut as signOutApi,
} from "@/lib/api/auth";

/**
 * `useUser()` — current account + anonymous-derived pmcUserId.
 *
 * Reads in this priority:
 *   1. Session token in localStorage → fetch /v1/auth/me, return
 *      that account; pmcUserId comes from the anonymous identity
 *      already on disk (claimed onto the account at sign-in time).
 *   2. No session → web's /api/auth/me (cookie-based) for browser
 *      users on the marketing site.
 *   3. No session, Tauri runtime → local anonymous UUID fallback
 *      so the existing flow keeps working through the migration
 *      window. This will be removed in V2 when sign-in is required.
 *
 * Cached at the module level so re-mounts within the same session
 * don't re-fetch.
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
  // 1. Authenticated session token (Mac app or web user who signed
  //    in via the FastAPI /v1/auth/* flow).
  const storedToken = getStoredToken();
  if (storedToken) {
    const storedAcct = getStoredAccount();
    if (storedAcct) {
      // Show the cached account immediately so the UI doesn't flicker.
      CACHED = {
        id: storedAcct.id,
        email: storedAcct.email,
        pmcUserId: getAnonymousPmcUserId() ?? storedAcct.id,
      };
    }
    // Validate against the backend; if it 401s, fetchMe clears the
    // stored session, and we fall through to the anon path.
    const live = await fetchMe();
    if (live) {
      CACHED = {
        id: live.id,
        email: live.email,
        pmcUserId: getAnonymousPmcUserId() ?? live.id,
      };
      CACHED_FETCHED = true;
      return;
    }
    // Token rejected — clear cache + fall through.
    CACHED = null;
  }

  // 2. Web cookie session (marketing site).
  try {
    const res = await fetch("/api/auth/me", { cache: "no-store" });
    if (res.ok) {
      CACHED = (await res.json()) as SessionUser;
      CACHED_FETCHED = true;
      return;
    }
  } catch {
    /* not on the web — fall through */
  }

  // 3. Tauri anon fallback (transition path; to be removed in V2).
  if (isTauriRuntime()) {
    CACHED = localAnonymousUser();
  } else {
    CACHED = null;
  }
  CACHED_FETCHED = true;
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

/** Sign out: clears server-side session, local cache, and storage. */
export async function signOutUser(): Promise<void> {
  const token = getStoredToken();
  await signOutApi(token);
  clearUserCache();
}
