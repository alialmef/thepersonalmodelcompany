"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import FirstLaunchScreen from "@/components/app/first-launch-screen";
import { useUser } from "@/hooks/use-user";
import { isTauri } from "@/lib/runtime";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * Screen 1 — First Launch.
 *
 * The Mac app opens here. Pure white. Three seconds of held silence
 * between "Hello." and the second line. Do not shorten.
 *
 * On Begin → /connect (Step 1 of 3).
 *
 * Returning users get a faint "Start over" affordance in the corner.
 * It only renders for users who already have data on disk; first-launch
 * users never see it. Clicking it calls reset_user (Tauri) or the
 * backend reset endpoint, clears the local pmcUserId, then reloads.
 */
export default function WelcomePage() {
  const router = useRouter();
  const { user } = useUser();
  const userId = user?.pmcUserId;

  const [hasExistingData, setHasExistingData] = useState(false);
  const [resetting, setResetting] = useState(false);

  // Only show "Start over" if the user actually has data attached to
  // their id on the backend. Avoids the affordance leaking to first-
  // launch users who haven't connected anything.
  useEffect(() => {
    if (!userId) return;
    let cancelled = false;
    fetch(`${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/status`, {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    })
      .then(async (res) => {
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as { raw_item_count?: number };
        if (!cancelled && (data.raw_item_count ?? 0) > 0) {
          setHasExistingData(true);
        }
      })
      .catch(() => {
        /* offline or no user — leave the affordance hidden */
      });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  const handleStartOver = async () => {
    if (!userId || resetting) return;
    if (!window.confirm(
      "Wipe everything for this user — your data, your memory, your trained model — and start over?\n\nThis cannot be undone.",
    )) return;

    setResetting(true);
    try {
      if (isTauri()) {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("reset_user", { userId });
      } else {
        await fetch(
          `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/reset`,
          { method: "POST" },
        );
      }
    } catch (e) {
      console.warn("reset_user failed", e);
    }
    // Clear localStorage so the user comes through as fresh next time.
    try { localStorage.removeItem("pmc-user"); } catch { /* private mode */ }
    setResetting(false);
    // Reload to land on /welcome with a clean slate.
    window.location.replace("/welcome");
  };

  // Where Begin goes:
  //   - if the user is already signed in OR has a session token,
  //     skip /sign-in and go straight to /connect
  //   - otherwise route to /sign-in first
  const beginTarget = (() => {
    try {
      if (typeof window === "undefined") return "/sign-in";
      const token = window.localStorage.getItem("pmc.sessionToken");
      return token ? "/connect" : "/sign-in";
    } catch {
      return "/sign-in";
    }
  })();

  return (
    <>
      <FirstLaunchScreen onBegin={() => router.push(beginTarget)} />
      {hasExistingData && (
        <button
          type="button"
          onClick={handleStartOver}
          disabled={resetting}
          className="fixed bottom-6 right-6 text-[11px] uppercase tracking-wider text-neutral-300 hover:text-neutral-500 transition-colors disabled:opacity-50"
          style={{ animationDelay: "8.5s" }}
        >
          {resetting ? "Resetting…" : "Start over"}
        </button>
      )}
    </>
  );
}
