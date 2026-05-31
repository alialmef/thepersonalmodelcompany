"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { BrandMark } from "@/components/shared/brand-mark";
import MemoryWeb from "@/components/app/memory-web";
import { useUser } from "@/hooks/use-user";
import { getConfig as getAgentConfig } from "@/lib/api/agent";
import {
  imessageStatus,
  isTauri,
  openFullDiskAccessSettings,
} from "@/lib/runtime";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * /reading — the moment.
 *
 * On mount, this page does three things in order:
 *
 *   1. Probes iMessage's chat.db via the imessage_status Tauri
 *      command. This is what *actively triggers* the macOS Full Disk
 *      Access TCC dialog — the OS only prompts when an entitled
 *      binary actually attempts to read a gated path. Without this
 *      explicit poke, the background scheduler would just silently
 *      hit PermissionDenied per source and write nothing, leaving
 *      the user staring at zero counts forever.
 *
 *   2. If FDA was granted (chat.db is readable), fires graph_kickoff
 *      which schedules every extractor in the background.
 *
 *   3. If FDA was denied or never asked, surfaces a clear "Grant
 *      Full Disk Access" affordance that deep-links into the right
 *      System Settings panel.
 *
 * Polls /v1/users/{id}/status every 3s and surfaces typed-prose
 * per-source counts as they come in. Auto-advances to /right-now
 * when totals are steady for ~5 polls + a minimum dwell.
 */

interface SourceBreakdown {
  source_id: string;
  kind: string;
  item_count: number;
}

interface UserStatus {
  raw_item_count?: number;
  raw_source_breakdown?: SourceBreakdown[];
}

type FdaState =
  | "unknown"        // haven't probed yet
  | "granted"        // probe returned readable
  | "needs_grant"    // probe returned permission_denied
  | "not_present"    // no chat.db on this Mac (skip the prompt)
  | "not_tauri";     // running in a browser, not the Mac app

const POLL_MS = 3_000;
const STEADY_AFTER_POLLS = 5;
const MIN_DWELL_MS = 6_000;

export default function ReadingPage() {
  const router = useRouter();
  const { user } = useUser();
  const [sources, setSources] = useState<SourceBreakdown[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [steady, setSteady] = useState(false);
  const [fda, setFda] = useState<FdaState>("unknown");
  const lastTotalRef = useRef<number>(-1);
  const steadyCountRef = useRef<number>(0);
  const mountedAt = useRef<number>(Date.now());

  // Step 0: gate on having an agent configured. The structuring pass
  // depends on the user's chosen frontier model running entity
  // resolution + theme detection. No agent → bounce to /settings/agent.
  useEffect(() => {
    if (!user?.pmcUserId) return;
    let cancelled = false;
    (async () => {
      try {
        const cfg = await getAgentConfig();
        if (cancelled) return;
        if (!cfg.configured) {
          router.replace("/settings/agent");
        }
      } catch {
        // If we can't reach the backend, don't bounce — let the user see
        // the FDA / reading state and figure it out.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user?.pmcUserId, router]);

  // Step 1: probe FDA + (if granted) fire graph_kickoff. Order matters —
  // the imessage_status probe is what makes macOS pop the TCC dialog.
  useEffect(() => {
    if (!user?.pmcUserId) return;
    if (!isTauri()) {
      setFda("not_tauri");
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const status = await imessageStatus();
        if (cancelled) return;
        if (!status.chat_db_exists) {
          setFda("not_present");
        } else if (status.can_read) {
          setFda("granted");
        } else {
          setFda("needs_grant");
          return; // don't kick off the scheduler — every extractor would PermissionDenied
        }
        // FDA OK (or chat.db not present — try the rest anyway). Fire
        // the scheduler.
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("graph_kickoff", { userId: user.pmcUserId });
      } catch {
        // If imessage_status itself errors out (rare), still let the
        // scheduler try — some extractors don't need FDA at all.
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("graph_kickoff", { userId: user.pmcUserId });
        } catch {
          /* nothing more we can do */
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user?.pmcUserId]);

  // Step 2: poll status (per-source breakdown). Runs regardless of FDA
  // state — if data does start coming in we want to surface it.
  useEffect(() => {
    if (!user?.pmcUserId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const r = await fetch(
          `${PMC_API_URL}/v1/users/${encodeURIComponent(user.pmcUserId)}/status`,
          { cache: "no-store" },
        );
        if (!r.ok || cancelled) return;
        const data = (await r.json()) as UserStatus;
        const t = data.raw_item_count ?? 0;
        setTotal(t);
        setSources(data.raw_source_breakdown ?? []);

        if (t === lastTotalRef.current && t > 0) {
          steadyCountRef.current += 1;
        } else {
          steadyCountRef.current = 0;
        }
        lastTotalRef.current = t;
        if (
          steadyCountRef.current >= STEADY_AFTER_POLLS &&
          Date.now() - mountedAt.current >= MIN_DWELL_MS
        ) {
          setSteady(true);
        }
      } catch {
        /* offline — keep polling */
      }
    };

    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [user?.pmcUserId]);

  // When the web reports stable (no new nodes for ~6 polls), advance
  // the user to /confirm — the validation pass before /right-now.
  const [webStable, setWebStable] = useState(false);
  useEffect(() => {
    if (!steady && !webStable) return;
    const t = setTimeout(() => router.push("/confirm"), 1400);
    return () => clearTimeout(t);
  }, [steady, webStable, router]);

  // Re-probe after the user comes back from System Settings.
  async function recheckFda() {
    if (!user?.pmcUserId || !isTauri()) return;
    const status = await imessageStatus();
    if (status.can_read) {
      setFda("granted");
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("graph_kickoff", { userId: user.pmcUserId });
    } else {
      setFda("needs_grant");
    }
  }

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-8 pb-16 pt-16">
        <div className="mb-12">
          <BrandMark />
        </div>

        {fda === "needs_grant" ? (
          <div className="mt-16">
            <FdaPrompt onGranted={recheckFda} />
          </div>
        ) : (
          <>
            {/* The web. Centered, full-bleed within the column. */}
            <div className="flex-1 flex items-center justify-center">
              <MemoryWeb
                userId={user?.pmcUserId ?? ""}
                className="w-full max-w-3xl aspect-[3/2]"
                onStable={() => setWebStable(true)}
              />
            </div>

            {/* Caption underneath the web. Calm, brief. */}
            <div className="text-center space-y-1 mt-8">
              <div className="text-base text-foreground/55">
                {webStable
                  ? "Done."
                  : total > 0
                  ? "Reading."
                  : fda === "granted"
                  ? "Opening your sources…"
                  : "Starting up…"}
              </div>
              {total > 0 && (
                <div className="text-xs text-foreground/30 font-mono">
                  {total.toLocaleString()} items
                </div>
              )}
            </div>
          </>
        )}

        <div className="mt-12 flex justify-center">
          <button
            type="button"
            onClick={() => router.push("/confirm")}
            className="text-sm text-foreground/45 hover:text-foreground/75 transition-colors"
          >
            Continue
          </button>
        </div>
      </div>
    </main>
  );
}

function FdaPrompt({ onGranted }: { onGranted: () => void | Promise<void> }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div className="text-xl font-semibold text-foreground">
          One permission first.
        </div>
        <div className="text-base text-foreground/55">
          PMC needs Full Disk Access to read your messages, mail,
          calendar, Screen Time, and the rest. macOS gates this — you
          have to flip the switch yourself.
        </div>
        <div className="text-sm text-foreground/45">
          Click below to open the right Settings panel, toggle
          <span className="font-mono"> Personal Model Company</span> on,
          come back here, and tap <span className="text-foreground/65">I&apos;ve granted access</span>.
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-8">
        <button
          type="button"
          onClick={async () => {
            await openFullDiskAccessSettings();
          }}
          className="text-base text-foreground/85 hover:text-foreground"
        >
          Open Full Disk Access settings
        </button>
        <button
          type="button"
          onClick={async () => {
            setBusy(true);
            try {
              await onGranted();
            } finally {
              setBusy(false);
            }
          }}
          disabled={busy}
          className="text-base text-foreground/55 hover:text-foreground/85 disabled:cursor-default disabled:text-foreground/25"
        >
          {busy ? "Checking…" : "I've granted access"}
        </button>
      </div>
    </div>
  );
}
