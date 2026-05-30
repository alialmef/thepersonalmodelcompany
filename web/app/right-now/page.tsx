"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { BrandMark } from "@/components/shared/brand-mark";
import { useUser } from "@/hooks/use-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * /right-now — Phase 1.1 terminal placeholder.
 *
 * After ingest completes the user lands here. The actual product —
 * sub-agent swarm that reads the structured graph and surfaces the
 * single most-pressing thing right now — is the next phase of the
 * build. For now this page acknowledges the user is set up and
 * shows a small honest count of what's been structured so they can
 * see the system did something real.
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

export default function RightNowPage() {
  const { user } = useUser();
  const [status, setStatus] = useState<UserStatus | null>(null);

  useEffect(() => {
    if (!user?.pmcUserId) return;
    let cancelled = false;
    const load = async () => {
      try {
        const r = await fetch(
          `${PMC_API_URL}/v1/users/${encodeURIComponent(user.pmcUserId)}/status`,
          { cache: "no-store" },
        );
        if (!r.ok || cancelled) return;
        const data = (await r.json()) as UserStatus;
        if (!cancelled) setStatus(data);
      } catch {
        /* offline */
      }
    };
    load();
    const t = setInterval(load, 15_000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [user?.pmcUserId]);

  const total = status?.raw_item_count ?? 0;
  const sources = status?.raw_source_breakdown ?? [];

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <div className="mt-16 space-y-12 text-foreground/85">
          <div className="space-y-3">
            <div className="text-xl font-semibold text-foreground">
              Your data is structured.
            </div>
            <div className="text-base text-foreground/55">
              {total > 0
                ? `${total.toLocaleString()} items across ${sources.length} ${sources.length === 1 ? "source" : "sources"}.`
                : "Reading is underway."}
            </div>
          </div>

          {sources.length > 0 && (
            <div className="space-y-1.5 text-[15px] text-foreground/70">
              {sources
                .slice()
                .sort((a, b) => (b.item_count ?? 0) - (a.item_count ?? 0))
                .slice(0, 10)
                .map((s) => (
                  <div
                    key={s.source_id}
                    className="flex items-baseline justify-between gap-4"
                  >
                    <span className="text-foreground/60">{s.kind}</span>
                    <span className="font-mono text-foreground/45 text-[13px]">
                      {s.item_count.toLocaleString()}
                    </span>
                  </div>
                ))}
            </div>
          )}

          <div className="space-y-3 text-foreground/55">
            <div>
              The agent that reads everything and surfaces what's most pressing
              for you right now is being built.
            </div>
            <div>
              Your ingest will keep running in the background.
            </div>
          </div>
        </div>

        <div className="mt-auto flex flex-wrap items-center gap-6 pt-16">
          <Link
            href="/settings/agent"
            className="text-sm text-foreground/55 hover:text-foreground/85"
          >
            Configure agent
          </Link>
          <Link
            href="/connect"
            className="text-sm text-foreground/45 hover:text-foreground/75"
          >
            Manage sources
          </Link>
        </div>
      </div>
    </main>
  );
}
