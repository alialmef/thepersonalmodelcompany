"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { BrandMark } from "@/components/shared/brand-mark";
import { useUser } from "@/hooks/use-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * /reading — Phase 1.1 ingest progress.
 *
 * After /connect kicks off graph extraction, this page polls
 * /v1/users/{id}/status and renders typed-prose lines describing
 * what's being read. Progresses to /right-now when ingest is
 * "steady enough" — i.e. the total raw item count hasn't moved for
 * a few polls. The user can also tap Continue at any time.
 *
 * No training run is involved. Phase 1.1 disconnects training from
 * the active flow; the underlying code is preserved but unreachable
 * from the user-facing path.
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

const POLL_MS = 3_000;
const STEADY_AFTER_POLLS = 5;
const MIN_DWELL_MS = 6_000;

export default function ReadingPage() {
  const router = useRouter();
  const { user } = useUser();
  const [sources, setSources] = useState<SourceBreakdown[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [steady, setSteady] = useState(false);
  const lastTotalRef = useRef<number>(-1);
  const steadyCountRef = useRef<number>(0);
  const mountedAt = useRef<number>(Date.now());

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

        // Steadiness check: same total across N polls + minimum dwell
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

  // Auto-advance once steady — give the user a beat to read the page first.
  useEffect(() => {
    if (!steady) return;
    const t = setTimeout(() => router.push("/right-now"), 1200);
    return () => clearTimeout(t);
  }, [steady, router]);

  const sortedSources = [...sources]
    .sort((a, b) => (b.item_count ?? 0) - (a.item_count ?? 0))
    .slice(0, 10);

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <div className="mt-16 space-y-10 text-foreground/85">
          <div className="space-y-3">
            <div className="text-xl font-semibold text-foreground">
              {steady ? "Done." : "Reading."}
            </div>
            <div className="text-base text-foreground/55">
              {total > 0
                ? `${total.toLocaleString()} items so far.`
                : "Opening your sources…"}
            </div>
          </div>

          {sortedSources.length > 0 && (
            <div className="space-y-1.5 text-[15px] text-foreground/70">
              {sortedSources.map((s) => (
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
        </div>

        <div className="mt-auto pt-16">
          <button
            type="button"
            onClick={() => router.push("/right-now")}
            className="text-base text-foreground/55 hover:text-foreground/85 transition-colors"
          >
            Continue
          </button>
        </div>
      </div>
    </main>
  );
}
