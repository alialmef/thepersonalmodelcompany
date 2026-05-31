"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { BrandMark } from "@/components/shared/brand-mark";
import { useUser } from "@/hooks/use-user";
import { withAuth } from "@/lib/api/auth";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * /confirm — the validation pass.
 *
 * After /reading completes, the agent has structured the user's graph
 * but hasn't told them anything specific yet. This is the trust-
 * building moment: the agent surfaces 5-10 concrete claims about the
 * user, each with evidence, and the user accepts/corrects each.
 *
 * The endpoint is /v1/users/{id}/synthesis/claims (POST, runs through
 * the user's configured frontier provider via the agent prompt module).
 * V1 falls back to a friendly placeholder if the endpoint isn't ready
 * — the screen still exists and the flow still works.
 */

interface Evidence {
  source: string;
  summary: string;
}

interface Claim {
  claim: string;
  kind: string;
  evidence: Evidence[];
}

type Verdict = "pending" | "yes" | "no";

export default function ConfirmPage() {
  const router = useRouter();
  const { user } = useUser();
  const [claims, setClaims] = useState<Claim[]>([]);
  const [verdicts, setVerdicts] = useState<Verdict[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    if (!user?.pmcUserId) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(
          `${PMC_API_URL}/v1/users/${encodeURIComponent(user.pmcUserId)}/synthesis/claims`,
          withAuth({ cache: "no-store" }),
        );
        if (cancelled) return;
        if (!r.ok) {
          // V1 stub: the endpoint may 404 / 501 until backend lands.
          // Don't block the flow — the user can still continue.
          setError("The agent is still settling in. Try again in a minute.");
          setLoading(false);
          return;
        }
        const body = (await r.json()) as { claims: Claim[] };
        if (cancelled) return;
        setClaims(body.claims ?? []);
        setVerdicts((body.claims ?? []).map(() => "pending" as Verdict));
      } catch {
        if (!cancelled) {
          setError("Couldn't reach the agent.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user?.pmcUserId]);

  const allAnswered = claims.length > 0 && verdicts.every((v) => v !== "pending");

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <header className="mt-12 space-y-3 text-foreground/85">
          <div className="text-xl font-semibold text-foreground">
            Let me check what I learned.
          </div>
          <div className="text-base text-foreground/55">
            Mark each one yes or no. This is how I learn what I got right
            and what to forget.
          </div>
        </header>

        <div className="mt-16 space-y-8">
          {loading && (
            <div className="text-sm text-foreground/40">Thinking…</div>
          )}

          {!loading && error && (
            <div className="space-y-4">
              <div className="text-sm text-foreground/55">{error}</div>
            </div>
          )}

          {!loading &&
            !error &&
            claims.map((c, i) => (
              <div key={i} className="space-y-3">
                <div className="text-[17px] text-foreground/90">{c.claim}</div>
                {c.evidence.length > 0 && (
                  <div className="text-xs text-foreground/40 space-y-0.5">
                    {c.evidence.map((e, j) => (
                      <div key={j}>
                        <span className="font-mono">{e.source}</span> · {e.summary}
                      </div>
                    ))}
                  </div>
                )}
                <div className="flex items-baseline gap-6">
                  <button
                    type="button"
                    onClick={() =>
                      setVerdicts((vs) => {
                        const next = vs.slice();
                        next[i] = "yes";
                        return next;
                      })
                    }
                    className={`text-sm transition-colors ${
                      verdicts[i] === "yes"
                        ? "text-foreground"
                        : "text-foreground/45 hover:text-foreground/80"
                    }`}
                  >
                    Yes
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setVerdicts((vs) => {
                        const next = vs.slice();
                        next[i] = "no";
                        return next;
                      })
                    }
                    className={`text-sm transition-colors ${
                      verdicts[i] === "no"
                        ? "text-foreground"
                        : "text-foreground/45 hover:text-foreground/80"
                    }`}
                  >
                    No
                  </button>
                </div>
              </div>
            ))}
        </div>

        <div className="mt-auto pt-16 flex items-baseline gap-8">
          <button
            type="button"
            onClick={() => router.push("/right-now")}
            disabled={!allAnswered && claims.length > 0}
            className={`text-base transition-colors ${
              !allAnswered && claims.length > 0
                ? "cursor-default text-foreground/25"
                : "text-foreground/80 hover:text-foreground"
            }`}
          >
            Continue
          </button>
          <button
            type="button"
            onClick={() => router.push("/right-now")}
            className="text-sm text-foreground/40 hover:text-foreground/65"
          >
            Skip
          </button>
        </div>
      </div>
    </main>
  );
}
