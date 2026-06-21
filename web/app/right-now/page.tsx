"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { BrandMark } from "@/components/shared/brand-mark";
import { useUser } from "@/hooks/use-user";
import {
  getPatterns,
  getThreads,
  runPatterns,
  runThreads,
  type Pattern,
  type Thread,
} from "@/lib/api/threads";

/**
 * /right-now — the first-boot moment.
 *
 * Reads graph/synth/threads.jsonl via /v1/users/{id}/synthesis/threads.
 * If the file is empty, prompts the user to run a synthesis pass —
 * which calls their configured frontier model with their structured
 * graph as context and produces the threads to surface.
 */

const URGENCY_LABEL: Record<string, string> = {
  now: "Now",
  this_week: "This week",
  soon: "Soon",
  someday: "Someday",
};

const URGENCY_ORDER = ["now", "this_week", "soon", "someday"];

export default function RightNowPage() {
  const { user } = useUser();
  const userId = user?.pmcUserId ?? "";

  const [threads, setThreads] = useState<Thread[] | null>(null);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initial load — fetch both threads and patterns in parallel
  useEffect(() => {
    if (!userId) return;
    let cancelled = false;
    (async () => {
      try {
        const [t, p] = await Promise.all([
          getThreads(userId),
          getPatterns(userId),
        ]);
        if (!cancelled) {
          setThreads(t.threads);
          setPatterns(p.patterns);
        }
      } catch {
        if (!cancelled) setThreads([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [userId]);

  async function handleRunSynthesis() {
    if (!userId) return;
    setBusy(true);
    setError(null);
    try {
      // Patterns first (pure compute, fast), then threads (agent call)
      const p = await runPatterns(userId);
      setPatterns(p.patterns);
      const r = await runThreads(userId);
      setThreads(r.threads);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Couldn't synthesize threads.",
      );
    } finally {
      setBusy(false);
    }
  }

  // Group by urgency, in the canonical order
  const grouped: Record<string, Thread[]> = {};
  for (const t of threads ?? []) {
    const u = URGENCY_ORDER.includes(t.urgency) ? t.urgency : "soon";
    (grouped[u] ||= []).push(t);
  }

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <header className="mt-12 space-y-3 text-foreground/85">
          <div className="text-2xl font-semibold text-foreground">
            What you should get moving on.
          </div>
          <div className="text-base text-foreground/55">
            I read everything on your Mac. These are the threads I think
            are in motion right now.
          </div>
        </header>

        {threads === null && (
          <div className="mt-20 text-sm text-foreground/40">Loading…</div>
        )}

        {threads !== null && threads.length === 0 && (
          <EmptyState busy={busy} error={error} onRun={handleRunSynthesis} />
        )}

        {threads !== null && threads.length > 0 && (
          <>
            <div className="mt-16 space-y-12">
              {URGENCY_ORDER.map((u) =>
                grouped[u] && grouped[u].length > 0 ? (
                  <UrgencySection
                    key={u}
                    label={URGENCY_LABEL[u]}
                    threads={grouped[u]}
                  />
                ) : null,
              )}
            </div>

            {patterns.length > 0 && (
              <section className="mt-20">
                <div className="text-xs uppercase tracking-[0.18em] text-foreground/40 mb-6">
                  What your life keeps doing
                </div>
                <div className="space-y-6">
                  {patterns.map((p) => (
                    <PatternCard key={p.id} pattern={p} />
                  ))}
                </div>
              </section>
            )}

            <div className="mt-16">
              <button
                type="button"
                onClick={handleRunSynthesis}
                disabled={busy}
                className={`text-sm transition-colors ${
                  busy
                    ? "cursor-default text-foreground/25"
                    : "text-foreground/45 hover:text-foreground/75"
                }`}
              >
                {busy ? "Reading again…" : "Read again"}
              </button>
            </div>
            {error && (
              <div className="mt-3 text-sm text-red-500">{error}</div>
            )}
          </>
        )}

        <div className="mt-auto flex flex-wrap items-center gap-6 pt-16">
          <Link
            href="/settings/agent"
            className="text-sm text-foreground/55 hover:text-foreground/85"
          >
            Configure agent
          </Link>
          <Link
            href="/knowledge-update"
            className="text-sm text-foreground/45 hover:text-foreground/75"
          >
            What I know
          </Link>
        </div>
      </div>
    </main>
  );
}

function EmptyState({
  busy,
  error,
  onRun,
}: {
  busy: boolean;
  error: string | null;
  onRun: () => void | Promise<void>;
}) {
  return (
    <div className="mt-20 space-y-6">
      <div className="text-base text-foreground/55">
        I haven&apos;t read your life yet. When you&apos;re ready, I&apos;ll go through
        the structure of what&apos;s on your Mac and name what&apos;s in motion.
      </div>
      <button
        type="button"
        onClick={onRun}
        disabled={busy}
        className={`text-base transition-colors ${
          busy
            ? "cursor-default text-foreground/25"
            : "text-foreground/80 hover:text-foreground"
        }`}
      >
        {busy ? "Reading…" : "Read my life"}
      </button>
      {error && <div className="text-sm text-red-500">{error}</div>}
    </div>
  );
}

function UrgencySection({
  label,
  threads,
}: {
  label: string;
  threads: Thread[];
}) {
  return (
    <section>
      <div className="text-xs uppercase tracking-[0.18em] text-foreground/40 mb-6">
        {label}
      </div>
      <div className="space-y-10">
        {threads.map((t) => (
          <ThreadCard key={t.id} thread={t} />
        ))}
      </div>
    </section>
  );
}

function PatternCard({ pattern }: { pattern: Pattern }) {
  return (
    <article className="space-y-1">
      <div className="text-[15px] text-foreground/90 leading-snug">
        {pattern.headline}
      </div>
      {pattern.detail && (
        <div className="text-[13px] text-foreground/55 leading-relaxed">
          {pattern.detail}
        </div>
      )}
      <div className="text-[11px] uppercase tracking-wider text-foreground/40 pt-1">
        {pattern.category}
      </div>
    </article>
  );
}

function ThreadCard({ thread }: { thread: Thread }) {
  const [showEvidence, setShowEvidence] = useState(false);
  return (
    <article className="space-y-2">
      <div className="text-[17px] text-foreground leading-snug">
        {thread.headline}
      </div>
      {thread.body && (
        <div className="text-sm text-foreground/60 leading-relaxed">
          {thread.body}
        </div>
      )}
      <div className="flex flex-wrap items-baseline gap-4 pt-1">
        <span className="text-[11px] uppercase tracking-wider text-foreground/40">
          {thread.kind.replace("_", " ")}
        </span>
        {thread.evidence.length > 0 && (
          <button
            type="button"
            onClick={() => setShowEvidence((v) => !v)}
            className="text-xs text-foreground/45 hover:text-foreground/75"
          >
            {showEvidence ? "Hide source" : "Why this"}
          </button>
        )}
      </div>
      {showEvidence && (
        <div className="mt-3 space-y-2 border-l border-foreground/15 pl-4">
          {thread.evidence.map((e, i) => (
            <div key={i} className="text-xs text-foreground/50">
              <span className="text-foreground/35">{e.source}: </span>
              <span className="italic">&ldquo;{e.excerpt}&rdquo;</span>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}
