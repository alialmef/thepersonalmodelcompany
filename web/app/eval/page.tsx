"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import EvalScreen from "@/components/app/eval-screen";
import { DEMO_USER_ID } from "@/lib/demo-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * Act 3.5 — the eval round. Shown the moment training completes.
 *
 * Wraps the designed <EvalScreen>. Two backend roles:
 *
 *  1. Fetch 5 grounded situations + the model's response for each:
 *       GET /v1/users/{id}/eval/prompts
 *
 *  2. Persist each judgment (accept / reject / edit + optional reason):
 *       POST /v1/users/{id}/eval/judgments
 *
 *     These judgments are DPO-grade preference data — they become training
 *     signal for the next retrain. We don't block the UI on the POST.
 *
 * On completion, /first-meeting takes over with the ceremonial arrival.
 */
interface EvalRound {
  id: string;
  situation: string;
  response: string;
}

export default function EvalPage() {
  const router = useRouter();
  const [rounds, setRounds] = useState<EvalRound[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exiting, setExiting] = useState(false);
  const userId = DEMO_USER_ID;

  useEffect(() => {
    let cancelled = false;
    fetch(
      `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/eval/prompts`,
    )
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { prompts: EvalRound[] };
        if (!cancelled) setRounds(data.prompts);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  // Fire-and-forget judgment POST. Failure shouldn't slow the user down —
  // the next retrain just doesn't get this signal.
  function persistJudgment(
    promptId: string,
    verdict: "accept" | "reject" | "edit",
    extra?: { editedText?: string; reason?: string },
  ) {
    fetch(
      `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/eval/judgments`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ promptId, verdict, ...extra }),
        keepalive: true,
      },
    ).catch(() => {
      /* best-effort */
    });
  }

  function onComplete() {
    // Play the swirl-out, then route to /first-meeting.
    setExiting(true);
    setTimeout(() => router.push(`/first-meeting?user=${encodeURIComponent(userId)}`), 600);
  }

  if (error) {
    return (
      <main className="mx-auto min-h-screen max-w-[560px] bg-white px-7 pt-11 pb-12">
        <p className="text-[14px] text-neutral-600">
          Couldn&apos;t load eval prompts: {error}. The backend may be down —
          check{" "}
          <code className="font-mono text-[12px]">./scripts/dev.sh</code>.
        </p>
      </main>
    );
  }

  if (!rounds) {
    return (
      <main className="mx-auto flex min-h-screen max-w-[560px] items-center justify-center bg-white">
        <div className="text-[13px] text-neutral-500">preparing…</div>
      </main>
    );
  }

  return (
    <div className={exiting ? "pmc-eval-exit" : undefined}>
      <EvalScreen
        rounds={rounds}
        onJudge={persistJudgment}
        onComplete={onComplete}
      />
    </div>
  );
}
