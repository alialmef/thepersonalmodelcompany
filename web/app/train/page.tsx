"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";

import { DEMO_USER_ID } from "@/lib/demo-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

const STAGES = [
  { key: "curate", label: "curating your writing" },
  { key: "train", label: "training your model" },
  { key: "eval", label: "evaluating" },
  { key: "publish", label: "packaging your model" },
] as const;

type StageKey = (typeof STAGES)[number]["key"];

interface AuditEvent {
  timestamp?: string;
  stage?: string;
  event?: string;
  data?: Record<string, unknown>;
  // job_finished frames have a different shape:
  status?: string;
  job_id?: string;
  result?: unknown;
  elapsed_seconds?: number | null;
  message?: string;
}

interface Activity {
  ts: number;
  stage: StageKey | "other";
  text: string;
}

/**
 * Act 3 — live training progress.
 *
 * The middle of the user journey: between connect and chat is a black hole
 * where training happens. This page makes the wait honest, alive, and
 * unembarrassing — every event from the backend shows up as a line in the
 * activity feed, the four stages light up in order, and "you can close this
 * window" is reassuring instead of evasive.
 *
 * Wire: the connect page POSTs to /v1/users/{id}/runs to start a job, then
 * router.push(`/train?job={jobId}`). This page connects EventSource directly
 * to the backend's SSE endpoint and renders what arrives.
 */
export default function TrainPage() {
  // useSearchParams requires a Suspense boundary at this level in Next 15+.
  return (
    <Suspense fallback={<NoJobState />}>
      <TrainPageInner />
    </Suspense>
  );
}

function TrainPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const userId = searchParams.get("user") ?? DEMO_USER_ID;

  const [currentStage, setCurrentStage] = useState<StageKey>("curate");
  const [completedStages, setCompletedStages] = useState<Set<StageKey>>(new Set());
  const [activity, setActivity] = useState<Activity[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [done, setDone] = useState<"ok" | "fail" | null>(null);
  const [doneSummary, setDoneSummary] = useState<string>("");
  const startedAtRef = useRef<number>(Date.now());

  // ---------- elapsed time ticker ----------
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ---------- SSE consumer ----------
  useEffect(() => {
    if (!jobId) {
      // No job — show a calm "no job" state. Useful for design preview too.
      return;
    }

    const url = `${PMC_API_URL}/v1/users/${encodeURIComponent(
      userId,
    )}/runs/${encodeURIComponent(jobId)}/events`;
    const es = new EventSource(url);

    es.onmessage = (msg) => {
      if (msg.data === "[DONE]") {
        es.close();
        return;
      }
      let parsed: AuditEvent;
      try {
        parsed = JSON.parse(msg.data) as AuditEvent;
      } catch {
        return;
      }
      handleEvent(parsed);
    };
    es.onerror = () => {
      // Browser auto-reconnects; we don't need to do anything special.
    };

    return () => {
      es.close();
    };
  }, [jobId, userId]);

  function handleEvent(ev: AuditEvent) {
    // Final job_finished frame
    if (ev.event === "job_finished") {
      if (ev.status === "completed") {
        setDone("ok");
        setDoneSummary("model is ready");
        setTimeout(() => router.push("/chat"), 1800);
      } else {
        setDone("fail");
        setDoneSummary(`training ${ev.status ?? "failed"} — see logs`);
      }
      return;
    }

    if (ev.event === "timeout") {
      setDone("fail");
      setDoneSummary(ev.message ?? "no progress for 10 minutes");
      return;
    }

    // Normal stage event
    const stage = (ev.stage ?? "other") as StageKey | "other";
    if (stage !== "other" && STAGES.some((s) => s.key === stage)) {
      setCurrentStage(stage);
      // Mark previous stages as complete based on canonical order
      setCompletedStages((prev) => {
        const next = new Set(prev);
        const idx = STAGES.findIndex((s) => s.key === stage);
        for (let i = 0; i < idx; i++) next.add(STAGES[i].key);
        return next;
      });
    }

    // Append to activity feed (most recent first)
    setActivity((prev) =>
      [
        {
          ts: ev.timestamp ? new Date(ev.timestamp).getTime() : Date.now(),
          stage: (stage === "other" ? "other" : stage) as StageKey | "other",
          text: formatEvent(ev),
        },
        ...prev,
      ].slice(0, 60),
    );
  }

  const elapsedLabel = useMemo(() => formatElapsed(elapsed), [elapsed]);

  if (!jobId) {
    return <NoJobState />;
  }

  return (
    <main className="train-root">
      <header className="train-header">
        <Link href="/connect" className="train-brand">
          The Personal Model Company
        </Link>
        <span className="train-elapsed">{elapsedLabel}</span>
      </header>

      <section className="train-stage-section">
        <h1 className="train-title">
          {done === "ok"
            ? "your model is ready"
            : done === "fail"
            ? doneSummary
            : "training your model"}
        </h1>
        {done === null && (
          <p className="train-subtitle">
            this usually takes 60 – 180 minutes. you can close this window and
            come back — we&apos;ll keep going.
          </p>
        )}

        <ol className="train-stages">
          {STAGES.map((s) => {
            const isCurrent = s.key === currentStage && done === null;
            const isComplete = completedStages.has(s.key) || done === "ok";
            return (
              <li
                key={s.key}
                className={`train-stage ${
                  isComplete
                    ? "train-stage--done"
                    : isCurrent
                    ? "train-stage--current"
                    : "train-stage--pending"
                }`}
              >
                <span className="train-stage-dot" />
                <span className="train-stage-label">{s.label}</span>
                {isCurrent && (
                  <span className="train-stage-spinner" aria-hidden="true">
                    <span /><span /><span />
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </section>

      <section className="train-activity">
        <h2 className="train-activity-title">activity</h2>
        {activity.length === 0 ? (
          <p className="train-activity-empty">waiting for first event…</p>
        ) : (
          <ul className="train-activity-list">
            {activity.map((a, i) => (
              <li key={i} className="train-activity-item">
                <span
                  className={`train-activity-tag train-activity-tag--${a.stage}`}
                >
                  {a.stage}
                </span>
                <span className="train-activity-text">{a.text}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s.toString().padStart(2, "0")}s`;
  return `${s}s`;
}

function formatEvent(ev: AuditEvent): string {
  // Audit events come in (stage, event, data). Render concisely.
  const evName = ev.event ?? "event";
  const data = ev.data ?? {};
  const dataBits = Object.entries(data)
    .filter(([_, v]) => typeof v !== "object" || v === null)
    .map(([k, v]) => `${k}=${formatValue(v)}`)
    .join(" ");
  return dataBits ? `${evName}  ${dataBits}` : evName;
}

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  }
  if (typeof v === "string") {
    return v.length > 60 ? v.slice(0, 60) + "…" : v;
  }
  return String(v);
}

function NoJobState() {
  return (
    <main className="train-root">
      <header className="train-header">
        <Link href="/connect" className="train-brand">
          The Personal Model Company
        </Link>
      </header>
      <section className="train-stage-section train-stage-section--empty">
        <h1 className="train-title">no training in progress</h1>
        <p className="train-subtitle">
          start a training run from{" "}
          <Link href="/connect" className="train-inline-link">
            connect
          </Link>{" "}
          and you&apos;ll land back here.
        </p>
      </section>
    </main>
  );
}
