"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import CurateScreen from "@/components/app/curate-screen";
import { DEMO_USER_ID } from "@/lib/demo-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

interface CurateEvent {
  ts: string;
  text: string;
}

interface AuditEvent {
  timestamp?: string;
  stage?: string;
  event?: string;
  data?: Record<string, unknown>;
  status?: string;
  result?: { style_profile_lines?: string[] } & Record<string, unknown>;
}

/**
 * Screen 3 — Reading you.
 *
 * Consumes the backend SSE stream
 *   /v1/users/{user_id}/runs/{job_id}/events
 *
 * Filters down to the "curate" stage and formats each backend event into the
 * shape CurateScreen expects: { ts, text }. The style-profile lines come
 * from the curate_completed event's `data.style_profile_summary` (one line
 * per observation).
 *
 * When we see the first "train" stage event arrive, we navigate to /train
 * — the curate phase is done, training has begun.
 */
function CuratePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const userId = searchParams.get("user") ?? DEMO_USER_ID;

  const [events, setEvents] = useState<CurateEvent[]>([]);
  const [profileLines, setProfileLines] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const trainSeenRef = useRef(false);

  useEffect(() => {
    if (!jobId) return;
    const url = `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/runs/${encodeURIComponent(jobId)}/events`;
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

      // Once the train stage emits an event, kick to /train.
      if (parsed.stage === "train" && !trainSeenRef.current) {
        trainSeenRef.current = true;
        router.push(
          `/train?job=${encodeURIComponent(jobId)}&user=${encodeURIComponent(userId)}`,
        );
        es.close();
        return;
      }

      // job_finished arriving here means curate or earlier stage failed,
      // OR the pipeline skipped train entirely. Either way we route on.
      if (parsed.event === "job_finished") {
        if (parsed.status === "completed") {
          router.push(`/eval?user=${encodeURIComponent(userId)}`);
        }
        es.close();
        return;
      }

      // Pull style profile lines from curate_completed payload.
      if (parsed.event === "curate_completed" && parsed.data) {
        const summary = parsed.data.style_profile_summary;
        if (Array.isArray(summary)) {
          setProfileLines(summary.filter((s): s is string => typeof s === "string"));
        }
        setIsComplete(true);
      }

      // Append every curate-stage event to the activity stream.
      if (parsed.stage === "curate" || parsed.stage === "ingest") {
        const ts = formatTime(parsed.timestamp);
        const text = formatEventText(parsed);
        setEvents((prev) => [...prev, { ts, text }]);
      }
    };

    return () => es.close();
  }, [jobId, userId, router]);

  if (!jobId) {
    return (
      <main className="mx-auto flex min-h-screen max-w-[620px] items-center justify-center bg-white px-7">
        <p className="text-[13px] text-neutral-500">no curate job in progress</p>
      </main>
    );
  }

  return (
    <CurateScreen
      events={events}
      profileLines={profileLines}
      isComplete={isComplete}
      onContinue={() => {
        // The pipeline is already running and will auto-navigate to /train
        // when training begins. Continue is a soft acknowledgment — if the
        // user clicks before training kicks off, just keep waiting.
        if (trainSeenRef.current) {
          router.push(
            `/train?job=${encodeURIComponent(jobId)}&user=${encodeURIComponent(userId)}`,
          );
        }
      }}
    />
  );
}

export default function CuratePage() {
  return (
    <Suspense
      fallback={<main className="min-h-screen bg-white" />}
    >
      <CuratePageInner />
    </Suspense>
  );
}

function formatTime(iso?: string): string {
  if (!iso) {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}`;
  }
  try {
    const d = new Date(iso);
    return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  } catch {
    return "";
  }
}

function formatEventText(ev: AuditEvent): string {
  // Friendly mapping for the common events we expect. Falls back to the
  // raw event name + data summary so nothing is hidden.
  const evName = ev.event ?? "event";
  const data = ev.data ?? {};

  const friendly: Record<string, (d: Record<string, unknown>) => string> = {
    ingest_completed: (d) => `Read ${num(d.items)} ${plural(d.items, "item")}.`,
    curate_started: () => "Curating your writing.",
    dedupe_completed: (d) => `Removed ${num(d.removed)} duplicates.`,
    quality_filter: (d) => `Filtered ${num(d.filtered)} short replies.`,
    pii_filter: (d) => `Excluded ${num(d.filtered)} items with sensitive info.`,
    style_profile_built: () => "Style profile built.",
    curate_completed: (d) =>
      `Curate finished — ${num(d.kept)} ${plural(d.kept, "example")} ready.`,
  };

  if (evName in friendly) {
    return friendly[evName](data);
  }
  // Generic fallback.
  const bits = Object.entries(data)
    .filter(([_, v]) => typeof v !== "object" || v === null)
    .map(([k, v]) => `${k}=${typeof v === "number" ? v : v}`)
    .slice(0, 2)
    .join(" ");
  return bits ? `${evName} · ${bits}` : evName;
}

function num(x: unknown): string {
  if (typeof x === "number") return x.toLocaleString();
  return String(x ?? "—");
}

function plural(x: unknown, singular: string): string {
  if (typeof x === "number" && x === 1) return singular;
  return singular + "s";
}
