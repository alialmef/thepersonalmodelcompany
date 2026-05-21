"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import CurateScreen from "@/components/app/curate-screen";
import { useUser } from "@/hooks/use-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

interface CurateEvent {
  ts: string;
  text: string;
}

interface SourceCount {
  label: string;
  count: number;
}

function sourceLabel(kind: string): string {
  switch (kind) {
    case "imessage":
      return "Messages";
    case "notes":
    case "text":
      return "Notes";
    case "document":
      return "Documents";
    case "email_mbox":
    case "email":
      return "Mail";
    case "whatsapp":
      return "WhatsApp";
    default:
      return kind || "Other";
  }
}

/**
 * When the backend doesn't ship a text style profile, synthesize a few
 * "lay of your writing" lines from the real curate stats. Keeps step 2
 * feeling substantive even without an LLM-generated narrative.
 */
function synthesizeProfile(d: Record<string, unknown>): string[] {
  const lines: string[] = [];
  const input = numFrom(d.input_conversations);
  const output = numFrom(d.output_completions);
  const droppedShort = numFrom(d.dropped_short);
  const droppedDup = numFrom(d.dropped_duplicate);
  const redacted = numFrom(d.redacted_severe);

  if (input !== null) {
    lines.push(`Read ${input.toLocaleString()} conversations from your writing.`);
  }
  if (output !== null) {
    lines.push(
      `Kept ${output.toLocaleString()} examples that look like you on a good day.`,
    );
  }
  if (droppedShort !== null && droppedShort > 0) {
    lines.push(`Dropped ${droppedShort.toLocaleString()} replies too short to teach from.`);
  }
  if (droppedDup !== null && droppedDup > 0) {
    lines.push(`Removed ${droppedDup.toLocaleString()} near-duplicates.`);
  }
  if (redacted !== null && redacted > 0) {
    lines.push(`Redacted ${redacted.toLocaleString()} items with sensitive content.`);
  }
  return lines;
}

function numFrom(x: unknown): number | null {
  return typeof x === "number" ? x : null;
}

function labelKind(k: unknown): string {
  if (typeof k !== "string") return "your data";
  switch (k) {
    case "imessage":
      return "Messages";
    case "notes":
    case "text":
      return "Notes";
    case "document":
      return "Documents";
    case "email_mbox":
    case "email":
      return "Mail";
    case "whatsapp":
      return "WhatsApp";
    default:
      return k;
  }
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
  const { user } = useUser();
  const userId = searchParams.get("user") ?? user?.pmcUserId ?? "";

  const [events, setEvents] = useState<CurateEvent[]>([]);
  const [profileLines, setProfileLines] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const [sourceCounts, setSourceCounts] = useState<SourceCount[]>([]);
  const trainSeenRef = useRef(false);

  // Poll user status to render a live per-source scoreboard. Step 2 should
  // feel substantive — "I am reading you" only works if the user can see
  // the actual counts as they accumulate.
  useEffect(() => {
    if (!userId) return;
    let cancelled = false;
    const fetchStatus = async () => {
      try {
        const res = await fetch(
          `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/status`,
          { cache: "no-store" },
        );
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as {
          raw_source_breakdown?: Array<{
            source_id?: string;
            kind?: string;
            item_count?: number;
          }>;
        };
        const merged = new Map<string, SourceCount>();
        for (const s of data.raw_source_breakdown ?? []) {
          const label = sourceLabel(s.kind ?? "");
          const prev = merged.get(label);
          merged.set(label, {
            label,
            count: (prev?.count ?? 0) + (s.item_count ?? 0),
          });
        }
        if (!cancelled) {
          setSourceCounts(
            Array.from(merged.values())
              .filter((s) => s.count > 0)
              .sort((a, b) => b.count - a.count),
          );
        }
      } catch {
        /* keep polling — backend hiccups are non-fatal here */
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 1500);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [userId]);

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

      // Track that training started so the Continue button can take the user
      // there, but DON'T auto-route. Step 2 is a deliberate pause — the user
      // gets to see their numbers and the lay of their writing first.
      if (parsed.stage === "train") {
        trainSeenRef.current = true;
        return;
      }

      // job_finished while we're still on curate means the pipeline finished
      // end-to-end. Mark complete; the user advances via Continue.
      if (parsed.event === "job_finished") {
        setIsComplete(true);
        es.close();
        return;
      }

      // Pull style profile lines from curate_completed. When the backend
      // doesn't carry text observations, synthesize a few lines from the
      // real stats so the right column has substance.
      if (parsed.event === "curate_completed" && parsed.data) {
        const summary = parsed.data.style_profile_summary;
        if (Array.isArray(summary)) {
          setProfileLines(summary.filter((s): s is string => typeof s === "string"));
        } else {
          setProfileLines(synthesizeProfile(parsed.data));
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
      sourceCounts={sourceCounts}
      isComplete={isComplete}
      onContinue={() => {
        router.push(
          `/train?job=${encodeURIComponent(jobId)}&user=${encodeURIComponent(userId)}`,
        );
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
    items_pushed_via_native: (d) =>
      `Read ${num(d.items)} ${plural(d.items, "item")} from ${labelKind(d.kind)}.`,
    ingest_completed: (d) => `Read ${num(d.items)} ${plural(d.items, "item")}.`,
    source_uploaded: (d) =>
      `Added ${num(d.items)} ${plural(d.items, "item")} from ${d.filename ?? labelKind(d.kind)}.`,
    curate_started: () => "Curating your writing.",
    dedupe_completed: (d) => `Removed ${num(d.removed)} duplicates.`,
    quality_filter: (d) => `Filtered ${num(d.filtered)} short replies.`,
    pii_filter: (d) => `Excluded ${num(d.filtered)} items with sensitive info.`,
    style_profile_built: () => "Style profile built.",
    curate_completed: (d) =>
      `Curate finished — kept ${num(d.output_completions)} of ${num(d.input_conversations)} conversations.`,
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
