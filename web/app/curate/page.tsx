"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import ReadingScreen, {
  type ReadingItem,
} from "@/components/app/reading-screen";
import { useUser } from "@/hooks/use-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/**
 * /curate — the screen between /connect and /train (we'll rename the
 * route to /reading once the rest of the flow stabilizes).
 *
 * Subscribes to the backend SSE event stream and turns
 * `reading_source_found` events into typed lines on the page.
 *
 *   stage=memory, event=reading_source_found
 *     data: { bucket: "voice" | "memory", kind, count, phrase }
 *
 * The screen advances when either:
 *   - `memory_consolidate_completed` fires (full smart memory built), OR
 *   - `memory_preview_completed`  fires (smart memory was skipped — preview only), OR
 *   - `curate_completed`          fires (in legacy runs without memory stage)
 */

interface AuditEvent {
  timestamp?: string;
  stage?: string;
  event?: string;
  data?: Record<string, unknown>;
  status?: string;
}

function CurateInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const { user } = useUser();
  const userId = searchParams.get("user") ?? user?.pmcUserId ?? "";

  const [items, setItems] = useState<ReadingItem[]>([]);
  const [ready, setReady] = useState(false);
  const dedupRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!jobId) return;
    const url = `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/runs/${encodeURIComponent(jobId)}/events`;
    const es = new EventSource(url);

    es.onmessage = (msg) => {
      if (msg.data === "[DONE]") {
        es.close();
        return;
      }
      let ev: AuditEvent;
      try {
        ev = JSON.parse(msg.data) as AuditEvent;
      } catch {
        return;
      }

      if (ev.event === "reading_source_found" && ev.data) {
        const bucket = ev.data.bucket as "voice" | "memory" | undefined;
        const kind = ev.data.kind as string | undefined;
        const count = ev.data.count as number | undefined;
        const phrase = ev.data.phrase as string | undefined;
        if (!bucket || !kind || count == null || !phrase) return;
        const key = `${bucket}:${kind}`;
        if (dedupRef.current.has(key)) return;
        dedupRef.current.add(key);
        setItems((prev) => [...prev, { bucket, kind, count, phrase }]);
        return;
      }

      // Only unlock Continue when the memory section has actually been
      // populated (and not just because curate finished while memory
      // was still empty). Hard gate: require memory_migrate_completed
      // — that's the event fired only after _emit_memory_sources has
      // run, which means memory items are guaranteed to have been
      // dispatched (or honestly skipped if the user has thin data).
      if (
        ev.event === "memory_migrate_completed" ||
        ev.event === "memory_consolidate_completed" ||
        ev.event === "job_finished"
      ) {
        setReady(true);
        return;
      }
    };

    es.onerror = () => {
      // Connection drops here are expected — backend closes the SSE
      // when the job ends. We've already collected what we need.
    };

    return () => es.close();
  }, [jobId, userId]);

  const sorted = useMemo(() => {
    // Stable order — voice first, then memory; within each bucket,
    // by the order events arrived.
    return [
      ...items.filter((i) => i.bucket === "voice"),
      ...items.filter((i) => i.bucket === "memory"),
    ];
  }, [items]);

  const handleContinue = () => {
    router.push(
      `/train?job=${encodeURIComponent(jobId ?? "")}&user=${encodeURIComponent(userId)}`,
    );
  };

  return (
    <ReadingScreen items={sorted} ready={ready} onContinue={handleContinue} />
  );
}

export default function CuratePage() {
  return (
    <Suspense fallback={<div className="min-h-screen w-full bg-background" />}>
      <CurateInner />
    </Suspense>
  );
}
