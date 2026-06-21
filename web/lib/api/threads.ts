"use client";

import { withAuth } from "@/lib/api/auth";

/**
 * Client for the threads synthesis layer — the user-visible "what's
 * in motion in your life right now."
 *
 *   getThreads(userId)   — cheap: reads graph/synth/threads.jsonl
 *   runThreads(userId)   — slow: kicks off agent synthesis pass
 *                          (uses the user's configured frontier model)
 */

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

export interface ThreadEvidence {
  source: string;
  excerpt: string;
}

export interface Thread {
  id: string;
  headline: string;
  body: string;
  kind: string;       // reply | decision | follow_up | draft | appointment | research
  urgency: string;    // now | this_week | soon | someday
  liveness: number;
  related_loop_ids?: string[];
  related_person_ids?: string[];
  related_theme_labels?: string[];
  evidence: ThreadEvidence[];
  created_at?: string;
}

export interface ThreadsResponse {
  count: number;
  threads: Thread[];
}

export async function getThreads(userId: string): Promise<ThreadsResponse> {
  const r = await fetch(
    `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/synthesis/threads`,
    { cache: "no-store" },
  );
  if (!r.ok) return { count: 0, threads: [] };
  return (await r.json()) as ThreadsResponse;
}

export async function runThreads(userId: string): Promise<ThreadsResponse> {
  const r = await fetch(
    `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/synthesis/threads/run`,
    withAuth({ method: "POST" }),
  );
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(detail || `HTTP ${r.status}`);
  }
  const body = (await r.json()) as { ok: boolean; count: number; threads: Thread[] };
  return { count: body.count, threads: body.threads };
}
