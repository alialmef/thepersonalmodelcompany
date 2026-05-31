"use client";

import { withAuth } from "@/lib/api/auth";

/**
 * Client for the backend's /v1/users/{id}/knowledge/* endpoints —
 * the redact + manage surface that the /knowledge-update screen renders.
 *
 * Backed by pmc/storage/redactions.py + the existing UserStore
 * (search + per-source counts).
 */

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

export interface KnowledgeSource {
  source_id: string;
  kind: string;
  item_count: number;
  paused: boolean;
}

export interface Redaction {
  id: string;
  kind: "person" | "topic" | "date_range";
  value: string;
  added_at: string;
  note?: string;
}

export interface KnowledgeOverview {
  sources: KnowledgeSource[];
  redactions: Redaction[];
  paused_sources: { source_id: string; paused_at: string }[];
}

export interface SearchResult {
  id?: string;
  source_id?: string;
  kind?: string;
  preview: string;
  timestamp?: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  truncated: boolean;
}

function userPath(userId: string, rest: string): string {
  return `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/knowledge${rest}`;
}

export async function getOverview(userId: string): Promise<KnowledgeOverview> {
  const r = await fetch(userPath(userId, "/overview"), withAuth({ cache: "no-store" }));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as KnowledgeOverview;
}

export async function pauseSource(userId: string, sourceId: string): Promise<void> {
  await fetch(
    userPath(userId, `/sources/${encodeURIComponent(sourceId)}/pause`),
    withAuth({ method: "POST" }),
  );
}

export async function resumeSource(userId: string, sourceId: string): Promise<void> {
  await fetch(
    userPath(userId, `/sources/${encodeURIComponent(sourceId)}/resume`),
    withAuth({ method: "POST" }),
  );
}

export async function addRedaction(
  userId: string,
  args: { kind: Redaction["kind"]; value: string; note?: string },
): Promise<Redaction> {
  const r = await fetch(
    userPath(userId, "/redactions"),
    withAuth({
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(args),
    }),
  );
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as Redaction;
}

export async function removeRedaction(
  userId: string,
  redactionId: string,
): Promise<void> {
  await fetch(
    userPath(userId, `/redactions/${encodeURIComponent(redactionId)}`),
    withAuth({ method: "DELETE" }),
  );
}

export async function search(
  userId: string,
  query: string,
  limit = 50,
): Promise<SearchResponse> {
  const u = new URL(userPath(userId, "/search"));
  u.searchParams.set("q", query);
  u.searchParams.set("limit", String(limit));
  const r = await fetch(u.toString(), withAuth({ cache: "no-store" }));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as SearchResponse;
}

/** Per-item forget. V1: backend returns 501 — UI shows a hint instead. */
export async function forgetItem(userId: string, itemId: string): Promise<boolean> {
  const r = await fetch(
    userPath(userId, `/items/${encodeURIComponent(itemId)}`),
    withAuth({ method: "DELETE" }),
  );
  return r.ok;
}

/** Nuclear erase — wipes user data + any deployed adapter + redactions. */
export async function eraseEverything(userId: string): Promise<void> {
  await fetch(
    `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/reset`,
    withAuth({ method: "POST" }),
  );
}
