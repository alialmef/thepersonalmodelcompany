"use client";

import { isTauri } from "@/lib/runtime";

/**
 * Client for the graph snapshot. Two paths:
 *   - Tauri webview: the Rust graph_snapshot command (in-process, fast)
 *   - Browser: GET /v1/users/{id}/graph/snapshot on the backend (which
 *     reads the same JSONL files when storage_root is shared with the
 *     Tauri-local path; in prod, after Mac-app push, it reads what got
 *     uploaded)
 *
 * Returns the user's personal knowledge graph as nodes + edges — what
 * the /reading web-of-memory visualization renders. No content fields
 * (no names, no message text). Just stable ids + entity-kind labels.
 */

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

export interface GraphNode {
  id: string;
  kind: string;          // "person" | "place" | "project" | "theme" | ...
  weight?: number;       // 0..1 importance, optional — scales node radius
}

export interface GraphEdge {
  id: string;
  source: string;        // node id
  target: string;        // node id
  kind: string;          // edge kind label
}

export interface GraphSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

const EMPTY: GraphSnapshot = { nodes: [], edges: [] };

export async function graphSnapshot(userId: string): Promise<GraphSnapshot> {
  if (!userId) return EMPTY;
  // Prefer the Tauri path when available — no network hop, also works
  // offline. Fall back to the backend endpoint for browser dev.
  if (isTauri()) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return (await invoke("graph_snapshot", { userId })) as GraphSnapshot;
    } catch {
      /* fall through to backend */
    }
  }
  try {
    const r = await fetch(
      `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/graph/snapshot`,
      { cache: "no-store" },
    );
    if (!r.ok) return EMPTY;
    return (await r.json()) as GraphSnapshot;
  } catch {
    return EMPTY;
  }
}
