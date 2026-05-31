"use client";

import { isTauri } from "@/lib/runtime";

/**
 * Client for the Tauri-side graph_snapshot command. Returns the user's
 * personal knowledge graph as nodes + edges — the data the /reading
 * web-of-memory visualization renders.
 *
 * No content fields are returned (no names, no message text). Just
 * stable ids + entity-kind labels. The visualization is intentionally
 * shape-only.
 */

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
  if (!isTauri()) return EMPTY;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return (await invoke("graph_snapshot", { userId })) as GraphSnapshot;
  } catch {
    return EMPTY;
  }
}
