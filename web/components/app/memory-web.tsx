"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";

import { graphSnapshot, type GraphEdge, type GraphNode } from "@/lib/api/graph";

/**
 * MemoryWeb — the /reading screen's web-of-memory visualization.
 *
 * Polls graph_snapshot every few seconds. Renders nodes + edges with a
 * d3-force layout in SVG. New nodes fade in over ~600ms as they appear
 * across polls; old nodes stay put. The whole thing is shape only —
 * no labels, no tooltips, no zoom. The goal is to feel a structure
 * being built, not to be a graph explorer.
 *
 * Reports `onStable(true)` when no new nodes have arrived for ~6
 * consecutive polls AND there's at least one node. The /reading page
 * uses that to advance to /confirm.
 */

interface Props {
  userId: string;
  pollIntervalMs?: number;
  className?: string;
  /** Fires once with true when the graph appears done growing. */
  onStable?: (stable: boolean) => void;
}

interface SimNode extends SimulationNodeDatum {
  id: string;
  kind: string;
  weight?: number;
  bornAt: number;
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  id: string;
  kind: string;
  bornAt: number;
}

const W = 720;
const H = 520;
const STABLE_POLLS = 6;

// Calm palette — every kind reads as the same conceptual family. The
// hue shifts are minimal; node SIZE + LAYOUT is what carries the
// distinction, not bright colors.
const KIND_COLOR: Record<string, string> = {
  person:       "#2D2D2D",
  place:        "#5A5A5A",
  project:      "#404040",
  theme:        "#1A1A1A",
  event:        "#666666",
  episode:      "#777777",
  open_loop:    "#8A6E3F",
  taste:        "#7A6B5C",
  file:         "#9A9A9A",
  repo:         "#5C6B7A",
  web:          "#8A9A8A",
  app:          "#888888",
  shell:        "#6A6A6A",
  notification: "#A89090",
};

function colorFor(kind: string): string {
  return KIND_COLOR[kind] ?? "#9A9A9A";
}

function baseRadius(kind: string): number {
  // Person + theme are the conceptual anchors → slightly bigger.
  if (kind === "person" || kind === "theme") return 5;
  if (kind === "place" || kind === "project") return 4.5;
  return 3.5;
}

export default function MemoryWeb({
  userId,
  pollIntervalMs = 4_000,
  className,
  onStable,
}: Props) {
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [links, setLinks] = useState<SimLink[]>([]);
  const [, forceRender] = useState(0);
  const stableCountRef = useRef(0);
  const lastNodeCountRef = useRef(0);

  // Poll the Tauri-side graph snapshot. Merge into the existing sim
  // state so newly-arrived nodes keep their `bornAt` timestamp for the
  // fade-in animation, and existing nodes keep their layout positions.
  useEffect(() => {
    if (!userId) return;
    let cancelled = false;
    let stableSent = false;

    const pull = async () => {
      const snap = await graphSnapshot(userId);
      if (cancelled) return;

      const now = Date.now();
      setNodes((prev) => {
        const prevById = new Map(prev.map((n) => [n.id, n]));
        const merged: SimNode[] = snap.nodes.map((n) => {
          const existing = prevById.get(n.id);
          if (existing) {
            existing.kind = n.kind;
            existing.weight = n.weight;
            return existing;
          }
          // New node — drop near center with a tiny random offset so
          // the simulation has something to work with.
          return {
            id: n.id,
            kind: n.kind,
            weight: n.weight,
            bornAt: now,
            x: W / 2 + (Math.random() - 0.5) * 20,
            y: H / 2 + (Math.random() - 0.5) * 20,
          };
        });
        return merged;
      });
      setLinks((prev) => {
        const prevById = new Map(prev.map((l) => [l.id, l]));
        return snap.edges.map((e) => {
          const existing = prevById.get(e.id);
          if (existing) return existing;
          return {
            id: e.id,
            kind: e.kind,
            source: e.source,
            target: e.target,
            bornAt: now,
          };
        });
      });

      // Stability: same node count across N consecutive polls AND > 0
      if (snap.nodes.length === lastNodeCountRef.current && snap.nodes.length > 0) {
        stableCountRef.current += 1;
      } else {
        stableCountRef.current = 0;
      }
      lastNodeCountRef.current = snap.nodes.length;
      if (stableCountRef.current >= STABLE_POLLS && !stableSent) {
        stableSent = true;
        onStable?.(true);
      }
    };

    pull();
    const id = setInterval(pull, pollIntervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [userId, pollIntervalMs, onStable]);

  // d3-force simulation. We restart on node-set change but keep the
  // alpha low so existing positions stay mostly put — feels like new
  // nodes are joining an already-coalesced structure, not a whole
  // re-layout each tick.
  const simRef = useRef<ReturnType<typeof forceSimulation<SimNode>> | null>(null);
  useEffect(() => {
    if (nodes.length === 0) {
      simRef.current?.stop();
      simRef.current = null;
      return;
    }
    const sim = forceSimulation<SimNode>(nodes)
      .force("charge", forceManyBody().strength(-30))
      .force("center", forceCenter(W / 2, H / 2).strength(0.05))
      .force(
        "collide",
        forceCollide<SimNode>().radius((n) => baseRadius(n.kind) + 3),
      )
      .force(
        "link",
        forceLink<SimNode, SimLink>(links)
          .id((n) => n.id)
          .distance(45)
          .strength(0.3),
      )
      .alpha(0.6)
      .alphaDecay(0.04)
      .on("tick", () => forceRender((n) => n + 1));

    simRef.current = sim;
    return () => {
      sim.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length, links.length]);

  // Animate node fade-in. We re-render every ~80ms while any node is
  // still in its 600ms birth window, so opacity transitions smoothly.
  const renderNow = useFadeTick(nodes);

  const renderedLinks = useMemo(() => {
    return links.map((l) => {
      const s = typeof l.source === "object" ? (l.source as SimNode) : null;
      const t = typeof l.target === "object" ? (l.target as SimNode) : null;
      if (!s || !t || s.x === undefined || t.x === undefined) return null;
      const age = renderNow - l.bornAt;
      const opacity = Math.min(0.18, age / 600 * 0.18);
      return (
        <line
          key={l.id}
          x1={s.x}
          y1={s.y!}
          x2={t.x}
          y2={t.y!}
          stroke="#000000"
          strokeOpacity={opacity}
          strokeWidth={0.5}
        />
      );
    });
  }, [links, renderNow]);

  const renderedNodes = useMemo(() => {
    return nodes.map((n) => {
      if (n.x === undefined || n.y === undefined) return null;
      const age = renderNow - n.bornAt;
      const opacity = Math.min(1, age / 600);
      const r = baseRadius(n.kind) + (n.weight ?? 0) * 3;
      return (
        <circle
          key={n.id}
          cx={n.x}
          cy={n.y}
          r={r}
          fill={colorFor(n.kind)}
          fillOpacity={opacity * 0.9}
        />
      );
    });
  }, [nodes, renderNow]);

  return (
    <div className={className}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{ display: "block" }}
      >
        <g>{renderedLinks}</g>
        <g>{renderedNodes}</g>
      </svg>
    </div>
  );
}

/** Re-render at ~12fps while any node is still in its 600ms birth window,
 *  so the fade-in is smooth. Returns the current Date.now(). */
function useFadeTick(nodes: SimNode[]): number {
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const anyAnimating = () =>
      nodes.some((n) => Date.now() - n.bornAt < 700);
    if (!anyAnimating()) {
      setNow(Date.now());
      return;
    }
    const id = setInterval(() => {
      setNow(Date.now());
      if (!anyAnimating()) clearInterval(id);
    }, 80);
    return () => clearInterval(id);
  }, [nodes]);
  return now;
}
