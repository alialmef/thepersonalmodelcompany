"use client";

import { useEffect, useState } from "react";

/**
 * Tiny ambient indicator showing which providers the backend is
 * actually wired to. Polls `/v1/runtime/capabilities` once on mount
 * and renders a single low-contrast line in a fixed corner:
 *
 *     ● training: together · inference: together
 *
 * Why this exists: it's surprisingly easy to think a user is hitting
 * the Together-hosted pipeline when they're actually getting MLX-
 * Llama-3B locally (or vice versa). The pulse makes the answer
 * visible without taking attention.
 *
 * Color: red dot if a critical provider is missing (training mocked,
 * inference mocked), foreground-tone dot otherwise.
 */

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

interface Capabilities {
  training?: {
    provider?: string;
    available?: boolean;
    forced?: string | null;
  };
  inference?: {
    provider?: string;
    engine?: string;
  };
}

function shortProvider(p?: string): string {
  if (!p) return "?";
  if (p === "together") return "together";
  if (p === "mlx") return "mlx";
  if (p === "mock") return "mock";
  return p;
}

function isProduction(caps: Capabilities): boolean {
  return caps.training?.provider === "together" && caps.inference?.provider === "together";
}

function hasIssue(caps: Capabilities): boolean {
  return caps.inference?.provider === "mock" || caps.training?.available === false;
}

export default function CapabilityPulse() {
  const [caps, setCaps] = useState<Capabilities | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${PMC_API_URL}/v1/runtime/capabilities`, {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    })
      .then(async (res) => {
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as Capabilities;
        if (!cancelled) setCaps(data);
      })
      .catch(() => {
        /* offline — don't render */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!caps) return null;

  const issue = hasIssue(caps);
  const dotColor = issue
    ? "bg-red-500/70"
    : isProduction(caps)
      ? "bg-foreground/35"
      : "bg-amber-500/55";

  return (
    <div
      className="pointer-events-none fixed bottom-3 right-3 z-50 flex items-center gap-2 text-[10px] lowercase tracking-wide text-foreground/30"
      title={`training: ${caps.training?.provider ?? "?"} · inference: ${caps.inference?.engine ?? "?"}`}
    >
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${dotColor}`} />
      <span>
        training: {shortProvider(caps.training?.provider)} · inference:{" "}
        {shortProvider(caps.inference?.provider)}
      </span>
    </div>
  );
}
