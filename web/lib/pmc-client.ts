/**
 * AI SDK provider pointed at the PMC backend.
 *
 * The backend exposes an OpenAI-compatible /v1/chat/completions endpoint
 * (see pmc/serve/api.py). The "model" identifier we pass IS the user_id —
 * the backend uses it to route to the right adapter and inject that user's
 * memory + identity prompt.
 */

import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

const PMC_API_URL = process.env.PMC_API_URL ?? "http://localhost:8000";

export const pmc = createOpenAICompatible({
  name: "pmc",
  baseURL: `${PMC_API_URL}/v1`,
  // Backend doesn't enforce auth in V0 — placeholder satisfies the SDK contract.
  apiKey: process.env.PMC_API_KEY ?? "local-dev",
});

/**
 * Returns true if the backend appears reachable. Used by the chat route to
 * fall back to a graceful "not connected" message instead of throwing.
 *
 * Cheap GET against /healthz with a tight timeout — the backend mounts this
 * unconditionally in pmc/serve/api.py.
 */
export async function backendReachable(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${PMC_API_URL}/healthz`, {
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}
