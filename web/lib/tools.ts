/**
 * Tools available to the chat model.
 *
 * Defined natively against the AI SDK's tool() helper for V0 simplicity.
 * The shape (name → description → input schema → execute) is the same shape
 * MCP servers expose, so swapping to external MCP-discovered tools later
 * is a contained change — see comment in chat/route.ts.
 *
 * Three starter tools, all no-auth, all useful immediately:
 *
 *   - current_time:   "what day is it?" / "what time is it in tokyo?"
 *   - recall_memory:  semantic search of the user's own writing
 *   - web_fetch:      fetch + summarize a URL the user mentions
 *
 * Each tool returns a small JSON object the model can reason over.
 */

import { tool } from "ai";
import { z } from "zod";
import { pmc, backendReachable } from "@/lib/pmc-client";
import { DEMO_USER_ID } from "@/lib/demo-user";

const PMC_API_URL = process.env.PMC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// current_time — the simplest possible tool. Useful for "what day is it?",
// "is it past noon for me?", "how long until thursday?", etc.
// ---------------------------------------------------------------------------

const currentTime = tool({
  description:
    "Get the current date and time. Returns ISO 8601 in UTC plus the day of week.",
  inputSchema: z.object({
    timezone: z
      .string()
      .optional()
      .describe(
        "Optional IANA timezone (e.g. 'America/New_York'). Defaults to UTC.",
      ),
  }),
  execute: async ({ timezone }) => {
    const now = new Date();
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone ?? "UTC",
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: false,
    });
    return {
      iso: now.toISOString(),
      readable: formatter.format(now),
      timezone: timezone ?? "UTC",
    };
  },
});

// ---------------------------------------------------------------------------
// recall_memory — semantic search of the user's own writing.
// Hits the PMC backend's retrieval endpoint. Useful when the user asks
// "what did I say about X?" beyond what's already in the system prompt's
// pre-loaded context.
// ---------------------------------------------------------------------------

interface RecallResult {
  source: string;
  text: string;
  when?: string;
}

const recallMemory = tool({
  description:
    "Search the user's past writing for relevant snippets. Use this when the " +
    "user asks about something they wrote, said, or planned that isn't in " +
    "the immediate conversation context.",
  inputSchema: z.object({
    query: z
      .string()
      .describe("A natural-language search query — what to look for in past writing"),
    k: z.number().int().min(1).max(20).optional().describe("Max results (default 5)"),
  }),
  execute: async ({ query, k = 5 }) => {
    if (!(await backendReachable())) {
      return { results: [], note: "backend offline — no recall available" };
    }
    try {
      const res = await fetch(
        `${PMC_API_URL}/v1/users/${encodeURIComponent(DEMO_USER_ID)}/memory/search`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ query, k }),
        },
      );
      if (!res.ok) {
        return { results: [], note: `recall failed (HTTP ${res.status})` };
      }
      const data = (await res.json()) as { results: RecallResult[] };
      return { results: data.results };
    } catch (e) {
      return {
        results: [],
        note: `recall error: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  },
});

// ---------------------------------------------------------------------------
// web_fetch — fetch a URL and return the visible text.
// For "look at this link and summarize" style requests. Caps response size
// so the model doesn't blow its context window on a heavy page.
// ---------------------------------------------------------------------------

const webFetch = tool({
  description:
    "Fetch a URL and return its visible text content. Use when the user " +
    "asks about a link or wants you to look at something on the web.",
  inputSchema: z.object({
    url: z.string().url().describe("The URL to fetch (must be https)"),
  }),
  execute: async ({ url }) => {
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
        return { ok: false, error: "only http(s) URLs allowed" };
      }
      const res = await fetch(url, {
        headers: { "user-agent": "PMC/0.1 (+https://thepersonalmodelcompany.com)" },
        signal: AbortSignal.timeout(10_000),
      });
      if (!res.ok) {
        return { ok: false, error: `HTTP ${res.status}` };
      }
      const html = await res.text();
      // Cheap HTML → text. Good enough for V0; swap for Readability or
      // jsdom + sanitize-html when we care about quality.
      const text = html
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 6000);
      return { ok: true, url, text };
    } catch (e) {
      return {
        ok: false,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  },
});

// ---------------------------------------------------------------------------
// Export bundle — passed to streamText({ tools: ... }) in the chat route.
// Names stay stable as tool identifiers visible to the model.
// ---------------------------------------------------------------------------

export const chatTools = {
  current_time: currentTime,
  recall_memory: recallMemory,
  web_fetch: webFetch,
};

// Unused — kept here to satisfy the import-graph optimizer + signal intent.
export { pmc };
