"use client";

import { withAuth } from "@/lib/api/auth";

/**
 * Client for the backend's /v1/agent/* endpoints — bring-your-own
 * frontier-model provider configuration.
 *
 * Settings flow:
 *   listProviders()        → catalog rendered in the provider picker
 *   getConfig()            → current saved choice (no key returned)
 *   validateKey()          → 'test connection' button
 *   setConfig()            → save provider+model+key (server validates first)
 *   clearConfig()          → forget the saved choice
 *
 * Chat (used by the right-now / chat flows once Phase 4 lands):
 *   chat()                 → non-streaming
 *   chatStream()           → SSE
 */

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

export interface ProviderInfo {
  id: string;
  label: string;
  default_models: string[];
  key_prefix_hint?: string;
  console_url?: string;
}

export interface AgentConfig {
  configured: boolean;
  provider?: string;
  model?: string;
  updated_at?: string;
  encryption_configured: boolean;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  text: string;
  model: string;
  usage: Record<string, number>;
  finish_reason?: string;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

export async function listProviders(): Promise<ProviderInfo[]> {
  const r = await fetch(`${PMC_API_URL}/v1/agent/providers`, {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const body = (await r.json()) as { providers: ProviderInfo[] };
  return body.providers;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export async function getConfig(): Promise<AgentConfig> {
  const r = await fetch(
    `${PMC_API_URL}/v1/agent/config`,
    withAuth({ cache: "no-store" }),
  );
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as AgentConfig;
}

export async function setConfig(args: {
  provider: string;
  model: string;
  api_key: string;
}): Promise<AgentConfig> {
  const r = await fetch(
    `${PMC_API_URL}/v1/agent/config`,
    withAuth({
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(args),
    }),
  );
  if (!r.ok) {
    const detail = await safeDetail(r);
    throw new Error(detail);
  }
  return (await r.json()) as AgentConfig;
}

export async function clearConfig(): Promise<void> {
  await fetch(
    `${PMC_API_URL}/v1/agent/config`,
    withAuth({ method: "DELETE" }),
  );
}

export async function validateKey(args: {
  provider: string;
  api_key: string;
}): Promise<boolean> {
  try {
    const r = await fetch(`${PMC_API_URL}/v1/agent/config/validate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(args),
    });
    if (!r.ok) return false;
    const body = (await r.json()) as { ok: boolean };
    return body.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export async function chat(args: {
  messages: ChatMessage[];
  system?: string;
  max_tokens?: number;
  model?: string;
}): Promise<ChatResponse> {
  const r = await fetch(
    `${PMC_API_URL}/v1/agent/chat`,
    withAuth({
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(args),
    }),
  );
  if (!r.ok) {
    const detail = await safeDetail(r);
    throw new Error(detail);
  }
  return (await r.json()) as ChatResponse;
}

/** Streaming chat. Yields raw text chunks as they arrive. */
export async function* chatStream(args: {
  messages: ChatMessage[];
  system?: string;
  max_tokens?: number;
  model?: string;
}): AsyncGenerator<string, void, unknown> {
  const r = await fetch(
    `${PMC_API_URL}/v1/agent/chat/stream`,
    withAuth({
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(args),
    }),
  );
  if (!r.ok || !r.body) {
    const detail = await safeDetail(r);
    throw new Error(detail);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    // SSE frames are separated by blank lines
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const dataLines = frame
        .split("\n")
        .filter((l) => l.startsWith("data: "))
        .map((l) => l.slice(6));
      if (dataLines.length === 0) continue;
      const payload = dataLines.join("\n");
      if (payload === "[DONE]") return;
      yield payload;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function safeDetail(r: Response): Promise<string> {
  try {
    const body = await r.json();
    if (typeof body === "object" && body !== null) {
      if ("detail" in body) {
        const d = (body as { detail: unknown }).detail;
        if (typeof d === "string") return d;
        if (typeof d === "object" && d !== null && "message" in d) {
          return String((d as { message: unknown }).message);
        }
        return JSON.stringify(d);
      }
    }
    return `HTTP ${r.status}`;
  } catch {
    return `HTTP ${r.status}`;
  }
}
