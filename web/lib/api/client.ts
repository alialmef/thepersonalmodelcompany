/**
 * Thin client for the PMC backend (the FastAPI we built in pmc.serve.api).
 *
 * In dev, requests go through Next.js rewrites (/pmc-api/* → http://localhost:8000)
 * so cookies and CORS Just Work. In production, set PMC_API_URL on the server.
 */

const SERVER_BASE = process.env.PMC_API_URL ?? "http://localhost:8000";
const BROWSER_BASE = "/pmc-api";

function baseUrl(): string {
  // On the server (route handlers, RSC), hit the backend directly.
  // In the browser, go through the rewrite so credentials work cleanly.
  return typeof window === "undefined" ? SERVER_BASE : BROWSER_BASE;
}

export type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type ChatCompletionRequest = {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  user?: string;
};

export async function listModels(): Promise<{ data: Array<{ id: string; base_model: string }> }> {
  const r = await fetch(`${baseUrl()}/v1/models`, { cache: "no-store" });
  if (!r.ok) throw new Error(`listModels failed: ${r.status}`);
  return r.json();
}

export async function chat(req: ChatCompletionRequest): Promise<unknown> {
  const r = await fetch(`${baseUrl()}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...req, stream: false }),
  });
  if (!r.ok) throw new Error(`chat failed: ${r.status}`);
  return r.json();
}

/**
 * Streaming chat — yields response text deltas via SSE. Use in a React
 * Server Component or route handler, or wrap with EventSource on the client.
 */
export async function* chatStream(req: ChatCompletionRequest): AsyncGenerator<string> {
  const r = await fetch(`${baseUrl()}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...req, stream: true }),
  });
  if (!r.ok || !r.body) throw new Error(`chatStream failed: ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are `data: <json>\n\n`
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const line = event.trim();
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice("data: ".length);
      if (payload === "[DONE]") return;
      try {
        const chunk = JSON.parse(payload);
        const delta = chunk?.choices?.[0]?.delta?.content;
        if (typeof delta === "string" && delta.length > 0) yield delta;
      } catch {
        // ignore malformed lines; SSE is best-effort
      }
    }
  }
}

export async function exportBundle(userId: string): Promise<Response> {
  return fetch(`${baseUrl()}/v1/models/${encodeURIComponent(userId)}/export`);
}

export async function deleteModel(userId: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/v1/models/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`deleteModel failed: ${r.status}`);
}

// ---------- Web-app endpoints (require backend with storage_root) ----------

export type SourceKind =
  | "text"
  | "document"
  | "email_mbox"
  | "imessage"
  | "whatsapp";

export type UploadSourceResult = {
  raw_items_ingested: number;
  source_id: string;
  kind: SourceKind;
  total_raw_items: number;
};

export type UserStatus = {
  user_id: string;
  has_profile: boolean;
  total_runs: number;
  active_run_id: string | null;
  last_training_at: string | null;
  last_eval_scores: Record<string, number>;
  retrain_needed: boolean;
  pending_tombstones: number;
  raw_sources: string[];
  raw_item_count: number;
  dataset_versions: string[];
  registered_for_serving: boolean;
  recent_events: Array<{
    user_id: string;
    timestamp: string;
    stage: string;
    event: string;
    run_id: string | null;
    data: Record<string, unknown>;
  }>;
};

export async function uploadSource(opts: {
  userId: string;
  file: File;
  kind: SourceKind;
  sourceId?: string;
  userEmails?: string[];
  userNames?: string[];
}): Promise<UploadSourceResult> {
  const form = new FormData();
  form.append("file", opts.file);
  form.append("kind", opts.kind);
  if (opts.sourceId) form.append("source_id", opts.sourceId);
  if (opts.userEmails?.length) form.append("user_emails", opts.userEmails.join(","));
  if (opts.userNames?.length) form.append("user_names", opts.userNames.join(","));

  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/sources/upload`,
    { method: "POST", body: form },
  );
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`uploadSource failed: ${r.status} ${detail}`);
  }
  return r.json();
}

export async function getUserStatus(userId: string): Promise<UserStatus> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(userId)}/status`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`getUserStatus failed: ${r.status}`);
  return r.json();
}

export async function deleteSource(
  userId: string,
  sourceId: string,
): Promise<{ deleted_source: string; items_removed: number; retrain_needed: boolean }> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(userId)}/sources/${encodeURIComponent(sourceId)}`,
    { method: "DELETE" },
  );
  if (!r.ok) throw new Error(`deleteSource failed: ${r.status}`);
  return r.json();
}
