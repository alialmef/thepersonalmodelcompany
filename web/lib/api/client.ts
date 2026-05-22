/**
 * Thin client for the PMC backend (the FastAPI we built in pmc.serve.api).
 *
 * In dev, requests go through Next.js rewrites (/pmc-api/* → http://localhost:8000)
 * so cookies and CORS Just Work. In production, set PMC_API_URL on the server.
 */

const SERVER_BASE = process.env.PMC_API_URL ?? "http://localhost:8000";
const PUBLIC_BASE =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";
const BROWSER_BASE = "/pmc-api";

function baseUrl(): string {
  // On the server (route handlers, RSC), hit the backend directly.
  // In the browser, go through the rewrite so credentials work cleanly.
  // Tauri static export has no Next rewrites, so desktop calls FastAPI directly.
  if (typeof window === "undefined") return SERVER_BASE;
  return isTauriRuntime() ? PUBLIC_BASE : BROWSER_BASE;
}

function isTauriRuntime(): boolean {
  const w = window as unknown as Record<string, unknown>;
  return !!(
    w.__TAURI_INTERNALS__ ||
    w.__TAURI__ ||
    w.__TAURI_METADATA__ ||
    w.isTauri
  );
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

export type RuntimeCapabilities = {
  training: {
    provider: string;
    forced: string | null;
    available: boolean;
    unavailable_reason: string | null;
    together_key_present: boolean;
    mlx_available: boolean;
  };
  inference: {
    provider: string;
    engine: string;
    base_model: string;
  };
  memory: {
    openai_key_present: boolean;
  };
  supervision: {
    anthropic_key_present: boolean;
  };
};

export async function getRuntimeCapabilities(): Promise<RuntimeCapabilities> {
  const r = await fetch(`${baseUrl()}/v1/runtime/capabilities`, {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`getRuntimeCapabilities failed: ${r.status}`);
  return r.json();
}

// ---------- Private verification ----------

export type TrustReport = {
  user_id: string;
  total_probes: number;
  total_judgments: number;
  total_action_traces: number;
  voice_approved: number;
  voice_total: number;
  action_approved: number;
  action_total: number;
  privacy_flags: number;
  scores: Record<string, number>;
  readiness: "unproven" | "voice" | "sandbox" | "supervised";
  generated_at: string;
};

export type EvalCandidate = {
  id: string;
  origin: string;
  text: string;
  model: string | null;
};

export type EvalPrompt = {
  id: string;
  kind: string;
  situation: string;
  response: string;
  reference?: string | null;
  source_completion_id?: string | null;
  dataset_version?: string | null;
  candidates: EvalCandidate[];
};

export type EvalPromptsResponse = {
  prompts: EvalPrompt[];
  trust_report: TrustReport;
};

export async function getEvalPrompts(userId: string): Promise<EvalPromptsResponse> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(userId)}/eval/prompts`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`getEvalPrompts failed: ${r.status}`);
  return r.json();
}

export async function submitEvalJudgment(opts: {
  userId: string;
  probeId: string;
  verdict: "approve" | "reject" | "edit" | "not_me" | "private" | "wrong" | "unsure";
  chosenCandidateId?: string;
  rejectedCandidateIds?: string[];
  editedText?: string;
  reason?: string;
  dimension?: string;
}): Promise<{ ok: boolean; trust_report: TrustReport }> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/eval/judgments`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        probeId: opts.probeId,
        verdict: opts.verdict,
        chosenCandidateId: opts.chosenCandidateId,
        rejectedCandidateIds: opts.rejectedCandidateIds,
        editedText: opts.editedText,
        reason: opts.reason,
        dimension: opts.dimension ?? "voice",
      }),
    },
  );
  if (!r.ok) throw new Error(`submitEvalJudgment failed: ${r.status}`);
  return r.json();
}

export async function promoteRun(
  userId: string,
  runId: string,
): Promise<{ ok: boolean; registered: boolean; trust_report: TrustReport }> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(userId)}/runs/${encodeURIComponent(runId)}/promote`,
    { method: "POST" },
  );
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`promoteRun failed: ${r.status} ${detail}`);
  }
  return r.json();
}

// ---------- Action sandbox ----------

export type ActionRisk = "low" | "medium" | "high";
export type ActionProposalStatus =
  | "proposed"
  | "approved"
  | "edited"
  | "rejected"
  | "executed"
  | "undone"
  | "expired";

export type ActionProposal = {
  id: string;
  user_id: string;
  surface: string;
  operation: string;
  proposed_text: string;
  proposed_payload: Record<string, unknown>;
  rationale: string;
  required_capability: string;
  risk_level: ActionRisk;
  status: ActionProposalStatus;
  model?: string | null;
  run_id?: string | null;
  created_at: string;
  reviewed_at?: string | null;
  metadata: Record<string, unknown>;
};

export type ActionReviewInfo = {
  execution_allowed: boolean;
  requires_confirmation: boolean;
  readiness_required: "unproven" | "sandbox" | "supervised";
};

export type ActionCapability = {
  key: string;
  surface: string;
  operation: string;
  risk_level: ActionRisk;
  supports_simulate: boolean;
  supports_stage: boolean;
  supports_execute: boolean;
  supports_undo: boolean;
  requires_confirmation: boolean;
  description: string;
};

export type ActionExecutionMode = "simulate" | "stage" | "execute" | "undo";

export type ActionExecutionReceipt = {
  id: string;
  user_id: string;
  proposal_id: string;
  surface: string;
  operation: string;
  mode: ActionExecutionMode;
  ok: boolean;
  preview: string;
  evidence: Record<string, unknown>;
  side_effects: string[];
  undo_token?: string | null;
  error?: string | null;
  created_at: string;
};

export async function listActionCapabilities(
  userId: string,
): Promise<{ capabilities: ActionCapability[]; trust_report: TrustReport }> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(userId)}/actions/capabilities`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`listActionCapabilities failed: ${r.status}`);
  return r.json();
}

export async function createActionProposal(opts: {
  userId: string;
  surface: string;
  operation: string;
  prompt?: string;
  proposedText?: string;
  proposedPayload?: Record<string, unknown>;
  rationale?: string;
  riskLevel?: ActionRisk;
}): Promise<{
  ok: boolean;
  proposal: ActionProposal;
  review: ActionReviewInfo;
  trust_report: TrustReport;
}> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/actions/proposals`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        surface: opts.surface,
        operation: opts.operation,
        prompt: opts.prompt,
        proposed_text: opts.proposedText,
        proposed_payload: opts.proposedPayload ?? {},
        rationale: opts.rationale,
        risk_level: opts.riskLevel,
      }),
    },
  );
  if (!r.ok) throw new Error(`createActionProposal failed: ${r.status}`);
  return r.json();
}

export async function listActionProposals(opts: {
  userId: string;
  status?: ActionProposalStatus;
  limit?: number;
}): Promise<{ proposals: ActionProposal[]; trust_report: TrustReport }> {
  const params = new URLSearchParams();
  if (opts.status) params.set("status", opts.status);
  if (opts.limit) params.set("limit", String(opts.limit));
  const qs = params.toString();
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/actions/proposals${qs ? `?${qs}` : ""}`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`listActionProposals failed: ${r.status}`);
  return r.json();
}

export async function runActionProposal(opts: {
  userId: string;
  proposalId: string;
  mode: ActionExecutionMode;
  payload?: Record<string, unknown>;
}): Promise<{
  ok: boolean;
  proposal: ActionProposal;
  receipt: ActionExecutionReceipt;
  review: ActionReviewInfo;
  trust_report: TrustReport;
}> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/actions/proposals/${encodeURIComponent(opts.proposalId)}/${opts.mode}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts.payload ?? {}),
    },
  );
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`runActionProposal failed: ${r.status} ${detail}`);
  }
  return r.json();
}

export async function reviewActionProposal(opts: {
  userId: string;
  proposalId: string;
  decision: "approved" | "edited" | "rejected" | "undone" | "ignored";
  editedText?: string;
  finalPayload?: Record<string, unknown>;
}): Promise<{
  ok: boolean;
  proposal: ActionProposal;
  review: ActionReviewInfo;
  trust_report: TrustReport;
}> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/actions/proposals/${encodeURIComponent(opts.proposalId)}/review`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision: opts.decision,
        edited_text: opts.editedText,
        final_payload: opts.finalPayload,
      }),
    },
  );
  if (!r.ok) throw new Error(`reviewActionProposal failed: ${r.status}`);
  return r.json();
}

// ---------- Laptop world ----------

export type WorldScanReport = {
  id: string;
  user_id: string;
  roots: string[];
  full_disk_requested: boolean;
  files_seen: number;
  files_indexed: number;
  dirs_skipped: number;
  bytes_indexed: number;
  errors: string[];
  started_at: string;
  finished_at: string;
};

export type WorldFile = {
  id: string;
  user_id: string;
  path: string;
  name: string;
  extension: string;
  kind: string;
  size_bytes: number;
  modified_at?: string | null;
  indexed_at: string;
  content_preview: string;
  metadata: Record<string, unknown>;
};

export async function scanWorld(opts: {
  userId: string;
  roots?: string[];
  fullDisk?: boolean;
  maxFiles?: number;
}): Promise<{ ok: boolean; scan: WorldScanReport; indexed: number }> {
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/world/scan`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        roots: opts.roots ?? [],
        full_disk: opts.fullDisk ?? true,
        max_files: opts.maxFiles ?? 2000,
      }),
    },
  );
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`scanWorld failed: ${r.status} ${detail}`);
  }
  return r.json();
}

export async function listWorldFiles(opts: {
  userId: string;
  query?: string;
  limit?: number;
}): Promise<{ files: WorldFile[]; latest_scan: WorldScanReport | null }> {
  const params = new URLSearchParams();
  if (opts.query) params.set("query", opts.query);
  if (opts.limit) params.set("limit", String(opts.limit));
  const qs = params.toString();
  const r = await fetch(
    `${baseUrl()}/v1/users/${encodeURIComponent(opts.userId)}/world/files${qs ? `?${qs}` : ""}`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`listWorldFiles failed: ${r.status}`);
  return r.json();
}
