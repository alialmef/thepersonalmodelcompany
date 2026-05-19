/**
 * Runtime context detection — are we running inside the Tauri desktop app,
 * or in a regular browser tab on thepersonalmodelcompany.com?
 *
 * The marketing site uses this to swap the hero CTA between "Begin" (web)
 * and "Open the app" (Tauri). App routes use it to call Tauri commands
 * when available, and otherwise show a "Download the app" prompt for
 * Mac-only features (iMessage ingest, etc.).
 */

export const isTauri = (): boolean => {
  if (typeof window === "undefined") return false;
  return Boolean((window as { __TAURI__?: unknown }).__TAURI__);
};

export type AppInfo = {
  name: string;
  version: string;
  platform: string;
  backend_url: string;
};

/**
 * Call the Rust `app_info` command. Throws if not in Tauri context — guard
 * with `isTauri()` first.
 */
export async function getAppInfo(): Promise<AppInfo> {
  if (!isTauri()) {
    throw new Error("getAppInfo() called outside the Tauri runtime");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<AppInfo>("app_info");
}

/**
 * Health check — round-trip through Rust. Used during scaffold to verify
 * the JS↔Rust bridge is wired correctly.
 */
export async function ping(): Promise<string> {
  if (!isTauri()) return "(web context, no Tauri)";
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<string>("ping");
}

// ---------- Native ingestion commands (Tauri only) ----------

export type IMessageStatus = {
  chat_db_exists: boolean;
  can_read: boolean;
  message_count: number | null;
  error: string | null;
};

export type IngestSummary = {
  source: string;
  source_id: string;
  items_ingested: number;
};

/** Check whether chat.db is readable and how many messages are inside. */
export async function imessageStatus(): Promise<IMessageStatus> {
  if (!isTauri()) {
    throw new Error("imessageStatus() requires the Tauri runtime");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<IMessageStatus>("imessage_status");
}

/** Deep-link to System Settings → Privacy & Security → Full Disk Access. */
export async function openFullDiskAccessSettings(): Promise<void> {
  if (!isTauri()) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("open_full_disk_access_settings");
}

/**
 * Read all iMessage messages from chat.db, batch them, POST to the backend.
 * Optional `limit` for testing/dry-runs.
 */
export async function ingestIMessage(
  userId: string,
  limit?: number,
): Promise<IngestSummary> {
  if (!isTauri()) {
    throw new Error("ingestIMessage() requires the Tauri runtime");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<IngestSummary>("ingest_imessage", { userId, limit });
}

/**
 * Map of source kinds → Tauri command bindings for native ingestion. The
 * Connect page uses this to decide whether to render the native flow or the
 * upload fallback for each row.
 */
export const NATIVE_INGEST: Partial<
  Record<
    string,
    {
      status: () => Promise<{ canRead: boolean; count: number | null; error: string | null }>;
      ingest: (userId: string) => Promise<IngestSummary>;
    }
  >
> = {
  imessage: {
    async status() {
      const s = await imessageStatus();
      return { canRead: s.can_read, count: s.message_count, error: s.error };
    },
    async ingest(userId: string) {
      return ingestIMessage(userId);
    },
  },
  // notes, email_mbox, whatsapp will be added as their Rust modules ship
};
