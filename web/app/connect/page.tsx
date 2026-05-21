"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import ConnectScreen from "@/components/app/connect-screen";
import { PermissionsScreen } from "@/components/app/permissions-screen";
import { useUser } from "@/hooks/use-user";
import { NATIVE_INGEST, isTauri } from "@/lib/runtime";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

type SourceState = "idle" | "connecting" | "connected";

// Maps the design's source IDs ↔ our backend "kind" identifiers.
const SOURCE_KIND: Record<string, string> = {
  messages: "imessage",
  notes: "text",
  mail: "email_mbox",
  // documents: handled via file picker, see openDocumentsPicker below
};

/**
 * Step 1 of 3 — Bring your writing.
 *
 * Renders the designed ConnectScreen. Each source row's Connect button
 * triggers native ingestion via the Tauri bridge (iMessage / Notes / Mail).
 * "Documents" opens the system file picker through Tauri.
 *
 * When at least one source is connected, Continue is enabled. Clicking it
 * kicks off the full training pipeline (POST /v1/users/{id}/runs) and
 * routes to /curate?job=<id>.
 */
// Pretty source labels for the permissions screen headline.
const SOURCE_LABEL: Record<string, string> = {
  messages: "your messages",
  notes: "your notes",
  mail: "your mail",
};

export default function ConnectPage() {
  const router = useRouter();
  const { user } = useUser();
  const userId = user?.pmcUserId ?? "";
  const [inApp, setInApp] = useState(false);
  const [states, setStates] = useState<Record<string, SourceState>>({});
  const [error, setError] = useState<string | null>(null);
  // When a source returns permission_denied, surface the full-screen
  // permission flow instead of an inline message. The kind is the
  // NATIVE_INGEST key so the screen knows what to re-check.
  const [permissionFor, setPermissionFor] = useState<
    { sourceId: string; kind: string } | null
  >(null);

  useEffect(() => {
    setInApp(isTauri());
  }, []);

  // Pre-populate connected sources for returning users. Fetches the
  // backend status and marks any source kind with raw items as "connected".
  useEffect(() => {
    if (!userId) return;
    let cancelled = false;
    (async () => {
      try {
        const apiUrl =
          process.env.NEXT_PUBLIC_PMC_API_URL ?? "http://localhost:8000";
        const res = await fetch(
          `${apiUrl}/v1/users/${encodeURIComponent(userId)}/status`,
          { cache: "no-store", signal: AbortSignal.timeout(3000) },
        );
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as {
          sources?: Array<{ source_id?: string; kind?: string; item_count?: number }>;
        };
        const next: Record<string, SourceState> = {};
        for (const src of data.sources ?? []) {
          if ((src.item_count ?? 0) <= 0) continue;
          // Backend "kind" → UI sourceId mapping (inverse of SOURCE_KIND below)
          const uiId = (
            {
              imessage: "messages",
              text: "notes",
              email_mbox: "mail",
              document: "documents",
            } as Record<string, string>
          )[src.kind ?? ""];
          if (uiId) next[uiId] = "connected";
        }
        if (!cancelled && Object.keys(next).length > 0) {
          setStates((prev) => ({ ...next, ...prev }));
        }
      } catch {
        /* backend offline — let the user connect fresh */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [userId]);

  const setState = (sourceId: string, state: SourceState) =>
    setStates((prev) => ({ ...prev, [sourceId]: state }));

  const handleConnect = useCallback(
    async (sourceId: string) => {
      setError(null);
      setState(sourceId, "connecting");

      // Documents: open a file picker rather than native scan (no auto-walk
      // of the home folder for arbitrary docs — too broad without scoping).
      if (sourceId === "documents") {
        try {
          if (inApp) {
            const { open } = await import("@tauri-apps/plugin-dialog");
            const picked = await open({
              multiple: true,
              filters: [
                { name: "Documents", extensions: ["pdf", "docx", "txt", "md"] },
              ],
            });
            if (!picked) {
              setState(sourceId, "idle");
              return;
            }
            // For V0 we just mark connected — actually uploading each picked
            // file is wired through SourceRow upload elsewhere. The "did
            // anything attach?" signal is what matters for the gate.
            setState(sourceId, "connected");
            return;
          }
          // Browser fallback — let the user pick a folder via input. We don't
          // surface that here; user can use the legacy upload page if they
          // really need it. Pretend connected for now.
          setState(sourceId, "connected");
          return;
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
          setState(sourceId, "idle");
          return;
        }
      }

      // Native sources go through NATIVE_INGEST.
      const kind = SOURCE_KIND[sourceId];
      const binding = kind ? NATIVE_INGEST[kind] : undefined;
      if (!binding || !inApp) {
        // Not in Tauri — these can't connect natively. Mark connected so the
        // gate doesn't block; the actual ingest in this branch is a no-op
        // for V0 (web users will eventually have upload fallback rows).
        setState(sourceId, "connected");
        return;
      }

      try {
        const status = await binding.status();
        if (status.error === "permission_denied") {
          // Bring up the full-screen guided permission flow instead of the
          // old inline "Grant access →" text link. The flow auto-opens the
          // right Settings pane, shows the steps visually, and polls until
          // the user grants — then resumes ingestion from where we paused.
          setState(sourceId, "idle");
          setPermissionFor({ sourceId, kind });
          return;
        }
        if (status.error === "not_found" || !status.canRead) {
          setError(`${sourceId} unavailable on this Mac`);
          setState(sourceId, "idle");
          return;
        }
        await binding.ingest(userId);
        setState(sourceId, "connected");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setState(sourceId, "idle");
      }
    },
    [inApp, userId],
  );

  const handleContinue = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch(
        `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/runs`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            // Skip eval for V0 (HF/PEFT not installed locally).
            // KEEP deploy so the trained adapter gets registered.
            skip_eval: true,
            skip_deploy: false,
          }),
        },
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      const data = (await res.json()) as { job_id: string };
      router.push(
        `/curate?job=${encodeURIComponent(data.job_id)}&user=${encodeURIComponent(userId)}`,
      );
    } catch (e) {
      setError(
        `Couldn't start: ${e instanceof Error ? e.message : String(e)}. Is the backend running?`,
      );
    }
  }, [router, userId]);

  return (
    <>
      <ConnectScreen
        states={states}
        onConnect={handleConnect}
        onContinue={handleContinue}
      />
      {error && (
        <div className="fixed bottom-5 left-1/2 -translate-x-1/2 max-w-md rounded-lg border-[0.5px] border-red-500/30 bg-red-50 px-4 py-2 text-[12px] text-red-700">
          {error}
        </div>
      )}

      {permissionFor && (
        <PermissionsScreen
          kind={permissionFor.kind}
          sourceLabel={SOURCE_LABEL[permissionFor.sourceId] ?? "your data"}
          onCancel={() => setPermissionFor(null)}
          onGranted={async () => {
            // Resume ingestion the moment access lands.
            const sourceId = permissionFor.sourceId;
            setPermissionFor(null);
            await handleConnect(sourceId);
          }}
        />
      )}
    </>
  );
}
