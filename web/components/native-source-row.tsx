"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import {
  NATIVE_INGEST,
  isTauri,
  openFullDiskAccessSettings,
} from "@/lib/runtime";

type Status =
  | "idle"
  | "checking"
  | "permission_denied"
  | "ready"
  | "ingesting"
  | "connected"
  | "error"
  | "not_found";

export interface NativeSourceRowProps {
  label: string;
  kind: string; // matches a key in NATIVE_INGEST
  userId: string;
  onChange?: (count: number) => void;
}

/**
 * Row variant for sources that ingest via Tauri commands (no file picker).
 * Three states matter:
 *   - permission_denied → "Grant Full Disk Access →" deep-links to System Settings
 *   - ready (N messages found) → "Connect" triggers ingest
 *   - connected (N items ingested) → small success state
 */
export function NativeSourceRow({
  label,
  kind,
  userId,
  onChange,
}: NativeSourceRowProps) {
  const [tauriReady, setTauriReady] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [messageCount, setMessageCount] = useState<number | null>(null);
  const [ingestedCount, setIngestedCount] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTauriReady(isTauri());
  }, []);

  const binding = NATIVE_INGEST[kind];
  if (!binding) {
    return null; // Not a native-eligible kind; caller should render upload row instead.
  }

  async function check() {
    if (!tauriReady) return;
    setStatus("checking");
    setError(null);
    try {
      const result = await binding!.status();
      if (result.error === "permission_denied") {
        setStatus("permission_denied");
        return;
      }
      if (result.error === "not_found") {
        setStatus("not_found");
        return;
      }
      if (!result.canRead) {
        setStatus("error");
        setError(result.error ?? "Couldn't read the source");
        return;
      }
      setMessageCount(result.count);
      setStatus("ready");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function ingest() {
    if (!tauriReady) return;
    setStatus("ingesting");
    setError(null);
    try {
      const result = await binding!.ingest(userId);
      setIngestedCount(result.items_ingested);
      setStatus("connected");
      onChange?.(result.items_ingested);
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function actionLabel(): string {
    switch (status) {
      case "idle":
        return "Check";
      case "checking":
        return "Checking…";
      case "permission_denied":
        return "Grant access →";
      case "ready":
        return messageCount !== null
          ? `Connect ${messageCount.toLocaleString()} messages`
          : "Connect";
      case "ingesting":
        return "Ingesting…";
      case "connected":
        return "Connected";
      case "not_found":
        return "Not found";
      case "error":
        return "Try again";
    }
  }

  async function onClick() {
    if (status === "permission_denied") {
      await openFullDiskAccessSettings();
      // Give the user time to grant, then re-check on next click
      setStatus("idle");
      return;
    }
    if (status === "ready") return ingest();
    if (status === "idle" || status === "error") return check();
  }

  const disabled =
    !tauriReady ||
    status === "checking" ||
    status === "ingesting" ||
    status === "connected" ||
    status === "not_found";

  return (
    <div className="border-b border-border last:border-b-0">
      <div className="flex items-center justify-between py-5">
        <div className="flex-1 min-w-0">
          <p className="text-[17px] text-foreground">{label}</p>
          {status === "connected" && (
            <p className="text-[13px] text-muted mt-0.5">
              {ingestedCount.toLocaleString()} items collected
            </p>
          )}
          {status === "permission_denied" && (
            <p className="text-[13px] text-muted mt-0.5">
              Grant Full Disk Access, then check again.
            </p>
          )}
          {status === "not_found" && (
            <p className="text-[13px] text-muted mt-0.5">
              No iMessage database found on this Mac.
            </p>
          )}
          {status === "error" && error && (
            <p className="text-[13px] text-foreground/70 mt-0.5 truncate">
              {error}. Try again?
            </p>
          )}
          {!tauriReady && (
            <p className="text-[13px] text-muted mt-0.5">
              Open the Mac app to connect this source.
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onClick}
          disabled={disabled}
          className={cn(
            "text-[15px] transition-colors",
            disabled
              ? "text-muted cursor-not-allowed"
              : "text-foreground hover:text-muted",
          )}
        >
          {actionLabel()}
        </button>
      </div>
    </div>
  );
}
