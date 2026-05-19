"use client";

import { useState, useRef, type ChangeEvent } from "react";
import { uploadSource, type SourceKind } from "@/lib/api/client";
import { cn } from "@/lib/utils";

type Status = "idle" | "uploading" | "connected" | "error";

export interface SourceRowProps {
  label: string;
  description: string;
  kind: SourceKind;
  accept: string;
  userId: string;
  onChange: () => void;
  // For mbox / whatsapp the backend needs identity info from the user
  identityPrompt?: {
    label: string;
    placeholder: string;
    field: "userEmails" | "userNames";
  };
  // For iMessage: show instructions instead of a file input
  instructions?: string;
}

/**
 * One row in the Connect screen. The row holds its own upload state — a global
 * "items collected" count lives on the parent.
 */
export function SourceRow({
  label,
  description,
  kind,
  accept,
  userId,
  onChange,
  identityPrompt,
  instructions,
}: SourceRowProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState<number>(0);
  const [identityValue, setIdentityValue] = useState("");
  const [showIdentity, setShowIdentity] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);

  async function onFileSelected(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (identityPrompt && !identityValue.trim()) {
      setShowIdentity(true);
      return;
    }

    setStatus("uploading");
    setError(null);
    try {
      const opts: Parameters<typeof uploadSource>[0] = {
        userId,
        file,
        kind,
      };
      if (identityPrompt?.field === "userEmails") {
        opts.userEmails = identityValue.split(",").map((s) => s.trim()).filter(Boolean);
      } else if (identityPrompt?.field === "userNames") {
        opts.userNames = identityValue.split(",").map((s) => s.trim()).filter(Boolean);
      }
      const result = await uploadSource(opts);
      setStatus("connected");
      setCount((c) => c + result.raw_items_ingested);
      onChange();
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  function trigger() {
    if (instructions) {
      setShowInstructions((s) => !s);
      return;
    }
    if (identityPrompt && !identityValue.trim()) {
      setShowIdentity(true);
      return;
    }
    inputRef.current?.click();
  }

  return (
    <div className="border-b border-border last:border-b-0">
      <div className="flex items-center justify-between py-5">
        <div className="flex-1 min-w-0">
          <p className="text-[17px] text-foreground">{label}</p>
          {status === "connected" && count > 0 && (
            <p className="text-[13px] text-muted mt-0.5">
              {count.toLocaleString()} items collected
            </p>
          )}
          {status === "error" && (
            <p className="text-[13px] text-foreground/70 mt-0.5">
              {error}. Try again?
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={trigger}
          disabled={status === "uploading"}
          className={cn(
            "text-[15px] transition-colors",
            status === "uploading"
              ? "text-muted cursor-wait"
              : "text-foreground hover:text-muted",
          )}
        >
          {status === "uploading" ? "Uploading…" : description}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={onFileSelected}
          className="hidden"
        />
      </div>

      {showIdentity && identityPrompt && (
        <div className="pb-5 space-y-3">
          <input
            type="text"
            value={identityValue}
            onChange={(e) => setIdentityValue(e.target.value)}
            placeholder={identityPrompt.placeholder}
            className="w-full h-10 px-3 rounded-lg bg-subtle text-foreground placeholder:text-muted/70 outline-none border border-border focus:border-foreground/40 text-[14px]"
          />
          <div className="flex items-center justify-between">
            <p className="text-[13px] text-muted">{identityPrompt.label}</p>
            <button
              type="button"
              onClick={() => {
                if (identityValue.trim()) {
                  setShowIdentity(false);
                  inputRef.current?.click();
                }
              }}
              className="text-[14px] text-foreground hover:text-muted"
            >
              Continue →
            </button>
          </div>
        </div>
      )}

      {showInstructions && instructions && (
        <div className="pb-5">
          <p className="text-[14px] text-muted leading-relaxed whitespace-pre-line">
            {instructions}
          </p>
        </div>
      )}
    </div>
  );
}
