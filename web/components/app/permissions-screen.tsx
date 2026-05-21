"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowRight, Check } from "lucide-react";
import { BrandMark } from "@/components/shared/brand-mark";
import {
  NATIVE_INGEST,
  openFullDiskAccessSettings,
} from "@/lib/runtime";

/**
 * Full-screen guided permission flow for macOS Full Disk Access.
 *
 * Why we can't just "auto-grant": macOS treats FDA as a hard security
 * boundary. No API can flip the toggle from inside an app — Apple requires
 * the user to do it themselves in System Settings. Best we can do:
 *
 *   1. Deep-link to the exact Settings pane (x-apple.systempreferences URL)
 *   2. Show a single page with three obvious steps and a clear illustration
 *      of which switch to toggle
 *   3. Poll the underlying permission state every second so the moment the
 *      user grants, we auto-advance and continue the original action
 *
 * Triggered from connect-page when a native source returns permission_denied
 * for the first time. The kind is passed in so we can re-check the right
 * source after the user grants access.
 */

export function PermissionsScreen({
  kind,
  sourceLabel,
  onGranted,
  onCancel,
}: {
  /** The NATIVE_INGEST key — "imessage", "text", "email_mbox", etc. */
  kind: string;
  /** Human-readable name shown in the headline ("your texts", "your mail"). */
  sourceLabel: string;
  /** Called when the OS reports access has been granted. */
  onGranted: () => void;
  /** User backed out of the flow. */
  onCancel: () => void;
}) {
  const [opened, setOpened] = useState(false);
  const [checking, setChecking] = useState(false);
  const [granted, setGranted] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-open Settings the moment this screen appears. The OS shows
  // the right pane — we just have to show the user what to do there.
  useEffect(() => {
    let cancelled = false;
    openFullDiskAccessSettings().finally(() => {
      if (!cancelled) setOpened(true);
    });
    return () => {
      cancelled = true;
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Poll the permission state every second. The moment FDA is granted
  // (canRead becomes true and error is null), advance.
  useEffect(() => {
    const binding = NATIVE_INGEST[kind];
    if (!binding) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await binding.status();
        if (status.canRead && !status.error) {
          setGranted(true);
          if (pollRef.current) clearInterval(pollRef.current);
          // Tiny celebratory beat before continuing.
          setTimeout(onGranted, 800);
        }
      } catch {
        /* ignore polling errors */
      }
    }, 1000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [kind, onGranted]);

  async function recheckNow() {
    setChecking(true);
    const binding = NATIVE_INGEST[kind];
    if (!binding) {
      setChecking(false);
      return;
    }
    try {
      const status = await binding.status();
      if (status.canRead && !status.error) {
        setGranted(true);
        setTimeout(onGranted, 600);
        return;
      }
    } catch {
      /* */
    }
    setChecking(false);
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white px-7">
      {/* Mark */}
      <div className="mb-10">
        <BrandMark size={64} />
      </div>

      {/* Headline */}
      <h1
        className="mb-4 text-center font-medium leading-[1.05] tracking-[-0.025em] text-neutral-900"
        style={{ fontSize: "clamp(28px, 4vw, 44px)" }}
      >
        {granted ? "you're set." : `let your model read ${sourceLabel}.`}
      </h1>

      {/* Subhead */}
      <p
        className="mb-12 max-w-[40ch] text-center text-neutral-500"
        style={{ fontSize: "clamp(14px, 1.4vw, 17px)" }}
      >
        {granted
          ? "starting ingestion now."
          : "macOS keeps this gated by design. three taps unlock it forever."}
      </p>

      {/* Steps — only shown until granted */}
      {!granted && (
        <ol className="mb-12 flex w-full max-w-[460px] flex-col gap-[2px] overflow-hidden rounded-[12px] border-[0.5px] border-neutral-200 bg-neutral-200">
          <Step
            n={1}
            label="we just opened System Settings"
            detail="Privacy & Security → Full Disk Access"
            done={opened}
            action={
              !opened ? (
                <button
                  onClick={() => openFullDiskAccessSettings().then(() => setOpened(true))}
                  className="rounded-full bg-neutral-900 px-4 py-1.5 text-[12px] font-medium text-white hover:bg-neutral-800"
                >
                  Open it
                </button>
              ) : undefined
            }
          />
          <Step
            n={2}
            label="find Personal Model Company in the list"
            detail="it appears the first time you connect a source"
          />
          <Step
            n={3}
            label="flip the switch on"
            detail="we'll detect it instantly"
          />
        </ol>
      )}

      {/* Action row */}
      <div className="flex items-center gap-3">
        {granted ? (
          <div className="flex items-center gap-2 rounded-full bg-[#34C759] px-5 py-2.5 text-[13px] font-medium text-white">
            <Check className="size-4" strokeWidth={2.25} />
            Granted
          </div>
        ) : (
          <>
            <button
              onClick={onCancel}
              className="rounded-full px-5 py-2.5 text-[13px] font-medium text-neutral-500 hover:text-neutral-900"
            >
              Not now
            </button>
            <button
              onClick={recheckNow}
              disabled={checking}
              className="flex items-center gap-2 rounded-full bg-neutral-900 px-5 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-neutral-800 disabled:opacity-60"
            >
              {checking ? "Checking…" : "I've done it"}
              {!checking && <ArrowRight className="size-3.5" strokeWidth={2} />}
            </button>
          </>
        )}
      </div>

      {/* Quiet status — auto-polling indicator */}
      {!granted && (
        <p className="mt-8 text-[11px] text-neutral-400">
          we're watching for the change. you don't have to come back here to confirm.
        </p>
      )}
    </div>
  );
}

function Step({
  n,
  label,
  detail,
  done = false,
  action,
}: {
  n: number;
  label: string;
  detail: string;
  done?: boolean;
  action?: React.ReactNode;
}) {
  return (
    <li className="flex items-start gap-4 bg-white px-5 py-4">
      <div
        className={`mt-0.5 flex size-6 flex-shrink-0 items-center justify-center rounded-full text-[11px] font-medium ${
          done
            ? "bg-[#34C759] text-white"
            : "bg-neutral-100 text-neutral-500"
        }`}
      >
        {done ? <Check className="size-3" strokeWidth={2.5} /> : n}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[14px] font-medium leading-tight tracking-[-0.005em] text-neutral-900">
          {label}
        </div>
        <div className="mt-0.5 text-[12px] leading-tight text-neutral-500">
          {detail}
        </div>
      </div>
      {action}
    </li>
  );
}
