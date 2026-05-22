"use client";

import { Check, Lock, Pencil, UserX } from "lucide-react";
import { useEffect, useState } from "react";

import { BrandMark } from "@/components/shared/brand-mark";
import type { EvalPrompt, TrustReport } from "@/lib/api/client";

export interface EvalFlag {
  category: "curate" | "memory" | "deploy";
  decision: string;
  reason?: string;
  excerpt?: string;
}

export interface EvalScreenProps {
  ready: boolean;
  flags: EvalFlag[];
  summary?: string;
  prompts: EvalPrompt[];
  currentIndex: number;
  draft: string;
  completed: boolean;
  submitting: boolean;
  promoting: boolean;
  error?: string;
  trustReport?: TrustReport;
  onDraftChange: (value: string) => void;
  onApprove: () => void;
  onSaveEdit: () => void;
  onNotMe: () => void;
  onPrivate: () => void;
  onContinue: () => void;
  onFixLater: () => void;
}

function FlagBlock({ flag }: { flag: EvalFlag }) {
  const labels: Record<string, string> = {
    curate: "training data",
    memory: "memory",
    deploy: "model output",
  };
  return (
    <div className="border-l-2 border-foreground/15 pl-4">
      <div className="text-xs uppercase text-foreground/35">
        {labels[flag.category] ?? flag.category} / {flag.decision}
      </div>
      {flag.reason && (
        <div className="mt-1 text-[0.95rem] text-foreground/75">{flag.reason}</div>
      )}
      {flag.excerpt && (
        <div className="mt-2 text-sm italic text-foreground/50">
          &ldquo;{flag.excerpt}&rdquo;
        </div>
      )}
    </div>
  );
}

function TrustLine({ report }: { report?: TrustReport }) {
  if (!report) return null;
  const voice =
    report.voice_total > 0
      ? `${report.voice_approved}/${report.voice_total}`
      : "0/0";
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-foreground/45">
      <span>readiness: {report.readiness}</span>
      <span>voice: {voice}</span>
      {report.privacy_flags > 0 && <span>privacy flags: {report.privacy_flags}</span>}
    </div>
  );
}

export default function EvalScreen({
  ready,
  flags,
  summary,
  prompts,
  currentIndex,
  draft,
  completed,
  submitting,
  promoting,
  error,
  trustReport,
  onDraftChange,
  onApprove,
  onSaveEdit,
  onNotMe,
  onPrivate,
  onContinue,
  onFixLater,
}: EvalScreenProps) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(() => setVisible(true), 180);
    return () => clearTimeout(t);
  }, [ready]);

  const prompt = prompts[currentIndex];
  const total = prompts.length;
  const indexLabel = total > 0 ? `${Math.min(currentIndex + 1, total)} / ${total}` : "0 / 0";
  const busy = submitting || promoting;

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 pb-24 pt-12 sm:px-8">
        <div className="mb-16">
          <BrandMark />
        </div>

        {!ready ? (
          <div className="space-y-4 text-foreground/55">
            <div className="text-base">Reviewing.</div>
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-foreground/40" />
          </div>
        ) : (
          <div
            className={`space-y-10 transition-opacity duration-700 ${
              visible ? "opacity-100" : "opacity-0"
            }`}
          >
            <div className="space-y-3">
              {/* No "X / Y" counter, no header asking "does this
                  sound like you?". The screen is the first meeting —
                  the model is presenting itself and the user is
                  deciding if it lands. Counter lives as a quiet line
                  at the bottom so the user can pace themselves
                  without feeling tested. */}
              <h1 className="text-2xl font-semibold tracking-normal text-foreground">
                {completed
                  ? "Ready."
                  : "I want to know if I'm you. Tell me when these don't land."}
              </h1>
              <TrustLine report={trustReport} />
            </div>

            {completed ? (
              <div className="space-y-8">
                <div className="max-w-2xl text-foreground/75">
                  {trustReport?.readiness === "unproven"
                    ? "Not enough signal yet — a few more would help."
                    : "Good. We can talk now."}
                </div>
                {error && <div className="text-sm text-red-500">{error}</div>}
                <div className="flex flex-wrap items-center gap-5">
                  <button
                    type="button"
                    onClick={onContinue}
                    disabled={busy || trustReport?.readiness === "unproven"}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/80 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Check size={17} aria-hidden="true" />
                    {promoting ? "Almost there" : "Begin"}
                  </button>
                  <button
                    type="button"
                    onClick={onFixLater}
                    disabled={busy}
                    className="text-base text-foreground/45 transition hover:text-foreground/65 disabled:cursor-default disabled:text-foreground/25"
                  >
                    Later
                  </button>
                </div>
              </div>
            ) : prompt ? (
              <div className="space-y-8">
                {/* Situation framed as context the model is responding
                    to, not "here's a probe to evaluate." Lowercased
                    label, no scoreboard feel. */}
                <div className="space-y-3">
                  <div className="text-xs lowercase tracking-wide text-foreground/35">if someone said this to you —</div>
                  <div className="max-w-2xl text-[1.05rem] leading-7 text-foreground/80">
                    {prompt.situation}
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="text-xs lowercase tracking-wide text-foreground/35">— I'd reply</div>
                  <textarea
                    value={draft}
                    onChange={(event) => onDraftChange(event.target.value)}
                    rows={7}
                    className="w-full resize-none border border-foreground/10 bg-transparent p-4 text-base leading-7 text-foreground outline-none transition focus:border-foreground/35"
                  />
                </div>

                {error && <div className="text-sm text-red-500">{error}</div>}

                <div className="flex flex-wrap items-center gap-4">
                  <button
                    type="button"
                    title="Approve"
                    onClick={onApprove}
                    disabled={busy}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/80 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Check size={17} aria-hidden="true" />
                    Approve
                  </button>
                  <button
                    type="button"
                    title="Save edit"
                    onClick={onSaveEdit}
                    disabled={busy || !draft.trim()}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/65 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Pencil size={17} aria-hidden="true" />
                    Save edit
                  </button>
                  <button
                    type="button"
                    title="Not me"
                    onClick={onNotMe}
                    disabled={busy}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/45 transition hover:text-foreground/70 disabled:cursor-default disabled:text-foreground/25"
                  >
                    <UserX size={17} aria-hidden="true" />
                    Not me
                  </button>
                  <button
                    type="button"
                    title="Private"
                    onClick={onPrivate}
                    disabled={busy}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/45 transition hover:text-foreground/70 disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Lock size={17} aria-hidden="true" />
                    Private
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-foreground/60">Nothing to verify yet — open the chat when you&apos;re ready.</div>
            )}

            {/* Quiet counter at the bottom — keeps the user oriented
                without making the screen feel like a test. */}
            {!completed && prompt && total > 0 && (
              <div className="pt-6 text-xs lowercase tracking-wide text-foreground/30">
                {indexLabel}
              </div>
            )}

            {/* Supervisor flags are diagnostic — they belong to us,
                not to the user's first meeting with the model. Kept
                in the SSE stream and audit log so we can see them on
                a future diagnostics surface; hidden here. */}
          </div>
        )}
      </div>
    </main>
  );
}

