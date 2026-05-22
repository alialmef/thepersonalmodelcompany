"use client";

import { useEffect, useState } from "react";

import { BrandMark } from "@/components/shared/brand-mark";

/**
 * /training — the screen where the user watches their voice arrive.
 *
 * Two checkpoint samples come through the SSE stream as
 * `checkpoint_sample` events:
 *   - stage="baseline" — the base model's generic response
 *   - stage="final"    — the fine-tuned model's response
 *
 * Each lands as a typed line stacked underneath a fixed prompt. The
 * user literally watches the model become *them* in two beats. No
 * loss curve. No progress bar. Just the same prompt answered twice —
 * before and after.
 *
 * The screen is *dismissable*: training takes 30+ min and the user
 * shouldn't be locked here. After the baseline appears we surface
 * a small "you can close this — we'll find you when it's ready"
 * line.
 */

export interface CheckpointSample {
  stage: "baseline" | "final";
  response: string;
  model?: string;
}

export interface TrainingScreenProps {
  prompt: string;
  samples: CheckpointSample[];
  /** True once `training_completed` arrives. Reveals the Continue affordance. */
  done: boolean;
  onContinue: () => void;
}

function SampleBlock({
  label,
  body,
  appearDelay,
}: {
  label: string;
  body: string;
  appearDelay: number;
}) {
  const [visible, setVisible] = useState(false);
  const [typed, setTyped] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), appearDelay);
    return () => clearTimeout(t);
  }, [appearDelay]);

  useEffect(() => {
    if (!visible) return;
    let i = 0;
    const tick = () => {
      i += 2;
      setTyped(body.slice(0, i));
      if (i < body.length) {
        timer = window.setTimeout(tick, 14);
      }
    };
    let timer = window.setTimeout(tick, 200);
    return () => window.clearTimeout(timer);
  }, [visible, body]);

  return (
    <div
      className={`transition-opacity duration-1000 ease-out ${
        visible ? "opacity-100" : "opacity-0"
      }`}
    >
      <div className="mb-3 text-xs uppercase tracking-wider text-foreground/35">
        {label}
      </div>
      <div className="text-[0.98rem] leading-relaxed text-foreground/85 whitespace-pre-wrap">
        {typed}
        {visible && typed.length < body.length && (
          <span className="ml-0.5 inline-block h-4 w-px animate-pulse bg-foreground/60 align-middle" />
        )}
      </div>
    </div>
  );
}

export default function TrainingScreen({
  prompt,
  samples,
  done,
  onContinue,
}: TrainingScreenProps) {
  const baseline = samples.find((s) => s.stage === "baseline");
  const final = samples.find((s) => s.stage === "final");

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <div className="space-y-16">
          <div className="space-y-4">
            <div className="text-xs uppercase tracking-wider text-foreground/35">
              The same question, asked twice.
            </div>
            <div className="text-lg italic text-foreground/80">
              &ldquo;{prompt}&rdquo;
            </div>
          </div>

          {baseline && (
            <SampleBlock
              label="before"
              body={baseline.response}
              appearDelay={300}
            />
          )}

          {final ? (
            <SampleBlock label="after" body={final.response} appearDelay={300} />
          ) : (
            baseline && (
              <div className="space-y-3 text-foreground/55">
                <div className="text-xs uppercase tracking-wider text-foreground/35">
                  after
                </div>
                <div className="text-sm italic">
                  Now the voice is being learned.
                </div>
                <div className="text-sm">
                  This is the slow part. You can close this window — we&apos;ll
                  find you when it&apos;s ready.
                </div>
                <div className="pt-2">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-foreground/40" />
                </div>
              </div>
            )
          )}

          {!baseline && (
            <div className="text-sm text-foreground/50">
              Preparing the base model…
            </div>
          )}
        </div>

        <div className="mt-auto pt-24">
          <button
            type="button"
            onClick={onContinue}
            disabled={!done}
            className={`text-base transition-opacity duration-700 ${
              done
                ? "cursor-pointer text-foreground/80 hover:text-foreground opacity-100"
                : "cursor-default text-foreground/30 opacity-50"
            }`}
          >
            {done ? "Continue" : "Still training…"}
          </button>
        </div>
      </div>
    </main>
  );
}
