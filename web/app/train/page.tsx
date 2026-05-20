"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { TrainInProgress } from "@/components/app/train-screen";
import { DEMO_USER_ID } from "@/lib/demo-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

interface AuditEvent {
  timestamp?: string;
  stage?: string;
  event?: string;
  data?: Record<string, unknown>;
  status?: string;
}

interface TrainingStats {
  loss: number;
  step: number;
  totalSteps: number;
  series: Array<{ step: number; loss: number }>;
  etaMinutes: number;
}

/**
 * Screen 5 — Training your model.
 *
 * Consumes the backend SSE event stream and surfaces the train-stage events
 * as live training statistics for the loss curve. Filters to `mlx_step`
 * events emitted by `pmc/train/mlx_trainer.py` and tracks (step, loss)
 * points as they arrive.
 *
 * On `job_finished` → /eval (which then leads to /first-meeting and /chat).
 */
function TrainPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const userId = searchParams.get("user") ?? DEMO_USER_ID;

  const [stats, setStats] = useState<TrainingStats>({
    loss: 0,
    step: 0,
    totalSteps: 100,
    series: [],
    etaMinutes: 40,
  });
  const [done, setDone] = useState<"ok" | "fail" | null>(null);
  const startedAtRef = useRef<number>(Date.now());

  useEffect(() => {
    if (!jobId) return;
    const url = `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/runs/${encodeURIComponent(jobId)}/events`;
    const es = new EventSource(url);

    es.onmessage = (msg) => {
      if (msg.data === "[DONE]") {
        es.close();
        return;
      }
      let parsed: AuditEvent;
      try {
        parsed = JSON.parse(msg.data) as AuditEvent;
      } catch {
        return;
      }

      if (parsed.event === "job_finished") {
        if (parsed.status === "completed") {
          setDone("ok");
          setTimeout(
            () => router.push(`/eval?user=${encodeURIComponent(userId)}`),
            1800,
          );
        } else {
          setDone("fail");
        }
        es.close();
        return;
      }

      // Pull mlx training metrics.
      const data = parsed.data ?? {};
      if (parsed.event === "mlx_train_starting") {
        const iters = typeof data.iters === "number" ? data.iters : 100;
        setStats((s) => ({ ...s, totalSteps: iters }));
      } else if (parsed.event === "mlx_step" && typeof data.step === "number") {
        const step = data.step;
        const loss =
          typeof data.train_loss === "number"
            ? data.train_loss
            : typeof data.val_loss === "number"
            ? data.val_loss
            : null;
        if (loss === null) return;

        setStats((s) => {
          const series = [...s.series, { step, loss }];
          // ETA: extrapolate from current rate.
          const elapsedMin = (Date.now() - startedAtRef.current) / 60_000;
          const remain =
            step > 0
              ? Math.max(1, Math.round((s.totalSteps - step) * (elapsedMin / step)))
              : s.etaMinutes;
          return { ...s, loss, step, series, etaMinutes: remain };
        });
      } else if (parsed.event === "mlx_train_completed") {
        const finalLoss = typeof data.train_loss === "number" ? data.train_loss : stats.loss;
        setStats((s) => ({ ...s, loss: finalLoss, step: s.totalSteps }));
      }
    };

    return () => es.close();
  }, [jobId, userId, router, stats.loss]);

  if (!jobId) {
    return (
      <main className="mx-auto flex min-h-screen max-w-[620px] items-center justify-center bg-white px-7">
        <p className="text-[13px] text-neutral-500">no training in progress</p>
      </main>
    );
  }

  if (done === "ok") {
    return (
      <main className="mx-auto flex min-h-screen max-w-[620px] flex-col items-center justify-center bg-white px-7">
        <h1 className="text-[32px] font-medium tracking-[-0.03em] text-neutral-900">
          your model is ready.
        </h1>
      </main>
    );
  }

  if (done === "fail") {
    return (
      <main className="mx-auto flex min-h-screen max-w-[620px] flex-col items-center justify-center bg-white px-7">
        <h1 className="text-[24px] font-medium tracking-[-0.02em] text-neutral-900">
          training failed.
        </h1>
        <p className="mt-2 text-[14px] text-neutral-500">
          see ~/.pmc-dev/storage/users/{userId}/audit.jsonl
        </p>
      </main>
    );
  }

  return <TrainInProgress stats={stats} />;
}

export default function TrainPage() {
  return (
    <Suspense fallback={<main className="min-h-screen bg-white" />}>
      <TrainPageInner />
    </Suspense>
  );
}
