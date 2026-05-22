"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import TrainingScreen, {
  type CheckpointSample,
} from "@/components/app/training-screen";
import { useUser } from "@/hooks/use-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

const FIXED_PROMPT = "Tell me about your weekend.";

interface AuditEvent {
  timestamp?: string;
  stage?: string;
  event?: string;
  data?: Record<string, unknown>;
}

function TrainInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const { user } = useUser();
  const userId = searchParams.get("user") ?? user?.pmcUserId ?? "";

  const [samples, setSamples] = useState<CheckpointSample[]>([]);
  const [done, setDone] = useState(false);
  const seenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!jobId) return;
    const url = `${PMC_API_URL}/v1/users/${encodeURIComponent(userId)}/runs/${encodeURIComponent(jobId)}/events`;
    const es = new EventSource(url);

    es.onmessage = (msg) => {
      if (msg.data === "[DONE]") {
        es.close();
        return;
      }
      let ev: AuditEvent;
      try {
        ev = JSON.parse(msg.data) as AuditEvent;
      } catch {
        return;
      }

      if (ev.event === "checkpoint_sample" && ev.data) {
        const stage = ev.data.stage as "baseline" | "final" | undefined;
        const response = ev.data.response as string | undefined;
        if (!stage || !response) return;
        if (seenRef.current.has(stage)) return;
        seenRef.current.add(stage);
        setSamples((prev) => [
          ...prev,
          { stage, response, model: ev.data?.model as string | undefined },
        ]);
        return;
      }

      if (ev.event === "training_completed" || ev.event === "job_finished") {
        setDone(true);
      }
    };

    return () => es.close();
  }, [jobId, userId]);

  const handleContinue = () => {
    router.push(
      `/eval?job=${encodeURIComponent(jobId ?? "")}&user=${encodeURIComponent(userId)}`,
    );
  };

  return (
    <TrainingScreen
      prompt={FIXED_PROMPT}
      samples={samples}
      done={done}
      onContinue={handleContinue}
    />
  );
}

export default function TrainPage() {
  return (
    <Suspense fallback={<div className="min-h-screen w-full bg-background" />}>
      <TrainInner />
    </Suspense>
  );
}
