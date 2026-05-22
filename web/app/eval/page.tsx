"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import EvalScreen, { type EvalFlag } from "@/components/app/eval-screen";
import {
  getEvalPrompts,
  promoteRun,
  submitEvalJudgment,
  type EvalPrompt,
  type TrustReport,
} from "@/lib/api/client";
import { useUser } from "@/hooks/use-user";

const PMC_API_URL =
  process.env.NEXT_PUBLIC_PMC_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

interface AuditEvent {
  timestamp?: string;
  stage?: string;
  event?: string;
  data?: Record<string, unknown>;
  result?: {
    run_id?: string | null;
    status?: string;
  } | null;
}

type Verdict = "approve" | "reject" | "edit" | "not_me" | "private";

function EvalInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const { user } = useUser();
  const userId = searchParams.get("user") ?? user?.pmcUserId ?? "";

  const [ready, setReady] = useState(false);
  const [flags, setFlags] = useState<EvalFlag[]>([]);
  const [summary, setSummary] = useState<string | undefined>(undefined);
  const [runId, setRunId] = useState<string | null>(null);
  const [prompts, setPrompts] = useState<EvalPrompt[]>([]);
  const [trustReport, setTrustReport] = useState<TrustReport | undefined>(undefined);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const seenRef = useRef<Set<string>>(new Set());
  const loadedPromptsFor = useRef<string | null>(null);

  useEffect(() => {
    if (!jobId) {
      setReady(true);
      return;
    }
    if (!userId) return;
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

      if (ev.event === "curate_supervisor_report" && ev.data) {
        const flagsIn =
          (ev.data.flags as
            | Array<{ decision?: string; reason?: string; index?: number }>
            | undefined) ?? [];
        for (const f of flagsIn) {
          const key = `curate:${f.index ?? ""}:${f.decision ?? ""}`;
          if (seenRef.current.has(key)) continue;
          seenRef.current.add(key);
          setFlags((prev) => [
            ...prev,
            {
              category: "curate",
              decision: f.decision ?? "flagged",
              reason: f.reason,
            },
          ]);
        }
      }

      if (ev.event === "memory_supervisor_report" && ev.data) {
        const flagsIn =
          (ev.data.flags as
            | Array<{ episode_id?: string; decision?: string; reason?: string }>
            | undefined) ?? [];
        for (const f of flagsIn) {
          const key = `memory:${f.episode_id ?? ""}:${f.decision ?? ""}`;
          if (seenRef.current.has(key)) continue;
          seenRef.current.add(key);
          setFlags((prev) => [
            ...prev,
            {
              category: "memory",
              decision: f.decision ?? "flagged",
              reason: f.reason,
            },
          ]);
        }
      }

      if (ev.event === "deploy_supervisor_report" && ev.data) {
        const sum = ev.data.summary as string | undefined;
        const issuesIn =
          (ev.data.issues as
            | Array<{ kind?: string; note?: string; prompt_index?: number }>
            | undefined) ?? [];
        if (sum) setSummary(sum);
        for (const i of issuesIn) {
          const key = `deploy:${i.prompt_index ?? ""}:${i.kind ?? ""}`;
          if (seenRef.current.has(key)) continue;
          seenRef.current.add(key);
          setFlags((prev) => [
            ...prev,
            {
              category: "deploy",
              decision: i.kind ?? "issue",
              reason: i.note,
            },
          ]);
        }
        setReady(true);
        return;
      }

      if (ev.event === "job_finished") {
        if (ev.result?.run_id) setRunId(ev.result.run_id);
        setReady(true);
      }

      if (ev.event === "adapter_deployed") {
        setReady(true);
      }
    };

    es.onerror = () => {
      setReady(true);
      es.close();
    };

    return () => es.close();
  }, [jobId, userId]);

  useEffect(() => {
    if (!ready || !userId) return;
    if (loadedPromptsFor.current === userId) return;
    loadedPromptsFor.current = userId;
    setError(undefined);
    getEvalPrompts(userId)
      .then((body) => {
        setPrompts(body.prompts);
        setTrustReport(body.trust_report);
        setCurrentIndex(0);
        setDraft(body.prompts[0]?.response ?? "");
      })
      .catch((err: unknown) => {
        loadedPromptsFor.current = null;
        setError(err instanceof Error ? err.message : "Failed to load private checks");
      });
  }, [ready, userId]);

  useEffect(() => {
    setDraft(prompts[currentIndex]?.response ?? "");
  }, [currentIndex, prompts]);

  const completed = prompts.length > 0 && currentIndex >= prompts.length;

  async function record(verdict: Verdict) {
    const prompt = prompts[currentIndex];
    if (!prompt || !userId) return;
    const candidate = prompt.candidates[0];
    setSubmitting(true);
    setError(undefined);
    try {
      const editedText = verdict === "edit" ? draft.trim() : undefined;
      const response = await submitEvalJudgment({
        userId,
        probeId: prompt.id,
        verdict,
        chosenCandidateId: candidate?.id,
        editedText,
        dimension: prompt.kind === "voice" ? "voice" : "overall",
      });
      setTrustReport(response.trust_report);
      setCurrentIndex((idx) => idx + 1);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save judgment");
    } finally {
      setSubmitting(false);
    }
  }

  // The /eval flow IS the first meeting now. Verification + first
  // meeting collapsed into one screen — the model demonstrates voice
  // across situations, the user decides if it lands, and that
  // judgement is the relationship's foundation. After promotion the
  // user goes straight into /chat with the trained adapter.
  const goNext = () => {
    router.push(`/chat`);
  };

  async function handleContinue() {
    if (!userId) return;
    if (!runId) {
      goNext();
      return;
    }
    setPromoting(true);
    setError(undefined);
    try {
      const response = await promoteRun(userId, runId);
      setTrustReport(response.trust_report);
      goNext();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Private verification is not ready");
    } finally {
      setPromoting(false);
    }
  }

  return (
    <EvalScreen
      ready={ready}
      flags={flags}
      summary={summary}
      prompts={prompts}
      currentIndex={currentIndex}
      draft={draft}
      completed={completed}
      submitting={submitting}
      promoting={promoting}
      error={error}
      trustReport={trustReport}
      onDraftChange={setDraft}
      onApprove={() => record("approve")}
      onSaveEdit={() => record("edit")}
      onNotMe={() => record("not_me")}
      onPrivate={() => record("private")}
      onContinue={handleContinue}
      onFixLater={goNext}
    />
  );
}

export default function EvalPage() {
  return (
    <Suspense fallback={<div className="min-h-screen w-full bg-background" />}>
      <EvalInner />
    </Suspense>
  );
}

