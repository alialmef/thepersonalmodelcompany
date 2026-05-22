"use client";

import { useEffect, useMemo, useState } from "react";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";

import ActionSandboxScreen from "@/components/app/action-sandbox-screen";
import {
  listActionCapabilities,
  createActionProposal,
  listActionProposals,
  listWorldFiles,
  reviewActionProposal,
  runActionProposal,
  scanWorld,
  type ActionCapability,
  type ActionExecutionMode,
  type ActionExecutionReceipt,
  type ActionProposal,
  type TrustReport,
  type WorldScanReport,
} from "@/lib/api/client";
import { useUser } from "@/hooks/use-user";

function ActionsInner() {
  const searchParams = useSearchParams();
  const { user } = useUser();
  const userId = searchParams.get("user") ?? user?.pmcUserId ?? "";

  const [proposals, setProposals] = useState<ActionProposal[]>([]);
  const [capabilities, setCapabilities] = useState<ActionCapability[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [trustReport, setTrustReport] = useState<TrustReport | undefined>(undefined);
  const [latestScan, setLatestScan] = useState<WorldScanReport | null>(null);
  const [receipts, setReceipts] = useState<Record<string, ActionExecutionReceipt>>({});
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);

  const selected = useMemo(
    () => proposals.find((proposal) => proposal.id === selectedId) ?? proposals[0],
    [proposals, selectedId],
  );

  useEffect(() => {
    if (!userId) return;
    setLoading(true);
    setError(undefined);
    Promise.all([
      listActionProposals({ userId, limit: 30 }),
      listActionCapabilities(userId),
      listWorldFiles({ userId, limit: 5 }),
    ])
      .then(([actions, capabilityBody, world]) => {
        setProposals(actions.proposals);
        setCapabilities(capabilityBody.capabilities);
        setTrustReport(actions.trust_report);
        setLatestScan(world.latest_scan);
        setSelectedId(actions.proposals[0]?.id);
        setDraft(actions.proposals[0]?.proposed_text ?? "");
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load actions");
      })
      .finally(() => setLoading(false));
  }, [userId]);

  useEffect(() => {
    setDraft(selected?.proposed_text ?? "");
  }, [selected?.id, selected?.proposed_text]);

  function upsertProposal(proposal: ActionProposal) {
    setProposals((prev) => {
      const next = prev.filter((item) => item.id !== proposal.id);
      return [proposal, ...next];
    });
    setSelectedId(proposal.id);
  }

  async function createDraftDemo() {
    if (!userId) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const body = await createActionProposal({
        userId,
        surface: "messages",
        operation: "draft_reply",
        prompt: "Maya asked if dinner Thursday still works.",
        proposedText: "yeah thursday still works for me",
        proposedPayload: {
          recipient: "Maya",
          mode: "draft_only",
        },
        rationale: "Low-risk draft for a familiar reply pattern.",
      });
      setTrustReport(body.trust_report);
      upsertProposal(body.proposal);
      setDraft(body.proposal.proposed_text);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create proposal");
    } finally {
      setSubmitting(false);
    }
  }

  async function createLaptopNoteDemo() {
    if (!userId) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const body = await createActionProposal({
        userId,
        surface: "notes",
        operation: "create",
        prompt: "Create a local note that proves the laptop action runtime works.",
        proposedText: "Create ~/Documents/PMC Notes/frontier_laptop_world.md",
        proposedPayload: {
          title: "frontier laptop world",
          content: "full-disk context, staged mutation, receipt-backed undo\n",
        },
        rationale: "Medium-risk local write. It must be reviewed before execution.",
        riskLevel: "medium",
      });
      setTrustReport(body.trust_report);
      upsertProposal(body.proposal);
      setDraft(body.proposal.proposed_text);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create proposal");
    } finally {
      setSubmitting(false);
    }
  }

  async function review(decision: "approved" | "edited" | "rejected") {
    if (!userId || !selected) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const body = await reviewActionProposal({
        userId,
        proposalId: selected.id,
        decision,
        editedText: decision === "edited" ? draft : undefined,
        finalPayload: selected.proposed_payload,
      });
      setTrustReport(body.trust_report);
      upsertProposal(body.proposal);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to review proposal");
    } finally {
      setSubmitting(false);
    }
  }

  async function run(mode: ActionExecutionMode) {
    if (!userId || !selected) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const last = receipts[selected.id];
      const body = await runActionProposal({
        userId,
        proposalId: selected.id,
        mode,
        payload: mode === "undo" && last?.undo_token ? { undo_token: last.undo_token } : undefined,
      });
      setTrustReport(body.trust_report);
      upsertProposal(body.proposal);
      setReceipts((prev) => ({ ...prev, [selected.id]: body.receipt }));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : `Failed to ${mode}`);
    } finally {
      setSubmitting(false);
    }
  }

  async function scanLaptop() {
    if (!userId) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const body = await scanWorld({ userId, fullDisk: true, maxFiles: 2000 });
      setLatestScan(body.scan);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to scan laptop");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <ActionSandboxScreen
      proposals={proposals}
      capabilities={capabilities}
      selectedId={selected?.id}
      draft={draft}
      receipt={selected ? receipts[selected.id] : undefined}
      loading={loading}
      submitting={submitting}
      error={error}
      trustReport={trustReport}
      latestScan={latestScan}
      onSelect={setSelectedId}
      onDraftChange={setDraft}
      onApprove={() => review("approved")}
      onSaveEdit={() => review("edited")}
      onReject={() => review("rejected")}
      onSimulate={() => run("simulate")}
      onExecute={() => run("execute")}
      onUndo={() => run("undo")}
      onScanLaptop={scanLaptop}
      onCreateDraftDemo={createDraftDemo}
      onCreateLaptopNoteDemo={createLaptopNoteDemo}
    />
  );
}

export default function ActionsPage() {
  return (
    <Suspense fallback={<div className="min-h-screen w-full bg-background" />}>
      <ActionsInner />
    </Suspense>
  );
}
