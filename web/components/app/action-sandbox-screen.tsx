"use client";

import { Check, FileText, Pencil, Play, RotateCcw, ScanLine, Send, ShieldAlert, X } from "lucide-react";

import { BrandMark } from "@/components/shared/brand-mark";
import type {
  ActionCapability,
  ActionExecutionReceipt,
  ActionProposal,
  TrustReport,
  WorldScanReport,
} from "@/lib/api/client";

export interface ActionSandboxScreenProps {
  proposals: ActionProposal[];
  capabilities: ActionCapability[];
  selectedId?: string;
  draft: string;
  receipt?: ActionExecutionReceipt;
  loading: boolean;
  submitting: boolean;
  error?: string;
  trustReport?: TrustReport;
  latestScan?: WorldScanReport | null;
  onSelect: (id: string) => void;
  onDraftChange: (value: string) => void;
  onApprove: () => void;
  onSaveEdit: () => void;
  onReject: () => void;
  onSimulate: () => void;
  onExecute: () => void;
  onUndo: () => void;
  onScanLaptop: () => void;
  onCreateDraftDemo: () => void;
  onCreateLaptopNoteDemo: () => void;
}

const riskLabel: Record<string, string> = {
  low: "draft",
  medium: "local write",
  high: "step-up",
};

export default function ActionSandboxScreen({
  proposals,
  capabilities,
  selectedId,
  draft,
  receipt,
  loading,
  submitting,
  error,
  trustReport,
  latestScan,
  onSelect,
  onDraftChange,
  onApprove,
  onSaveEdit,
  onReject,
  onSimulate,
  onExecute,
  onUndo,
  onScanLaptop,
  onCreateDraftDemo,
  onCreateLaptopNoteDemo,
}: ActionSandboxScreenProps) {
  const selected = proposals.find((proposal) => proposal.id === selectedId) ?? proposals[0];
  const busy = loading || submitting;
  const selectedCapability = selected
    ? capabilities.find(
        (capability) =>
          capability.surface === selected.surface && capability.operation === selected.operation,
      )
    : undefined;
  const canExecute =
    !!selectedCapability?.supports_execute &&
    !!selected &&
    (selected.risk_level === "low" || selected.status === "approved" || selected.status === "edited");
  const canUndo = !!selectedCapability?.supports_undo && !!receipt?.undo_token;

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto grid min-h-screen max-w-6xl grid-cols-1 gap-10 px-6 py-10 md:grid-cols-[300px_1fr] md:px-8">
        <aside className="space-y-8">
          <BrandMark />
          <div className="space-y-3">
            <h1 className="text-2xl font-semibold">Action sandbox</h1>
            <div className="text-sm leading-6 text-foreground/55">
              Proposed actions stop here first. Your approvals, edits, and rejections become action training data.
            </div>
          </div>
          <div className="space-y-2 text-sm text-foreground/45">
            <div>readiness: {trustReport?.readiness ?? "unproven"}</div>
            <div>actions: {trustReport?.action_approved ?? 0}/{trustReport?.action_total ?? 0}</div>
            <div>world files: {latestScan?.files_indexed ?? 0}</div>
          </div>
          <div className="space-y-3">
            <button
              type="button"
              onClick={onScanLaptop}
              disabled={busy}
              className="inline-flex min-h-10 items-center gap-2 text-sm text-foreground/70 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
            >
              <ScanLine size={16} aria-hidden="true" />
              Scan laptop world
            </button>
            <button
              type="button"
              onClick={onCreateDraftDemo}
              disabled={busy}
              className="inline-flex min-h-10 items-center gap-2 text-sm text-foreground/70 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
            >
              <Send size={16} aria-hidden="true" />
              Create draft proposal
            </button>
            <button
              type="button"
              onClick={onCreateLaptopNoteDemo}
              disabled={busy}
              className="inline-flex min-h-10 items-center gap-2 text-sm text-foreground/70 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
            >
              <FileText size={16} aria-hidden="true" />
              Create laptop note
            </button>
          </div>
          {capabilities.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs uppercase text-foreground/35">adapters</div>
              <div className="space-y-1 text-xs text-foreground/45">
                {capabilities.slice(0, 7).map((capability) => (
                  <div key={capability.key}>{capability.key}</div>
                ))}
              </div>
            </div>
          )}
        </aside>

        <section className="grid min-h-[70vh] grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
          <div className="space-y-3">
            <div className="text-xs uppercase text-foreground/35">queue</div>
            {loading ? (
              <div className="text-sm text-foreground/45">Loading.</div>
            ) : proposals.length === 0 ? (
              <div className="text-sm leading-6 text-foreground/50">
                No pending actions. Create a draft proposal to test the loop.
              </div>
            ) : (
              <div className="space-y-2">
                {proposals.map((proposal) => (
                  <button
                    key={proposal.id}
                    type="button"
                    onClick={() => onSelect(proposal.id)}
                    className={`w-full border-l-2 py-3 pl-4 pr-2 text-left transition ${
                      proposal.id === selected?.id
                        ? "border-foreground text-foreground"
                        : "border-foreground/10 text-foreground/55 hover:text-foreground"
                    }`}
                  >
                    <div className="text-sm font-medium">{proposal.surface}</div>
                    <div className="mt-1 text-xs text-foreground/40">{proposal.operation}</div>
                    <div className="mt-2 text-xs text-foreground/35">{proposal.status}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-8">
            {selected ? (
              <>
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-3 text-xs uppercase text-foreground/35">
                    <span>{selected.status}</span>
                    <span>{riskLabel[selected.risk_level] ?? selected.risk_level}</span>
                    {selected.risk_level !== "low" && (
                      <span className="inline-flex items-center gap-1">
                        <ShieldAlert size={13} aria-hidden="true" />
                        confirmation required
                      </span>
                    )}
                  </div>
                  <h2 className="text-xl font-semibold">
                    {selected.operation.replaceAll("_", " ")}
                  </h2>
                  {selected.rationale && (
                    <p className="max-w-2xl text-sm leading-6 text-foreground/55">
                      {selected.rationale}
                    </p>
                  )}
                </div>

                <div className="space-y-3">
                  <div className="text-xs uppercase text-foreground/35">preview</div>
                  <textarea
                    value={draft}
                    onChange={(event) => onDraftChange(event.target.value)}
                    rows={9}
                    className="w-full resize-none border border-foreground/10 bg-transparent p-4 text-base leading-7 text-foreground outline-none transition focus:border-foreground/35"
                  />
                </div>

                {Object.keys(selected.proposed_payload).length > 0 && (
                  <pre className="overflow-auto border border-foreground/10 p-4 text-xs leading-5 text-foreground/55">
                    {JSON.stringify(selected.proposed_payload, null, 2)}
                  </pre>
                )}

                {error && <div className="text-sm text-red-500">{error}</div>}

                <div className="flex flex-wrap items-center gap-5">
                  <button
                    type="button"
                    onClick={onSimulate}
                    disabled={busy || !selectedCapability?.supports_simulate}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/80 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Play size={17} aria-hidden="true" />
                    Simulate
                  </button>
                  <button
                    type="button"
                    onClick={onApprove}
                    disabled={busy}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/80 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Check size={17} aria-hidden="true" />
                    Approve
                  </button>
                  <button
                    type="button"
                    onClick={onSaveEdit}
                    disabled={busy || !draft.trim()}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/65 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Pencil size={17} aria-hidden="true" />
                    Save edit
                  </button>
                  <button
                    type="button"
                    onClick={onReject}
                    disabled={busy}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/45 transition hover:text-foreground/70 disabled:cursor-default disabled:text-foreground/25"
                  >
                    <X size={17} aria-hidden="true" />
                    Reject
                  </button>
                </div>

                <div className="flex flex-wrap items-center gap-5">
                  <button
                    type="button"
                    onClick={onExecute}
                    disabled={busy || !canExecute}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/80 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <Check size={17} aria-hidden="true" />
                    Execute
                  </button>
                  <button
                    type="button"
                    onClick={onUndo}
                    disabled={busy || !canUndo}
                    className="inline-flex min-h-10 items-center gap-2 text-base text-foreground/55 transition hover:text-foreground disabled:cursor-default disabled:text-foreground/25"
                  >
                    <RotateCcw size={17} aria-hidden="true" />
                    Undo
                  </button>
                </div>

                {receipt && (
                  <div className="space-y-3">
                    <div className="text-xs uppercase text-foreground/35">receipt</div>
                    <div className="grid gap-2 text-sm text-foreground/55">
                      <div>
                        {receipt.mode}: {receipt.ok ? "ok" : "failed"}
                      </div>
                      {receipt.error && <div className="text-red-500">{receipt.error}</div>}
                      {receipt.side_effects.length > 0 && (
                        <div>{receipt.side_effects.join(", ")}</div>
                      )}
                      {receipt.undo_token && <div>undo: {receipt.undo_token}</div>}
                    </div>
                    {receipt.preview && (
                      <pre className="max-h-64 overflow-auto border border-foreground/10 p-4 text-xs leading-5 text-foreground/55">
                        {receipt.preview}
                      </pre>
                    )}
                    {Object.keys(receipt.evidence).length > 0 && (
                      <pre className="overflow-auto border border-foreground/10 p-4 text-xs leading-5 text-foreground/45">
                        {JSON.stringify(receipt.evidence, null, 2)}
                      </pre>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="text-sm text-foreground/45">Select or create an action proposal.</div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
