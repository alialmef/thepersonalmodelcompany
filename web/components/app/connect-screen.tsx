'use client';

import {
  Notebook,
  FileText,
  MessageSquare,
  type LucideIcon,
} from 'lucide-react';
import { BrandMark } from '@/components/shared/brand-mark';

type SourceState = 'idle' | 'connecting' | 'connected';

interface Source {
  id: string;
  icon: LucideIcon;
  name: string;
  detail: string;
  /** "Connect" for OAuth/permissioned; "Add" for drag-and-drop. */
  cta: 'Connect' | 'Add';
}

const SOURCES: Source[] = [
  {
    id: 'messages',
    icon: MessageSquare,
    name: 'Messages',
    detail: 'Read locally · last 24 months',
    cta: 'Connect',
  },
  {
    id: 'notes',
    icon: Notebook,
    name: 'Notes',
    detail: 'Apple Notes · read locally',
    cta: 'Connect',
  },
  {
    id: 'documents',
    icon: FileText,
    name: 'Documents',
    detail: 'Drag in, or pick a folder',
    cta: 'Add',
  },
];

/**
 * Step 1 of 3. The user has just clicked `Begin.` on the first-launch
 * screen. This screen asks them to bring their writing — the trust moment.
 *
 * Design principles encoded here:
 *   - Privacy is part of the page, not a footer (the sub-line beneath
 *     the headline carries it).
 *   - "Bring your writing" — the user is the active party, handing
 *     something over.
 *   - At least one source must be connected to continue. The Continue
 *     button reflects this with a disabled state until a connection
 *     succeeds.
 *
 * The brand mark in the header is intentionally small (32px). It's been
 * the user's persistent companion since the download screen, and continues
 * to pulse through the whole onboarding flow.
 */
export default function ConnectScreen({
  states,
  onConnect,
  onContinue,
}: {
  states: Record<string, SourceState>;
  onConnect: (sourceId: string) => void;
  onContinue: () => void;
}) {
  const connectedCount = Object.values(states).filter(
    (s) => s === 'connected',
  ).length;
  const canContinue = connectedCount > 0;

  return (
    <div className="mx-auto min-h-screen max-w-[520px] bg-white px-7 pt-12 pb-14">
      <header className="mb-10 flex items-center gap-3">
        <BrandMark size={32} />
        <div className="text-[11px] uppercase tracking-[0.04em] text-neutral-500">
          Step 1 of 3
        </div>
      </header>

      <h1 className="mb-3 text-[32px] font-medium leading-[1.1] tracking-[-0.03em] text-neutral-900 pmc-anim-fade-up">
        Bring your writing.
      </h1>
      <p
        className="mb-9 text-[14px] leading-[1.5] text-neutral-500 pmc-anim-fade-up"
        style={{ animationDelay: '0.3s' }}
      >
        Everything stays on your Mac. The audit log shows every byte.
      </p>

      <div
        className="flex flex-col gap-px overflow-hidden rounded-[10px] border-[0.5px] border-neutral-200 bg-neutral-200"
        role="list"
      >
        {SOURCES.map((source, i) => (
          <SourceRow
            key={source.id}
            source={source}
            state={states[source.id] ?? 'idle'}
            onConnect={() => onConnect(source.id)}
            animationDelay={0.7 + i * 0.1}
          />
        ))}
      </div>

      <div className="mt-8 flex items-center justify-between">
        <p className="text-[12px] text-neutral-500">
          {canContinue
            ? `${connectedCount} connected.`
            : 'Connect at least one to continue.'}
        </p>
        <button
          onClick={onContinue}
          disabled={!canContinue}
          className={`rounded-full px-[22px] py-[9px] text-[13px] font-medium transition-colors ${
            canContinue
              ? 'cursor-pointer bg-neutral-900 text-white hover:bg-neutral-800'
              : 'cursor-not-allowed bg-neutral-900/8 text-neutral-900/40'
          }`}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

function SourceRow({
  source,
  state,
  onConnect,
  animationDelay,
}: {
  source: Source;
  state: SourceState;
  onConnect: () => void;
  animationDelay: number;
}) {
  const Icon = source.icon;
  const connected = state === 'connected';

  return (
    <div
      role="listitem"
      className="pmc-anim-fade-up flex items-center gap-3.5 bg-white px-[18px] py-4 transition-colors hover:bg-neutral-50"
      style={{ animationDelay: `${animationDelay}s` }}
    >
      <div className="flex size-7 items-center justify-center rounded-md bg-neutral-100 text-neutral-900">
        <Icon className="size-4" strokeWidth={1.5} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[14px] font-medium leading-tight tracking-[-0.01em] text-neutral-900">
          {source.name}
        </div>
        <div className="mt-0.5 text-[12px] text-neutral-500">
          {source.detail}
        </div>
      </div>
      {connected ? (
        <span className="text-[12px] text-neutral-500">Connected</span>
      ) : (
        <button
          onClick={onConnect}
          disabled={state === 'connecting'}
          className={`rounded-full px-3.5 py-1.5 text-[12px] font-medium transition-colors ${
            source.cta === 'Connect'
              ? 'bg-neutral-900 text-white hover:bg-neutral-800'
              : 'border-[0.5px] border-neutral-900/15 bg-white text-neutral-900 hover:bg-neutral-50'
          } ${state === 'connecting' ? 'opacity-60' : ''}`}
        >
          {state === 'connecting' ? 'Connecting…' : source.cta}
        </button>
      )}
    </div>
  );
}
