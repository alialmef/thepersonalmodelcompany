'use client';

import { BrandMark } from '@/components/shared/brand-mark';
import { LetterCascade } from '@/components/shared/letter-cascade';

/**
 * Step 2 of 3. While curate runs (typically a few minutes), the user
 * watches two things happen in parallel:
 *
 *   Left column — discrete events as the pipeline runs ("Read 12,431
 *   messages", "Filtered 2,108 short replies"). These come from the
 *   backend SSE event stream at /v1/users/{id}/runs/{job_id}/events.
 *
 *   Right column — the style profile assembles itself line by line, each
 *   line using the LetterCascade primitive. This is the most satisfying
 *   moment in the funnel: the user watches their own voice being described
 *   back to them.
 *
 * The screen does not auto-advance when the pipeline completes. After the
 * final event fires, a Continue button fades in (handled by the parent
 * component watching for the `complete` event).
 *
 * BACKEND CONTRACT:
 *   Events are pushed via SSE. Each event has shape:
 *     { ts: ISO string, kind: 'milestone' | 'micro', text: string }
 *   Milestone events are the headline numbers; micro events keep the
 *   stream alive between milestones (and are typically not shown, or are
 *   replaced by their successor).
 *
 *   The style profile arrives as a single event when ready:
 *     { kind: 'profile', lines: string[] }
 *   Each line in the array is one observation about the user.
 */

interface CurateEvent {
  ts: string;
  text: string;
}

interface SourceCount {
  label: string;
  count: number;
}

interface CurateScreenProps {
  events: CurateEvent[];
  profileLines: string[];
  sourceCounts?: SourceCount[];
  onContinue?: () => void;
  isComplete?: boolean;
}

export default function CurateScreen({
  events,
  profileLines,
  sourceCounts = [],
  onContinue,
  isComplete,
}: CurateScreenProps) {
  const total = sourceCounts.reduce((s, x) => s + x.count, 0);
  return (
    <div className="mx-auto min-h-screen max-w-[620px] bg-white px-7 pt-12 pb-14">
      <header className="mb-10 flex items-center gap-3">
        <BrandMark size={32} />
        <div className="text-[11px] uppercase tracking-[0.04em] text-neutral-500">
          Step 2 of 3 · Reading your writing
        </div>
      </header>

      <h1 className="mb-3 text-[32px] font-medium leading-[1.1] tracking-[-0.03em] text-neutral-900 pmc-anim-fade-up">
        Reading you.
      </h1>
      <p
        className="mb-8 text-[14px] leading-[1.5] text-neutral-500 pmc-anim-fade-up"
        style={{ animationDelay: '0.3s' }}
      >
        This takes a few minutes. You can close the window — we&apos;ll keep
        going.
      </p>

      <SourceScoreboard sources={sourceCounts} total={total} />

      <div className="grid grid-cols-2 gap-6">
        <EventStream events={events} />
        <StyleProfile lines={profileLines} />
      </div>

      {isComplete && (
        <div className="mt-10 flex justify-end pmc-anim-fade-up">
          <button
            onClick={onContinue}
            className="rounded-full bg-neutral-900 px-[22px] py-[9px] text-[13px] font-medium text-white transition-colors hover:bg-neutral-800"
          >
            Continue
          </button>
        </div>
      )}
    </div>
  );
}

function SourceScoreboard({
  sources,
  total,
}: {
  sources: SourceCount[];
  total: number;
}) {
  const empty = sources.length === 0;
  return (
    <div className="mb-10 rounded-[10px] border-[0.5px] border-neutral-200 bg-white">
      <div className="flex items-baseline justify-between px-5 pt-4 pb-2">
        <h2 className="text-[10px] uppercase tracking-[0.08em] text-neutral-500">
          Your writing
        </h2>
        <div className="font-mono text-[12px] text-neutral-500 tabular-nums">
          {empty ? "counting…" : `${total.toLocaleString()} items`}
        </div>
      </div>
      <div className="flex flex-col gap-px overflow-hidden bg-neutral-200">
        {empty ? (
          <div className="flex items-center justify-between bg-white px-5 py-3">
            <div className="text-[14px] tracking-[-0.005em] text-neutral-500">
              Reading from your connected sources…
            </div>
          </div>
        ) : (
          sources.map((s) => (
            <div
              key={s.label}
              className="flex items-center justify-between bg-white px-5 py-3"
            >
              <div className="text-[14px] tracking-[-0.005em] text-neutral-900">
                {s.label}
              </div>
              <div className="font-mono text-[14px] tabular-nums text-neutral-900">
                {s.count.toLocaleString()}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function EventStream({ events }: { events: CurateEvent[] }) {
  return (
    <div>
      <h2 className="mb-3.5 text-[10px] uppercase tracking-[0.08em] text-neutral-500">
        Events
      </h2>
      <div className="flex flex-col gap-2.5 font-mono text-[12px]">
        {events.map((event, i) => (
          <div
            key={i}
            className="flex gap-2.5 pmc-anim-fade-up"
            // Stagger each new event in by 200ms relative to its arrival.
            // In production the events arrive over real time and don't
            // need a stagger — but if multiple arrive in the same tick,
            // this prevents them from popping in simultaneously.
            style={{ animationDelay: `${i * 0.1}s` }}
          >
            <span className="min-w-[36px] text-neutral-500">{event.ts}</span>
            <span className="text-neutral-900">{event.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StyleProfile({ lines }: { lines: string[] }) {
  return (
    <div>
      <h2 className="mb-3.5 text-[10px] uppercase tracking-[0.08em] text-neutral-500">
        Style profile
      </h2>
      <div className="rounded-[10px] border-[0.5px] border-neutral-200 bg-neutral-50 p-5">
        <div className="text-[14px] leading-[1.7] tracking-[-0.01em] text-neutral-900">
          {lines.map((line, i) => (
            <div key={i} className="mb-2 last:mb-0">
              {/*
                Each line cascades in letter-by-letter. The startMs offset
                ensures lines arrive sequentially, not simultaneously.
                Assumes lines are added to the array as they arrive from
                the backend — so the staggered offset is relative to
                _mount_ time of the line, not absolute page time.
              */}
              <LetterCascade text={line} startMs={0} perLetterMs={28} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
