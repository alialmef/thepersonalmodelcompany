'use client';

import { useState } from 'react';
import { BrandMark } from '@/components/shared/brand-mark';

/**
 * Step 3 of 3. Two sequential states.
 *
 *   STATE A — Tier selection ("Pick a base.")
 *     User chooses Try / Standard / Frontier. Tiers are described by
 *     capability, not parameters. Try is free.
 *
 *   STATE B — Training ("Training your model.")
 *     Loss curve descends. Two big stat numbers update from the backend
 *     SSE stream. macOS notification fires when the run completes; the
 *     parent navigates to /app/ready (Act 4) on completion.
 *
 * The parent component is responsible for managing which state is shown
 * and handing in the live training stats.
 */

type Tier = 'try' | 'standard' | 'frontier';

interface TierDef {
  id: Tier;
  name: string;
  baseModel: string;
  price: string;
  capability: string;
  description: string;
}

const TIERS: TierDef[] = [
  {
    id: 'try',
    name: 'Try',
    baseModel: 'Llama 3.1 8B',
    price: 'Free',
    capability: 'A model you can chat with.',
    description:
      'Drafts in your voice. Answers in your voice. Just chat.',
  },
  {
    id: 'standard',
    name: 'Standard',
    baseModel: 'Qwen 3.6 27B Dense',
    price: '$79',
    capability: 'An assistant that can do things.',
    description:
      'Reads your mail, browses, takes notes, books, schedules. Light agent tasks.',
  },
  {
    id: 'frontier',
    name: 'Frontier',
    baseModel: 'Kimi K2.6',
    price: '$299',
    capability: 'A model that can code.',
    description:
      'Everything Standard does, plus writes software at frontier quality. Builds, tests, ships.',
  },
];

export function TrainPickBase({
  onTrain,
}: {
  onTrain: (tier: Tier) => void;
}) {
  const [selected, setSelected] = useState<Tier>('try');

  return (
    <div className="mx-auto min-h-screen max-w-[560px] bg-white px-7 pt-12 pb-14">
      <header className="mb-10 flex items-center gap-3">
        <BrandMark size={32} />
        <div className="text-[11px] uppercase tracking-[0.04em] text-neutral-500">
          Step 3 of 3 · Pick your base
        </div>
      </header>

      <h1 className="mb-3 text-[32px] font-medium leading-[1.1] tracking-[-0.03em] text-neutral-900 pmc-anim-fade-up">
        Pick a base.
      </h1>
      <p
        className="mb-9 text-[14px] leading-[1.5] text-neutral-500 pmc-anim-fade-up"
        style={{ animationDelay: '0.3s' }}
      >
        Your writing is the same. The base is what we train it onto.
      </p>

      <div className="flex flex-col gap-3">
        {TIERS.map((tier, i) => (
          <TierCard
            key={tier.id}
            tier={tier}
            selected={selected === tier.id}
            onSelect={() => setSelected(tier.id)}
            animationDelay={0.7 + i * 0.15}
          />
        ))}
      </div>

      <div
        className="mt-8 flex items-center justify-between pmc-anim-fade-up"
        style={{ animationDelay: '1.6s' }}
      >
        <p className="text-[12px] text-neutral-500">
          Cancel anytime. Your bundle is yours.
        </p>
        <button
          onClick={() => onTrain(selected)}
          className="rounded-full bg-neutral-900 px-[22px] py-[9px] text-[13px] font-medium text-white transition-colors hover:bg-neutral-800"
        >
          Train
        </button>
      </div>
    </div>
  );
}

function TierCard({
  tier,
  selected,
  onSelect,
  animationDelay,
}: {
  tier: TierDef;
  selected: boolean;
  onSelect: () => void;
  animationDelay: number;
}) {
  return (
    <button
      onClick={onSelect}
      className={`pmc-anim-fade-up cursor-pointer rounded-[10px] border-[0.5px] px-[22px] py-5 text-left transition-colors ${
        selected
          ? 'border-neutral-300 bg-neutral-100/60'
          : 'border-neutral-200 bg-white hover:bg-neutral-50'
      }`}
      style={{ animationDelay: `${animationDelay}s` }}
    >
      <div className="mb-2.5 flex items-start justify-between">
        <div>
          <div className="text-[16px] font-medium tracking-[-0.01em] text-neutral-900">
            {tier.name}
          </div>
          <div className="mt-0.5 text-[11px] text-neutral-500">
            {tier.baseModel}
          </div>
        </div>
        <div className="flex items-center gap-2.5">
          <div className="text-[17px] font-medium tracking-[-0.02em] text-neutral-900">
            {tier.price}
            {tier.price !== 'Free' && (
              <span className="text-[11px] font-normal text-neutral-500">
                {' '}
                /mo
              </span>
            )}
          </div>
          {selected && <CheckBadge />}
        </div>
      </div>
      <div className="mb-1 text-[13px] leading-[1.55] text-neutral-900">
        {tier.capability}
      </div>
      <div className="text-[12px] leading-[1.55] text-neutral-500">
        {tier.description}
      </div>
    </button>
  );
}

function CheckBadge() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="7.5" fill="#1D1D1F" />
      <path
        d="M 4.5 8 L 7 10.5 L 11.5 6"
        fill="none"
        stroke="#FFFFFF"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// STATE B — Training (loss curve)
// ---------------------------------------------------------------------------

interface TrainingStats {
  /** Current cross-entropy loss, e.g. 2.317. */
  loss: number;
  /** Current step. */
  step: number;
  /** Total steps in the planned run. */
  totalSteps: number;
  /**
   * Series of (step, loss) points, plotted as the loss curve. Should be
   * appended to as new datapoints arrive from the SSE stream.
   */
  series: Array<{ step: number; loss: number }>;
  /** Rough ETA in minutes — "About N minutes." */
  etaMinutes: number;
}

export function TrainInProgress({ stats }: { stats: TrainingStats }) {
  const pathD = lossCurvePath(stats.series, stats.totalSteps);

  return (
    <div className="mx-auto min-h-screen max-w-[620px] bg-white px-7 pt-12 pb-14">
      <header className="mb-10 flex items-center gap-3">
        <BrandMark size={32} />
        <div className="text-[11px] uppercase tracking-[0.04em] text-neutral-500">
          Step 3 of 3 · Training
        </div>
      </header>

      <h1 className="mb-3 text-[32px] font-medium leading-[1.1] tracking-[-0.03em] text-neutral-900">
        Training your model.
      </h1>
      <p className="mb-12 text-[14px] leading-[1.5] text-neutral-500">
        About {stats.etaMinutes} minutes. We&apos;ll notify you when it&apos;s
        ready.
      </p>

      <div className="mb-8 rounded-[10px] border-[0.5px] border-neutral-200 bg-neutral-50 px-8 py-7">
        <div className="mb-4 flex items-start justify-between">
          <Stat label="Loss" value={stats.loss.toFixed(3)} />
          <Stat
            label="Step"
            value={`${stats.step.toLocaleString()} / ${stats.totalSteps.toLocaleString()}`}
            align="right"
          />
        </div>

        <svg
          viewBox="0 0 480 180"
          className="block h-[180px] w-full"
          aria-label="Training loss curve"
        >
          {/* Faint horizontal gridlines */}
          <line x1="0" y1="40" x2="480" y2="40" className="pmc-loss-grid" />
          <line x1="0" y1="80" x2="480" y2="80" className="pmc-loss-grid" />
          <line x1="0" y1="120" x2="480" y2="120" className="pmc-loss-grid" />
          <line x1="0" y1="160" x2="480" y2="160" className="pmc-loss-grid" />

          {/*
            The loss curve. On initial render, the path animates from
            stroke-dashoffset:fullLength → 0 over 6 seconds via the
            .pmc-loss-path class. Subsequent updates re-render the path
            instantly (use a `key` derived from series.length to force
            re-mount on the very first draw if you want the dramatic
            entrance only once).
          */}
          <path
            d={pathD}
            fill="none"
            stroke="#1D1D1F"
            strokeWidth="1.25"
            strokeLinecap="round"
            className="pmc-loss-path"
          />
        </svg>
      </div>

      <div className="flex items-center gap-3 rounded-[10px] border-[0.5px] border-neutral-200 bg-white px-[18px] py-3.5">
        <div className="size-1.5 flex-shrink-0 rounded-full bg-[#FF3B30]" />
        <p className="flex-1 text-[13px] leading-[1.5] text-neutral-900">
          You can close this window. The model is training in our enclave — your
          Mac will ping when it&apos;s ready.
        </p>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  align = 'left',
}: {
  label: string;
  value: string;
  align?: 'left' | 'right';
}) {
  return (
    <div className={align === 'right' ? 'text-right' : ''}>
      <div className="mb-1 text-[10px] uppercase tracking-[0.08em] text-neutral-500">
        {label}
      </div>
      <div className="text-[24px] font-medium tracking-[-0.02em] tabular-nums text-neutral-900">
        {value}
      </div>
    </div>
  );
}

/**
 * Generate the SVG path d-attribute for the loss curve given the current
 * series of (step, loss) points. The series is normalized to the 480x180
 * viewBox; loss values are clamped to a reasonable range and inverted
 * (lower loss = lower y-position on screen).
 */
function lossCurvePath(
  series: Array<{ step: number; loss: number }>,
  totalSteps: number,
): string {
  if (series.length === 0) return '';

  // Y-axis: normalize against the series' own min/max loss with a buffer.
  const losses = series.map((p) => p.loss);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const range = Math.max(0.5, maxLoss - minLoss);

  const points = series.map((p) => {
    const x = (p.step / totalSteps) * 480;
    const y = 10 + ((maxLoss - p.loss) / range) * 160;
    return { x, y };
  });

  // Build a smooth-ish path with simple line segments. (Could be upgraded
  // to a Catmull-Rom or cubic-bezier spline for a smoother visual.)
  return points
    .map((pt, i) => (i === 0 ? `M ${pt.x} ${pt.y}` : `L ${pt.x} ${pt.y}`))
    .join(' ');
}
