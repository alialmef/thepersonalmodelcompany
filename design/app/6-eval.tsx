'use client';

import { useState } from 'react';
import { Check, X } from 'lucide-react';
import { BrandMark } from '@/components/shared/brand-mark';

/**
 * The Evaluation screen. Shown the moment training completes (after the
 * macOS notification brings the user back). The user judges the model's
 * responses to a handful of real situations — does this sound like me?
 *
 * What it really is: a preference-data collection pipeline disguised as a
 * reveal. Each accept / reject / edit is training signal (DPO-grade). The
 * user is doing RLHF on their own model and it feels like quality control.
 *
 * It also completes the "personal AI lab" arc: the user has now run
 * curation, training, AND evaluation — the whole pipeline.
 *
 * FLOW: 5 rounds. Each round:
 *   1. A grounded situation appears (ideally drawn from the user's own
 *      patterns — a real contact, a real email type).
 *   2. The model responds live, streaming word-by-word.
 *   3. Three affordances: "Not quite" / "That's me" / "Edit instead".
 *
 * After round 5, the screen transitions (swirls) into the first-meeting
 * chat arrival. See first-meeting.tsx.
 *
 * BACKEND CONTRACT:
 *   - Situations come from /v1/users/{id}/eval/prompts — array of
 *     { id, situation: string }.
 *   - Each model response is streamed from /v1/chat/completions with the
 *     situation as input, stream: true.
 *   - Judgments POST to /v1/users/{id}/eval/judgments:
 *       { promptId, verdict: 'accept' | 'reject' | 'edit',
 *         editedText?: string, reason?: string }
 *   - These judgments feed the next retrain and the "sounds like you"
 *     score shown later.
 */

interface EvalRound {
  id: string;
  situation: string;
  /** The model's response. In production this streams in live. */
  response: string;
}

type Verdict = 'accept' | 'reject' | 'edit';

const REJECT_REASONS = ['too formal', 'not my words', 'wrong tone', 'too long'];

export default function EvalScreen({
  rounds,
  onJudge,
  onComplete,
}: {
  rounds: EvalRound[];
  onJudge: (
    promptId: string,
    verdict: Verdict,
    extra?: { editedText?: string; reason?: string },
  ) => void;
  onComplete: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [showReasons, setShowReasons] = useState(false);
  const round = rounds[index];
  const total = rounds.length;

  function advance() {
    if (index + 1 >= total) {
      onComplete();
    } else {
      setIndex(index + 1);
      setShowReasons(false);
    }
  }

  function handleAccept() {
    onJudge(round.id, 'accept');
    advance();
  }

  function handleReject(reason?: string) {
    onJudge(round.id, 'reject', reason ? { reason } : undefined);
    advance();
  }

  return (
    <div className="mx-auto min-h-screen max-w-[560px] bg-white px-7 pt-11 pb-12">
      <header className="mb-9 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BrandMark size={28} />
          <span className="text-[11px] uppercase tracking-[0.04em] text-neutral-500">
            Your model is ready
          </span>
        </div>
        <ProgressDots current={index + 1} total={total} />
      </header>

      <h1 className="mb-2.5 text-[28px] font-medium leading-[1.15] tracking-[-0.03em] text-neutral-900 pmc-anim-fade">
        Does this sound like you?
      </h1>
      <p
        className="mb-10 text-[14px] leading-[1.5] text-neutral-500 pmc-anim-fade"
        style={{ animationDelay: '0.2s' }}
      >
        Tell your model when it gets you right. Every answer makes it sharper.
      </p>

      {/* The situation */}
      <div className="mb-[18px] pmc-anim-fade-up" style={{ animationDelay: '0.5s' }}>
        <div className="mb-2.5 text-[10px] uppercase tracking-[0.08em] text-neutral-500">
          The situation
        </div>
        <div className="flex justify-end">
          <div className="max-w-[80%] rounded-2xl bg-neutral-100 px-4 py-3 text-[14px] leading-[1.45] text-neutral-900">
            {round.situation}
          </div>
        </div>
      </div>

      {/* The model's live response */}
      <div className="mb-9 pmc-anim-fade-up" style={{ animationDelay: '1.0s' }}>
        <div className="mb-2.5 flex items-center gap-2">
          <BrandMark size={15} />
          <span className="text-[10px] uppercase tracking-[0.08em] text-neutral-500">
            Your model
          </span>
        </div>
        <div className="flex justify-start">
          <div className="max-w-[80%] rounded-2xl bg-neutral-900 px-4 py-3 text-[14px] leading-[1.5] text-white">
            {/*
              In production, stream this token-by-token. The <StreamingText>
              helper (see chat-screen.tsx) handles the live cascade. Here
              it's the final string.
            */}
            {round.response}
          </div>
        </div>
      </div>

      {/* Accept / reject */}
      {!showReasons ? (
        <div className="flex gap-3 pmc-anim-fade" style={{ animationDelay: '2.4s' }}>
          <button
            onClick={() => setShowReasons(true)}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border-[0.5px] border-neutral-900/15 bg-white py-4 text-[14px] font-medium text-neutral-900 transition-colors hover:bg-neutral-50"
          >
            <X className="size-4" /> Not quite
          </button>
          <button
            onClick={handleAccept}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border-[0.5px] border-neutral-900/15 bg-white py-4 text-[14px] font-medium text-neutral-900 transition-colors hover:border-neutral-900 hover:bg-neutral-900 hover:text-white"
          >
            <Check className="size-4" /> That&apos;s me
          </button>
        </div>
      ) : (
        // Optional reason chips. Tapping one captures it; "skip" advances
        // with no reason. Either way the reject is already counted.
        <div className="flex flex-col gap-3">
          <div className="text-center text-[12px] text-neutral-500">
            What was off? (optional)
          </div>
          <div className="flex flex-wrap justify-center gap-2">
            {REJECT_REASONS.map((reason) => (
              <button
                key={reason}
                onClick={() => handleReject(reason)}
                className="rounded-full border-[0.5px] border-neutral-900/15 bg-white px-3.5 py-2 text-[12px] text-neutral-900 transition-colors hover:bg-neutral-50"
              >
                {reason}
              </button>
            ))}
            <button
              onClick={() => handleReject()}
              className="rounded-full px-3.5 py-2 text-[12px] text-neutral-500 underline"
            >
              skip
            </button>
          </div>
        </div>
      )}

      <button
        onClick={() => onJudge(round.id, 'edit')}
        className="mx-auto mt-6 block text-[12px] text-neutral-500 underline"
      >
        Edit the response instead
      </button>
    </div>
  );
}

function ProgressDots({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-1.5">
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          className={`h-[3px] w-[18px] rounded-sm ${
            i < current ? 'bg-neutral-900' : 'bg-neutral-900/15'
          }`}
        />
      ))}
      <span className="ml-1.5 text-[11px] tabular-nums text-neutral-500">
        {current} of {total}
      </span>
    </div>
  );
}
