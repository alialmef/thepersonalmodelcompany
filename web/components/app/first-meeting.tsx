'use client';

import { useState } from 'react';

/**
 * The First Meeting. After the eval's final round, the screen swirls and
 * resolves to white, and the model speaks its first words — alone, centered,
 * no chat chrome, no history.
 *
 * The emotional target: meeting someone for the first time who has known you
 * forever. No transcript is shown because the history isn't in the
 * transcript — it's in the model. They already know you. The first line
 * carries all of it.
 *
 * This is the "loads like magic" moment. In production it is ONE continuous
 * animation from the eval screen, not a route change:
 *   - The eval UI (headline, buttons, progress) fades and falls away.
 *   - The screen washes to clean white.
 *   - The mark SWIRLS in (the only rotation anywhere in the product —
 *     reserved for this single moment).
 *   - A ring expands outward once and dissipates (arrival energy).
 *   - The red dot blooms with overshoot — the heartbeat begins.
 *   - The first line arrives, centered, at 22px (larger than chat text —
 *     this is not a chat message yet, it's THE FIRST THING SAID).
 *   - The composer rises from the bottom last.
 *
 * After the user sends their first reply, this ceremonial space resolves
 * into the working chat (see chat-screen.tsx). The ceremony is for the
 * meeting; the work is for the chat. The magic is precious because it ends.
 *
 * THE OPENING LINE should be GENERATED, not hardcoded — conditioned on the
 * user's style profile so it arrives in their own register. The structure
 * stays constant:
 *   recognition → "known you forever" → humility → hand over control
 * e.g. "there you are. i've read everything you've ever written — i think
 *       i get you. where do you want to start?"
 * A terser user gets something clipped; a warmer user something softer.
 */

interface FirstMeetingProps {
  /** Generated opening line, conditioned on the style profile. */
  openingLine: string;
  /** Called when the user sends their first message — triggers the
   *  resolve into working chat. */
  onFirstMessage: (text: string) => void;
}

export default function FirstMeeting({
  openingLine,
  onFirstMessage,
}: FirstMeetingProps) {
  const [draft, setDraft] = useState('');

  function send() {
    const text = draft.trim();
    if (text) onFirstMessage(text);
  }

  return (
    <div className="flex min-h-screen flex-col bg-white">
      {/* Center stage: the model alone */}
      <div className="flex flex-1 flex-col items-center justify-center px-7 py-16">
        <div className="relative mb-11">
          <svg viewBox="0 0 120 120" width="92" height="92" aria-hidden="true">
            {/* Expanding ring — fires once on arrival */}
            <circle
              cx="60"
              cy="60"
              r="44"
              fill="none"
              stroke="#1D1D1F"
              strokeWidth="0.75"
              className="pmc-fm-ring-pulse"
            />
            {/* The mark — swirls in */}
            <circle
              cx="60"
              cy="60"
              r="44"
              fill="none"
              stroke="#1D1D1F"
              strokeWidth="0.75"
              className="pmc-fm-ring"
            />
            {/* The heartbeat — blooms last, then pulses forever */}
            <circle
              cx="60"
              cy="60"
              r="4"
              fill="#FF3B30"
              className="pmc-fm-dot"
            />
          </svg>
        </div>

        <p className="pmc-fm-line m-0 max-w-[380px] text-center text-[22px] font-normal leading-[1.4] tracking-[-0.02em] text-neutral-900">
          {openingLine}
        </p>
      </div>

      {/* Composer — rises last */}
      <div className="px-[22px] pb-6 pt-3.5 pmc-fm-composer">
        <div className="mx-auto flex max-w-[560px] items-center gap-3 rounded-full border-[0.5px] border-neutral-900/8 bg-neutral-50 px-[18px] py-[13px]">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && send()}
            placeholder="Message your model"
            className="flex-1 bg-transparent text-[14px] text-neutral-900 placeholder:text-neutral-400 focus:outline-none"
          />
          <button
            onClick={send}
            className="flex size-7 items-center justify-center rounded-full bg-neutral-900/80 text-white transition-colors hover:bg-neutral-900"
            aria-label="Send"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 19V5M12 5l-6 6M12 5l6 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
