'use client';

import { useEffect, useState } from 'react';
import { Settings } from 'lucide-react';
import { BrandMark } from '@/components/shared/brand-mark';

/**
 * The working chat. After the first-meeting ceremony, the user sends a
 * reply, and the centered ceremonial space resolves into this: a normal,
 * quiet chat.
 *
 * Chat shape (resolving the brief's open question): iMessage-style bubbles,
 * but QUIET ones. The model is meant to feel like a person you text — and
 * the eval already established the texting metaphor. So: tight radius,
 * system greys, model in near-black to carry weight. NOT playful iMessage
 * blue. Quiet texting.
 *
 * The header is minimal — mark + dot + "Your model" left, a single settings
 * gear right. That gear opens the settings drawer where the whole brief's
 * settings live (retrain, export bundle, manage sources, API key, delete).
 * One icon. No nav chrome. Linear-quiet.
 *
 * AGENT ACTIONS (Standard / Frontier tiers): when the model takes an action
 * (sends an email, runs code, schedules), it should render inline as a
 * compact action card in the message stream, not a separate log. That card
 * is a TODO — see <ActionCard> stub at the bottom.
 */

type Role = 'user' | 'model';

interface Message {
  id: string;
  role: Role;
  text: string;
  /** When true, the text streams in token-by-token on mount. */
  streaming?: boolean;
}

export default function ChatScreen({
  messages,
  onSend,
  onOpenSettings,
}: {
  messages: Message[];
  onSend: (text: string) => void;
  onOpenSettings: () => void;
}) {
  const [draft, setDraft] = useState('');

  function send() {
    const text = draft.trim();
    if (text) {
      onSend(text);
      setDraft('');
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-white">
      {/* Header */}
      <header className="flex items-center justify-between border-b-[0.5px] border-neutral-900/8 px-[22px] py-3.5">
        <div className="flex items-center gap-2.5">
          <BrandMark size={20} />
          <span className="text-[13px] font-medium tracking-[-0.01em] text-neutral-900">
            Your model
          </span>
        </div>
        <button
          onClick={onOpenSettings}
          className="text-neutral-500 transition-colors hover:text-neutral-900"
          aria-label="Settings"
        >
          <Settings className="size-[18px]" strokeWidth={1.5} />
        </button>
      </header>

      {/* Message list */}
      <div className="flex flex-1 flex-col gap-4 px-[22px] py-7">
        {messages.map((msg) =>
          msg.role === 'user' ? (
            <div key={msg.id} className="flex justify-end">
              <div className="max-w-[75%] rounded-2xl bg-neutral-100 px-[15px] py-2.5 text-[14px] leading-[1.45] text-neutral-900">
                {msg.text}
              </div>
            </div>
          ) : (
            <div key={msg.id} className="flex items-end justify-start gap-2">
              <BrandMark size={16} className="mb-1 shrink-0" />
              <div className="max-w-[78%] rounded-2xl bg-neutral-900 px-[15px] py-2.5 text-[14px] leading-[1.5] text-white">
                {msg.streaming ? <StreamingText text={msg.text} /> : msg.text}
              </div>
            </div>
          ),
        )}
      </div>

      {/* Composer */}
      <div className="border-t-[0.5px] border-neutral-900/8 px-[22px] pb-6 pt-3.5">
        <div className="mx-auto flex max-w-[640px] items-center gap-3 rounded-full border-[0.5px] border-neutral-900/8 bg-neutral-50 px-[18px] py-3">
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

/**
 * Streams text in word-by-word — the model "typing." This is the same
 * generation-feel used in the hero, the curate style profile, and the eval.
 * Token streaming in production should be driven by the SSE stream from
 * /v1/chat/completions (stream: true), appending tokens as they arrive
 * rather than replaying a known string. This component is the
 * known-string fallback / demo version.
 */
function StreamingText({ text }: { text: string }) {
  const [shown, setShown] = useState('');

  useEffect(() => {
    const words = text.split(' ');
    let i = 0;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      if (i >= words.length) return;
      setShown(words.slice(0, i + 1).join(' '));
      i++;
      timer = setTimeout(tick, 85 + Math.random() * 65);
    };
    tick();
    return () => clearTimeout(timer);
  }, [text]);

  return <>{shown}</>;
}

/**
 * TODO — agent action card. When the model performs an action (Standard /
 * Frontier tiers), render this inline in the message stream instead of a
 * plain bubble. Compact, quiet, with the action verb, target, and a
 * confirm/undo affordance where appropriate.
 *
 * Example shape:
 *   <ActionCard
 *     verb="Drafted email"
 *     target="to marcus@example.com"
 *     preview="hey marcus — heater's been out since..."
 *     status="awaiting_confirm"   // requires explicit user OK to send
 *   />
 *
 * Per the product's safety posture, irreversible actions (send, publish,
 * purchase) must require explicit user confirmation in this card before
 * the model executes them.
 */
