"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

const SUGGESTED_PROMPTS = [
  "draft a text to my best friend about plans for this weekend",
  "what's the last thing i wrote about feeling stuck?",
  "summarize the way i've been thinking about my work this month",
  "write a short note to myself about what i'd want to remember from today",
];

/**
 * Act 5 — chat with your model.
 *
 * The first emotional beat is the "model speaks first" moment: on initial
 * load with no history, we fire a hidden first-contact request and let the
 * model introduce itself in the user's voice. Only the model's response is
 * visible in the transcript.
 *
 * After that, the surface is intentionally minimal — one bubble at a time,
 * iMessage-rhythm, no chrome to draw attention away from the conversation.
 */
export default function ChatPage() {
  const [firstContactFired, setFirstContactFired] = useState(false);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );

  const { messages, sendMessage, status, error } = useChat({
    transport,
  });

  // Visible messages exclude the hidden first-contact trigger.
  const visibleMessages = useMemo(
    () => messages.filter((m) => m.id !== "first-contact"),
    [messages],
  );

  // Trigger the model's opening line exactly once on mount.
  useEffect(() => {
    if (!firstContactFired && messages.length === 0) {
      setFirstContactFired(true);
      sendMessage(
        {
          // Hidden — stripped from the visible transcript above.
          messageId: "first-contact",
          text: "__first_contact__",
        },
        { body: { firstContact: true } },
      );
    }
  }, [firstContactFired, messages.length, sendMessage]);

  // Auto-scroll to bottom on every new message / streaming chunk.
  const bottomRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isWaiting = status === "submitted" || status === "streaming";
  const isFirstContactPending =
    firstContactFired && visibleMessages.length === 0 && isWaiting;

  return (
    <main className="chat-root">
      <header className="chat-header">
        <Link href="/" className="chat-brand">
          The Personal Model Company
        </Link>
        <Link href="/connect" className="chat-settings" title="Sources">
          ⋯
        </Link>
      </header>

      <section className="chat-stream">
        {isFirstContactPending && <ThinkingBubble label="learning who you are" />}

        {visibleMessages.map((m) => (
          <MessageRow key={m.id} role={m.role} parts={m.parts as MessagePart[]} />
        ))}

        {visibleMessages.length > 0 && isWaiting && (
          <ThinkingBubble label="thinking" />
        )}

        {error && (
          <div className="chat-error">
            something went wrong — {error.message}
          </div>
        )}

        <div ref={bottomRef} />
      </section>

      {visibleMessages.length === 1 && !isWaiting && (
        <Suggestions
          prompts={SUGGESTED_PROMPTS}
          onPick={(p) => sendMessage({ text: p })}
        />
      )}

      <Composer
        disabled={isWaiting}
        onSubmit={(text) => sendMessage({ text })}
      />
    </main>
  );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

interface MessagePart {
  type: string;
  text?: string;
  state?: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
}

function messagePartsToText(parts: MessagePart[]): string {
  return parts
    .filter((p) => p.type === "text" && typeof p.text === "string")
    .map((p) => p.text as string)
    .join("");
}

/**
 * The pretty label for a tool call. AI SDK v6 uses 'tool-<toolName>' as the
 * part type, so we strip the prefix and humanize.
 */
function toolLabel(type: string): string {
  const name = type.replace(/^tool-/, "");
  return name.replace(/_/g, " ");
}

// ---------------------------------------------------------------------------
// components
// ---------------------------------------------------------------------------

function ChatBubble({
  role,
  text,
}: {
  role: string;
  text: string;
}) {
  const isUser = role === "user";
  return (
    <div className={`chat-row ${isUser ? "chat-row--user" : "chat-row--model"}`}>
      <div
        className={`chat-bubble ${
          isUser ? "chat-bubble--user" : "chat-bubble--model"
        }`}
      >
        {text}
      </div>
    </div>
  );
}

/**
 * Renders a single message's parts in order. Each text part becomes a
 * bubble; each tool-call part becomes a small "used <tool> · <state>"
 * indicator in the model column. This way you see the model's reasoning
 * interleaved with its prose, the way you'd expect from an agent.
 */
function MessageRow({ role, parts }: { role: string; parts: MessagePart[] }) {
  return (
    <>
      {parts.map((p, i) => {
        if (p.type === "text") {
          return <ChatBubble key={i} role={role} text={p.text ?? ""} />;
        }
        if (p.type.startsWith("tool-")) {
          return <ToolBubble key={i} part={p} />;
        }
        return null;
      })}
    </>
  );
}

function ToolBubble({ part }: { part: MessagePart }) {
  const label = toolLabel(part.type);
  const state = part.state ?? "running";

  // Show one-line input summary so the chat reads as agentic activity, not
  // opaque magic. Output stays collapsed unless the user asks for it.
  const inputSummary = part.input ? summarizeInput(part.input) : "";

  return (
    <div className="chat-row chat-row--model">
      <div className="chat-tool">
        <span className="chat-tool-icon">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path
              d="M3 6l2 2 4-5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <span className="chat-tool-name">{label}</span>
        {inputSummary && (
          <span className="chat-tool-input">{inputSummary}</span>
        )}
        <span className={`chat-tool-state chat-tool-state--${state}`}>
          {state === "output-available"
            ? "done"
            : state === "input-streaming" || state === "input-available"
            ? "running"
            : state}
        </span>
      </div>
    </div>
  );
}

function summarizeInput(input: unknown): string {
  if (typeof input !== "object" || input === null) return "";
  const obj = input as Record<string, unknown>;
  // Pick the first meaningful string value
  for (const v of Object.values(obj)) {
    if (typeof v === "string" && v.length > 0) {
      return v.length > 60 ? v.slice(0, 60) + "…" : v;
    }
  }
  return "";
}

function ThinkingBubble({ label }: { label: string }) {
  return (
    <div className="chat-row chat-row--model">
      <div className="chat-bubble chat-bubble--model chat-bubble--thinking">
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-label">{label}</span>
      </div>
    </div>
  );
}

function Suggestions({
  prompts,
  onPick,
}: {
  prompts: string[];
  onPick: (text: string) => void;
}) {
  return (
    <div className="chat-suggestions">
      {prompts.map((p) => (
        <button
          key={p}
          type="button"
          className="chat-suggestion"
          onClick={() => onPick(p)}
        >
          {p}
        </button>
      ))}
    </div>
  );
}

function Composer({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (text: string) => void;
}) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-resize the textarea up to a reasonable cap.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setText("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-composer">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="say something"
        rows={1}
        disabled={disabled}
        className="chat-composer-input"
      />
      <button
        type="submit"
        disabled={disabled || !text.trim()}
        className="chat-composer-send"
        aria-label="Send"
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path
            d="M2 9L16 2L12 9L16 16L2 9Z"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </form>
  );
}
