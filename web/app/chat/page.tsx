"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useMemo, useState } from "react";

import ChatScreen from "@/components/app/chat-screen";

interface Message {
  id: string;
  role: "user" | "model";
  text: string;
  streaming?: boolean;
}

interface MessagePart {
  type: string;
  text?: string;
}

/**
 * The working chat surface. The ceremony of /first-meeting has resolved
 * into this — quiet bubbles, real streaming, agent action cards (TODO).
 *
 * We read the opening line + first reply that /first-meeting handed off via
 * sessionStorage, seed the transcript with them, and immediately fire the
 * first turn to the model. From there it's a normal chat loop.
 */
export default function ChatPage() {
  const [seeded, setSeeded] = useState(false);
  const [carry, setCarry] = useState<{ opening: string; firstReply: string } | null>(null);

  const transport = useMemo(
    () => new DefaultChatTransport({ api: "/api/chat" }),
    [],
  );

  const { messages, sendMessage, status, error } = useChat({ transport });

  // On first mount, lift the handoff from /first-meeting if present.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("pmc-first-meeting");
      if (raw) {
        const parsed = JSON.parse(raw) as { opening: string; firstReply: string };
        setCarry(parsed);
        sessionStorage.removeItem("pmc-first-meeting");
      }
    } catch {
      /* private mode — ignore */
    }
    setSeeded(true);
  }, []);

  // After the carry-over is loaded, send the user's first reply so the
  // model responds in-context. The opening line is already known on the
  // page side; we don't send it back through the model.
  useEffect(() => {
    if (!seeded || !carry) return;
    sendMessage({ text: carry.firstReply });
  }, [seeded, carry, sendMessage]);

  const isWaiting = status === "submitted" || status === "streaming";

  // Compose visible transcript: carried opening (model) + the user's first
  // reply, then everything the AI SDK has accumulated.
  const visible: Message[] = useMemo(() => {
    const out: Message[] = [];
    if (carry) {
      out.push({ id: "opening", role: "model", text: carry.opening });
      out.push({ id: "first-reply", role: "user", text: carry.firstReply });
    }
    for (const m of messages) {
      // Skip the first user message we already rendered as carry.firstReply
      // (AI SDK's records start at the first user send, which IS the carry).
      if (
        carry &&
        m.role === "user" &&
        partsToText(m.parts as MessagePart[]) === carry.firstReply &&
        !out.some((o) => o.id === m.id)
      ) {
        continue;
      }
      out.push({
        id: m.id,
        role: m.role === "user" ? "user" : "model",
        text: partsToText(m.parts as MessagePart[]),
        streaming: isWaiting && m === messages[messages.length - 1] && m.role !== "user",
      });
    }
    return out;
  }, [carry, messages, isWaiting]);

  return (
    <>
      <ChatScreen
        messages={visible}
        onSend={(text) => sendMessage({ text })}
        onOpenSettings={() => {
          // Settings drawer is a follow-up — for now a no-op.
        }}
      />
      {error && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 rounded-lg bg-red-50 px-4 py-2 text-[12px] text-red-700">
          {error.message}
        </div>
      )}
    </>
  );
}

function partsToText(parts: MessagePart[]): string {
  return parts
    .filter((p) => p.type === "text" && typeof p.text === "string")
    .map((p) => p.text as string)
    .join("");
}
