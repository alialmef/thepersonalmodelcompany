"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import CapabilityPulse from "@/components/app/capability-pulse";
import ChatScreen from "@/components/app/chat-screen";
import { SettingsDrawer } from "@/components/app/settings-drawer";
import { chatStream, type ChatMessage } from "@/lib/api/client";
import { useUser } from "@/hooks/use-user";
import { isTauri } from "@/lib/runtime";

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

type DirectStatus = "ready" | "submitted" | "streaming" | "error";

/**
 * The working chat surface. The ceremony of /first-meeting has resolved
 * into this — quiet bubbles, real streaming, agent action cards (TODO).
 *
 * We read the opening line + first reply that /first-meeting handed off via
 * sessionStorage, seed the transcript with them, and immediately fire the
 * first turn to the model. From there it's a normal chat loop.
 */
export default function ChatPage() {
  const { user } = useUser();
  const userId = user?.pmcUserId ?? "";
  const [seeded, setSeeded] = useState(false);
  const [carry, setCarry] = useState<{ opening: string; firstReply: string } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inApp, setInApp] = useState(false);
  const [runtimeChecked, setRuntimeChecked] = useState(false);
  const [directMessages, setDirectMessages] = useState<Message[]>([]);
  const [directStatus, setDirectStatus] = useState<DirectStatus>("ready");
  const [directError, setDirectError] = useState<string | null>(null);
  const directMessagesRef = useRef<Message[]>([]);
  const carrySentRef = useRef(false);

  const transport = useMemo(
    () => new DefaultChatTransport({ api: "/api/chat" }),
    [],
  );

  const { messages, sendMessage, status, error } = useChat({ transport });

  useEffect(() => {
    setInApp(isTauri());
    setRuntimeChecked(true);
  }, []);

  const setDirectTranscript = useCallback((next: Message[]) => {
    directMessagesRef.current = next;
    setDirectMessages(next);
  }, []);

  const directContextMessages = useCallback(
    (messagesForTurn: Message[]): ChatMessage[] => {
      const out: ChatMessage[] = [];
      if (carry) {
        out.push({ role: "assistant", content: carry.opening });
        out.push({ role: "user", content: carry.firstReply });
      }
      for (const msg of messagesForTurn) {
        out.push({
          role: msg.role === "user" ? "user" : "assistant",
          content: msg.text,
        });
      }
      return out.filter((msg) => msg.content.trim().length > 0);
    },
    [carry],
  );

  const sendDirectMessage = useCallback(
    async (text: string, opts: { renderUser?: boolean } = {}) => {
      const renderUser = opts.renderUser ?? true;
      const trimmed = text.trim();
      if (!trimmed) return;
      if (!userId) {
        setDirectError("No local user is available yet.");
        setDirectStatus("error");
        return;
      }

      setDirectError(null);
      setDirectStatus("submitted");

      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        text: trimmed,
      };
      const beforeAssistant = renderUser
        ? [...directMessagesRef.current, userMessage]
        : [...directMessagesRef.current];
      const assistantId = `model-${Date.now()}`;
      const assistantMessage: Message = {
        id: assistantId,
        role: "model",
        text: "",
        streaming: true,
      };
      setDirectTranscript([...beforeAssistant, assistantMessage]);
      setDirectStatus("streaming");

      let response = "";
      try {
        for await (const delta of chatStream({
          model: userId,
          user: userId,
          messages: directContextMessages(beforeAssistant),
          max_tokens: 512,
          temperature: 0.7,
          stream: true,
        })) {
          response += delta;
          setDirectTranscript([
            ...beforeAssistant,
            {
              ...assistantMessage,
              text: response,
              streaming: true,
            },
          ]);
        }
        setDirectTranscript([
          ...beforeAssistant,
          {
            ...assistantMessage,
            text: response,
            streaming: false,
          },
        ]);
        setDirectStatus("ready");
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        setDirectError(message);
        setDirectTranscript([
          ...beforeAssistant,
          {
            ...assistantMessage,
            text:
              "Your model isn't connected right now. Make sure the PMC backend is running.",
            streaming: false,
          },
        ]);
        setDirectStatus("error");
      }
    },
    [directContextMessages, setDirectTranscript, userId],
  );

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
    if (!seeded || !carry || !runtimeChecked || carrySentRef.current) return;
    if (inApp && !userId) return;
    carrySentRef.current = true;
    if (inApp) {
      void sendDirectMessage(carry.firstReply, { renderUser: false });
    } else {
      sendMessage({ text: carry.firstReply });
    }
  }, [
    seeded,
    carry,
    runtimeChecked,
    inApp,
    userId,
    sendDirectMessage,
    sendMessage,
  ]);

  const isWaiting = inApp
    ? directStatus === "submitted" || directStatus === "streaming"
    : status === "submitted" || status === "streaming";

  // Compose visible transcript: carried opening (model) + the user's first
  // reply, then everything the AI SDK has accumulated.
  const visible: Message[] = useMemo(() => {
    const out: Message[] = [];
    if (carry) {
      out.push({ id: "opening", role: "model", text: carry.opening });
      out.push({ id: "first-reply", role: "user", text: carry.firstReply });
    }
    if (inApp) {
      out.push(...directMessages);
      return out;
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
  }, [carry, inApp, directMessages, messages, isWaiting]);

  return (
    <>
      <ChatScreen
        messages={visible}
        onSend={(text) => {
          if (inApp) {
            void sendDirectMessage(text);
          } else {
            sendMessage({ text });
          }
        }}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
      {/* Ambient indicator showing which providers the backend is
          actually wired to — catches the dev/prod confusion where
          you think you're on hosted Together inference but you're
          actually on a local mock. */}
      <CapabilityPulse />
      {!inApp && error && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 rounded-lg bg-red-50 px-4 py-2 text-[12px] text-red-700">
          {error.message}
        </div>
      )}
      {directError && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 rounded-lg bg-red-50 px-4 py-2 text-[12px] text-red-700">
          {directError}
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
