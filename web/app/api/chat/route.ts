/**
 * /api/chat — server route that proxies AI SDK chat traffic to the PMC backend.
 *
 * The frontend's useChat() POSTs here; this route hands the messages to the
 * PMC backend's OpenAI-compatible endpoint and streams the response back.
 * Memory retrieval + identity-prompt injection happen on the backend side
 * (see pmc/serve/server.py::_prepared_messages), so this route stays thin.
 *
 * Two special concerns handled here:
 *   1. "First-contact" mode — when the request body has `firstContact: true`,
 *      we replace the user's (hidden) trigger with the introduction prompt
 *      from pmc.memory.identity.build_first_contact_message. The model then
 *      generates its opening line in the user's voice.
 *   2. Graceful degradation — if the backend isn't reachable, return a
 *      friendly stream explaining what's wrong instead of a 500.
 */

import { convertToModelMessages, stepCountIs, streamText, type UIMessage } from "ai";
import { pmc, backendReachable } from "@/lib/pmc-client";
import { chatTools } from "@/lib/tools";
import { DEMO_USER_ID } from "@/lib/demo-user";

export const runtime = "nodejs";
export const maxDuration = 120;

interface ChatRequestBody {
  messages: UIMessage[];
  userId?: string;
  firstContact?: boolean;
}

export async function POST(req: Request) {
  const body = (await req.json()) as ChatRequestBody;
  const userId = body.userId ?? DEMO_USER_ID;

  if (!(await backendReachable())) {
    return offlineResponse();
  }

  // First contact: the user hasn't typed anything yet — the model speaks first.
  // We inject a single hidden user-role instruction that tells the model to
  // introduce itself. The frontend strips this from the visible transcript.
  const messages = body.firstContact
    ? [
        {
          id: "first-contact",
          role: "user" as const,
          parts: [
            {
              type: "text" as const,
              text:
                "Write a single short opening message introducing yourself as my " +
                "personal model. Mention 2 or 3 things you've observed about how I " +
                "write. End with a question that invites me to start a conversation. " +
                "Address me as 'you', never as 'I'. One short paragraph.",
            },
          ],
        },
      ]
    : body.messages;

  const result = streamText({
    model: pmc(userId),
    messages: await convertToModelMessages(messages),
    tools: chatTools,
    // Allow up to 5 tool-call → response round trips before stopping.
    // Cap is here so a misbehaving model can't loop indefinitely.
    stopWhen: stepCountIs(5),
  });

  return result.toUIMessageStreamResponse();
}

function offlineResponse(): Response {
  // Format a minimal AI-SDK-compatible data stream that yields one assistant
  // message. The chat UI renders this as a regular bubble.
  const encoder = new TextEncoder();
  const text =
    "Your model isn't connected right now. Make sure the PMC backend is " +
    "running locally (./scripts/dev.sh) or that PMC_API_URL points at a " +
    "deployed instance.";
  const stream = new ReadableStream({
    start(controller) {
      const msgId = `offline-${Date.now()}`;
      // AI SDK v6 UIMessage stream protocol
      controller.enqueue(
        encoder.encode(
          formatLine({ type: "start" }) +
            formatLine({ type: "start-step" }) +
            formatLine({ type: "text-start", id: msgId }) +
            formatLine({ type: "text-delta", id: msgId, delta: text }) +
            formatLine({ type: "text-end", id: msgId }) +
            formatLine({ type: "finish-step" }) +
            formatLine({ type: "finish" }),
        ),
      );
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "x-vercel-ai-ui-message-stream": "v1",
    },
  });
}

function formatLine(payload: Record<string, unknown>): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}
