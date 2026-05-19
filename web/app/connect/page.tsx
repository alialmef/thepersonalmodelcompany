"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";
import { SourceRow } from "@/components/source-row";
import { NativeSourceRow } from "@/components/native-source-row";
import { ItemsCounter } from "@/components/items-counter";
import { DEMO_USER_ID } from "@/lib/demo-user";
import { isTauri } from "@/lib/runtime";

/**
 * Act 2 — Gather.
 *
 * The hardest emotional beat: handing over your writing. The page is calm,
 * one source at a time, and the privacy line is part of the page (not a footer
 * disclaimer). The live counter is the only animated thing — it's the signal
 * that something is happening on their side, not ours.
 */
export default function ConnectPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [inApp, setInApp] = useState(false);
  const userId = DEMO_USER_ID;
  const refresh = () => setRefreshKey((k) => k + 1);

  useEffect(() => {
    setInApp(isTauri());
  }, []);

  return (
    <main className="min-h-screen flex flex-col">
      <header className="px-6 md:px-10 py-6 flex items-center justify-between">
        <Link href="/">
          <Brand size="small" className="text-muted hover:text-foreground transition-colors" />
        </Link>
        <span className="text-[15px] text-muted">{userId}</span>
      </header>

      <section className="flex-1 max-w-2xl w-full mx-auto px-6 py-12 md:py-16">
        <div className="mb-12">
          <h1 className="text-3xl md:text-4xl tracking-tight font-medium mb-3">
            Let&apos;s start with your writing.
          </h1>
          <p className="text-[17px] text-muted leading-relaxed max-w-[44ch]">
            The more we have, the better it sounds like you.
          </p>
        </div>

        <div>
          <SourceRow
            label="Apple Mail / Outlook"
            description="Upload .mbox"
            kind="email_mbox"
            accept=".mbox,application/mbox"
            userId={userId}
            onChange={refresh}
            identityPrompt={{
              label: "Your email addresses (comma-separated)",
              placeholder: "you@example.com",
              field: "userEmails",
            }}
          />

          {inApp ? (
            <NativeSourceRow
              label="iMessage"
              kind="imessage"
              userId={userId}
              onChange={refresh}
            />
          ) : (
            <SourceRow
              label="iMessage"
              description="Show how"
              kind="imessage"
              accept=".db"
              userId={userId}
              onChange={refresh}
              instructions={
                "iMessage ingestion requires the Mac app. Download it for the full experience.\n\n" +
                "If you're on the Mac app already, restart it — Tauri detection failed."
              }
            />
          )}

          <SourceRow
            label="WhatsApp"
            description="Upload chat .txt"
            kind="whatsapp"
            accept=".txt,text/plain"
            userId={userId}
            onChange={refresh}
            identityPrompt={{
              label: "Your name in the chat (as it appears)",
              placeholder: "Alex",
              field: "userNames",
            }}
          />

          <SourceRow
            label="Notes"
            description="Upload .md or .txt"
            kind="text"
            accept=".md,.txt,.markdown,text/plain,text/markdown"
            userId={userId}
            onChange={refresh}
          />

          <SourceRow
            label="Documents"
            description="Upload .pdf or .docx"
            kind="document"
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            userId={userId}
            onChange={refresh}
          />
        </div>

        <div className="mt-12 mb-16">
          <p className="text-[15px]">
            <ItemsCounter userId={userId} refreshKey={refreshKey} />
          </p>
        </div>

        <div className="border-t border-border pt-8 mb-12">
          <p className="text-[13px] uppercase tracking-[0.18em] text-muted mb-3">
            Privacy
          </p>
          <p className="text-[15px] text-muted leading-relaxed max-w-[58ch]">
            Everything stays on your tenant. Encrypted at rest. Only you can read
            it. Delete a source and your model retrains from what remains.
          </p>
        </div>

        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="text-[15px] text-muted hover:text-foreground transition-colors"
          >
            ← Back
          </Link>
          <Link href="/curate">
            <Button>Continue</Button>
          </Link>
        </div>
      </section>
    </main>
  );
}
