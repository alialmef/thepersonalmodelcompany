"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";

/**
 * Sign-in. One input, one button. Magic-link sent.
 *
 * After submit, the form stays (so the user remembers what email they used)
 * and a confirmation message appears below. If the request comes back with
 * `dev.link` (no Resend configured), we show a "Open dev link" affordance
 * so local development works without setting up email DNS.
 *
 * If the user arrived here via `/sign-in?expired=1` (the verify route's
 * fallback for missing/expired/used tokens), we render a small banner.
 *
 * Honors `?next=<path>` — preserved through the email round-trip via the
 * magic-link URL, so post-sign-in the user lands on the page they wanted.
 */

interface MagicLinkResponse {
  ok?: boolean;
  dev?: { link: string };
}

function SignInInner() {
  const params = useSearchParams();
  const expired = params.get("expired") === "1";
  const next = params.get("next") ?? "/welcome";

  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [devLink, setDevLink] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email) return;
    setState("sending");
    setError(null);
    setDevLink(null);
    try {
      const r = await fetch("/api/auth/magic-link", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, next }),
      });
      if (!r.ok) throw new Error("Could not send the link. Try again.");
      const data = (await r.json()) as MagicLinkResponse;
      if (data.dev?.link) {
        // Append next= so post-verify lands on the right page even in dev.
        const withNext = appendQuery(data.dev.link, { next });
        setDevLink(withNext);
      }
      setState("sent");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setState("error");
    }
  }

  return (
    <main className="min-h-screen flex flex-col">
      <header className="px-6 md:px-10 py-6">
        <Link href="/" className="inline-block">
          <Brand size="small" className="text-muted hover:text-foreground transition-colors" />
        </Link>
      </header>

      <section className="flex-1 flex flex-col items-center justify-center px-6">
        <div className="w-full max-w-sm">
          <h1 className="text-3xl md:text-4xl tracking-tight font-medium mb-3 text-center">
            Sign in
          </h1>
          <p className="text-[15px] text-muted text-center mb-10">
            Enter your email. We&apos;ll send a link.
          </p>

          {expired && (
            <p className="mb-6 text-[13px] text-muted text-center">
              That link is expired or already used. Send a new one.
            </p>
          )}

          <form onSubmit={onSubmit} className="space-y-4">
            <input
              type="email"
              required
              autoFocus
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              disabled={state === "sending" || state === "sent"}
              className="w-full h-12 px-4 rounded-xl bg-subtle text-foreground placeholder:text-muted/70 outline-none border border-border focus:border-foreground/40 transition-colors disabled:opacity-60"
            />
            <Button
              type="submit"
              className="w-full"
              disabled={state === "sending" || state === "sent"}
            >
              {state === "sending" ? "Sending…" : state === "sent" ? "Sent" : "Continue"}
            </Button>
          </form>

          {state === "sent" && (
            <div className="mt-8 text-center">
              <p className="text-[14px] text-muted leading-relaxed">
                We sent a link to <span className="text-foreground">{email}</span>.
                <br />
                Open it on this device to sign in.
              </p>
              {devLink && (
                <p className="mt-6 text-[12px] text-muted">
                  Dev mode (no email service configured):{" "}
                  <a
                    href={devLink}
                    className="underline text-foreground"
                  >
                    open the link here
                  </a>
                  .
                </p>
              )}
            </div>
          )}

          {error && (
            <p className="mt-6 text-[14px] text-foreground text-center">
              {error}
            </p>
          )}
        </div>
      </section>

      <footer className="px-6 py-8 text-[13px] text-muted text-center">
        <Link href="/" className="hover:text-foreground transition-colors">
          ← Back
        </Link>
      </footer>
    </main>
  );
}

function appendQuery(url: string, extra: Record<string, string>): string {
  try {
    const u = new URL(url);
    for (const [k, v] of Object.entries(extra)) {
      u.searchParams.set(k, v);
    }
    return u.toString();
  } catch {
    return url;
  }
}

export default function SignInPage() {
  return (
    <Suspense fallback={<main className="min-h-screen" />}>
      <SignInInner />
    </Suspense>
  );
}
