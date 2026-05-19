"use client";

import { useState } from "react";
import Link from "next/link";
import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";

/**
 * Act 1 — sign-in.
 *
 * One input, one button. Magic link sent. The mode after submit is a small
 * confirmation, not a screen change — the form stays so they remember what
 * email they used.
 */
export default function SignInPage() {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email) return;
    setState("sending");
    setError(null);
    try {
      const r = await fetch("/api/auth/magic-link", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (!r.ok) throw new Error("Could not send the link. Try again.");
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
            <p className="mt-8 text-[14px] text-muted text-center leading-relaxed">
              We sent a link to <span className="text-foreground">{email}</span>.
              <br />
              Open it on this device to sign in.
            </p>
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
