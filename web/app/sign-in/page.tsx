"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";
import SignInScreen, {
  type SignInStep,
} from "@/components/app/sign-in-screen";
import {
  claimAnonymousIfAny,
  exchangeCode,
  requestEmailCode,
  storeSession,
} from "@/lib/api/auth";
import { isTauri } from "@/lib/runtime";

/**
 * Sign-in. Two renderings, one route:
 *
 *   Web (marketing site)     →  email + magic-link sent by Next.js
 *                                /api/auth/magic-link route, server-
 *                                rendered cookie session, drizzle-
 *                                backed users table.
 *   Tauri (Mac app)          →  email + 6-char code typed back, talks
 *                                directly to FastAPI's /v1/auth/*
 *                                endpoints, session token in
 *                                localStorage. After sign-in, the
 *                                anonymous pmcUserId already in
 *                                localStorage gets claimed onto the
 *                                new account so accumulated data
 *                                survives.
 *
 * Same route, branched by the runtime detection so the Tauri webview
 * doesn't try to talk to /api/auth/* (which gets stripped from the
 * static export).
 */

interface MagicLinkResponse {
  ok?: boolean;
  dev?: { link: string };
}

// ---------------------------------------------------------------------------
// Tauri (Mac app) path — code-paste flow against FastAPI
// ---------------------------------------------------------------------------

function TauriSignIn() {
  const router = useRouter();
  const [step, setStep] = useState<SignInStep>("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | undefined>();

  async function handleSubmitEmail() {
    setError(undefined);
    const e = email.trim().toLowerCase();
    if (!e.includes("@")) {
      setError("That doesn't look like an email.");
      return;
    }
    setBusy(true);
    try {
      await requestEmailCode(e);
      setEmail(e);
      setStep("code");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't send the code.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmitCode() {
    setError(undefined);
    const trimmed = code.trim();
    if (!trimmed) {
      setError("Type the code we just sent.");
      return;
    }
    setBusy(true);
    try {
      const result = await exchangeCode(email, trimmed);
      storeSession(result.session_token, result.account);
      // Bind any pre-existing anonymous pmcUserId to this account so
      // accumulated data (raw ingest, recall.db, etc.) survives.
      await claimAnonymousIfAny(result.session_token);
      router.push("/connect");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Couldn't sign in.";
      const friendlier = msg.includes("expired") || msg.includes("match")
        ? "That code didn't match or expired. Try again."
        : msg;
      setError(friendlier);
    } finally {
      setBusy(false);
    }
  }

  return (
    <SignInScreen
      step={step}
      email={email}
      code={code}
      busy={busy}
      error={error}
      onEmailChange={setEmail}
      onCodeChange={setCode}
      onSubmitEmail={handleSubmitEmail}
      onSubmitCode={handleSubmitCode}
      onBackToEmail={() => {
        setStep("email");
        setCode("");
        setError(undefined);
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Web (marketing site) path — magic-link via Next.js /api/auth/magic-link
// ---------------------------------------------------------------------------

function WebSignIn() {
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
                We sent a link to <span className="text-foreground">{email}</span>.<br />
                Open it on this device to sign in.
              </p>
              {devLink && (
                <p className="mt-6 text-[12px] text-muted">
                  Dev mode (no email service configured):{" "}
                  <a href={devLink} className="underline text-foreground">open the link here</a>.
                </p>
              )}
            </div>
          )}
          {error && (
            <p className="mt-6 text-[14px] text-foreground text-center">{error}</p>
          )}
        </div>
      </section>
      <footer className="px-6 py-8 text-[13px] text-muted text-center">
        <Link href="/" className="hover:text-foreground transition-colors">← Back</Link>
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

// ---------------------------------------------------------------------------
// Entry — branch on runtime
// ---------------------------------------------------------------------------

function SignInInner() {
  // isTauri() depends on window; render a stable placeholder on the
  // first paint to avoid hydration mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <main className="min-h-screen w-full bg-background" />;
  return isTauri() ? <TauriSignIn /> : <WebSignIn />;
}

export default function SignInPage() {
  return (
    <Suspense fallback={<main className="min-h-screen" />}>
      <SignInInner />
    </Suspense>
  );
}
