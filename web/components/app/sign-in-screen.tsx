"use client";

import { useEffect, useRef, useState } from "react";

import { BrandMark } from "@/components/shared/brand-mark";

/**
 * /sign-in — between /welcome and /connect.
 *
 * Two states:
 *   1. EMAIL  — user types email, taps Continue → server emails code
 *   2. CODE   — user types the 6-char code → server returns session
 *
 * Aesthetic matches the rest: typed prose, centered, generous
 * whitespace, no banners. The only widgets are the input field and
 * a single subtle Continue affordance.
 */

export type SignInStep = "email" | "code";

export interface SignInScreenProps {
  step: SignInStep;
  email: string;
  code: string;
  busy: boolean;
  error?: string;
  onEmailChange: (email: string) => void;
  onCodeChange: (code: string) => void;
  onSubmitEmail: () => void;
  onSubmitCode: () => void;
  onBackToEmail: () => void;
}

function EmailStep({
  email,
  busy,
  error,
  onChange,
  onSubmit,
}: {
  email: string;
  busy: boolean;
  error?: string;
  onChange: (e: string) => void;
  onSubmit: () => void;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const t = setTimeout(() => ref.current?.focus(), 600);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="space-y-10">
      <div className="space-y-2 text-foreground/85">
        <div className="text-xl font-semibold text-foreground">Tell me your email.</div>
        <div className="text-base text-foreground/55">
          So we can find you across devices.
        </div>
      </div>
      <input
        ref={ref}
        type="email"
        inputMode="email"
        autoComplete="email"
        autoCapitalize="off"
        spellCheck={false}
        value={email}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          }
        }}
        disabled={busy}
        placeholder="you@somewhere.com"
        className="w-full max-w-md border-0 border-b border-foreground/15 bg-transparent pb-2 text-lg text-foreground outline-none placeholder:text-foreground/25 focus:border-foreground/50"
      />
      {error && <div className="text-sm text-red-500">{error}</div>}
      <button
        type="button"
        onClick={onSubmit}
        disabled={busy || !email.trim()}
        className={`text-base transition-opacity duration-300 ${
          busy || !email.trim()
            ? "cursor-default text-foreground/25"
            : "cursor-pointer text-foreground/80 hover:text-foreground"
        }`}
      >
        {busy ? "Sending…" : "Continue"}
      </button>
    </div>
  );
}

function CodeStep({
  email,
  code,
  busy,
  error,
  onChange,
  onSubmit,
  onBack,
}: {
  email: string;
  code: string;
  busy: boolean;
  error?: string;
  onChange: (c: string) => void;
  onSubmit: () => void;
  onBack: () => void;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const t = setTimeout(() => ref.current?.focus(), 400);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="space-y-10">
      <div className="space-y-2 text-foreground/85">
        <div className="text-xl font-semibold text-foreground">Check your email.</div>
        <div className="text-base text-foreground/55">
          We sent a code to <span className="text-foreground/80">{email}</span>.
        </div>
      </div>
      <input
        ref={ref}
        type="text"
        inputMode="text"
        autoComplete="one-time-code"
        autoCapitalize="characters"
        spellCheck={false}
        value={code}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          }
        }}
        disabled={busy}
        placeholder="ABC-DEF"
        maxLength={10}
        className="w-full max-w-xs border-0 border-b border-foreground/15 bg-transparent pb-2 font-mono text-2xl tracking-[0.2em] text-foreground outline-none placeholder:text-foreground/20 focus:border-foreground/50"
      />
      {error && <div className="text-sm text-red-500">{error}</div>}
      <div className="flex flex-wrap items-center gap-8">
        <button
          type="button"
          onClick={onSubmit}
          disabled={busy || !code.trim()}
          className={`text-base transition-opacity duration-300 ${
            busy || !code.trim()
              ? "cursor-default text-foreground/25"
              : "cursor-pointer text-foreground/80 hover:text-foreground"
          }`}
        >
          {busy ? "Signing in…" : "Continue"}
        </button>
        <button
          type="button"
          onClick={onBack}
          disabled={busy}
          className="text-base text-foreground/45 hover:text-foreground/65 disabled:cursor-default disabled:text-foreground/25"
        >
          Different email
        </button>
      </div>
    </div>
  );
}

export default function SignInScreen(props: SignInScreenProps) {
  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <div className="mt-16">
          {props.step === "email" ? (
            <EmailStep
              email={props.email}
              busy={props.busy}
              error={props.error}
              onChange={props.onEmailChange}
              onSubmit={props.onSubmitEmail}
            />
          ) : (
            <CodeStep
              email={props.email}
              code={props.code}
              busy={props.busy}
              error={props.error}
              onChange={props.onCodeChange}
              onSubmit={props.onSubmitCode}
              onBack={props.onBackToEmail}
            />
          )}
        </div>
      </div>
    </main>
  );
}
