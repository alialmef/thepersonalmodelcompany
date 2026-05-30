"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Settings as SettingsIcon,
  X,
  Cpu,
  Database,
  LogOut,
  ChevronRight,
} from "lucide-react";
import { BrandMark } from "@/components/shared/brand-mark";
import { useUser, clearUserCache } from "@/hooks/use-user";

/**
 * Settings drawer.
 *
 * Phase 1.1 cleanup: training-related rows ("Train another", "Export
 * bundle") removed from the surface. The underlying retrain + export
 * endpoints still exist in the backend but aren't reachable from the
 * active flow now that the product has moved off voice fine-tuning.
 *
 * Linear-quiet aesthetic: thin separators, system greys, no nav chrome.
 */
export function SettingsDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  const { user } = useUser();
  const [busy, setBusy] = useState<null | "signout">(null);
  const [error] = useState<string | null>(null);

  // Esc closes
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  async function signOut() {
    setBusy("signout");
    try {
      await fetch("/api/auth/signout", { method: "POST" });
      clearUserCache();
      router.push("/sign-in");
    } catch {
      setBusy(null);
    }
  }

  return (
    <>
      {/* Scrim — fades behind the panel */}
      <div
        aria-hidden="true"
        onClick={onClose}
        className={`fixed inset-0 z-40 bg-neutral-900/15 transition-opacity duration-200 ${
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        }`}
      />

      {/* Panel */}
      <aside
        aria-hidden={!open}
        className={`fixed right-0 top-0 z-50 flex h-full w-[360px] flex-col bg-white shadow-[0_8px_40px_rgba(0,0,0,0.08)] transition-transform duration-300 ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
        style={{
          transitionTimingFunction: "var(--ease-emphasized, cubic-bezier(0.16,1,0.3,1))",
        }}
      >
        <header className="flex items-center justify-between border-b-[0.5px] border-neutral-900/8 px-6 py-5">
          <div className="flex items-center gap-2.5">
            <BrandMark size={20} />
            <span className="text-[13px] font-medium tracking-[-0.005em] text-neutral-900">
              Settings
            </span>
          </div>
          <button
            onClick={onClose}
            className="rounded-full p-1.5 text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-900"
            aria-label="Close"
          >
            <X className="size-4" strokeWidth={1.75} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-6">
          {user && (
            <div className="mb-7">
              <p className="text-[10px] uppercase tracking-[0.08em] text-neutral-500 mb-2">
                Account
              </p>
              <p className="text-[13px] text-neutral-900 truncate">{user.email}</p>
            </div>
          )}

          <div className="mb-6">
            <p className="text-[10px] uppercase tracking-[0.08em] text-neutral-500 mb-3">
              Agent
            </p>
            <div className="flex flex-col gap-px overflow-hidden rounded-[10px] border-[0.5px] border-neutral-200 bg-neutral-200">
              <Row
                icon={<Cpu className="size-4" strokeWidth={1.5} />}
                label="Configure agent"
                detail="Pick the frontier model and key it uses."
                onClick={() => {
                  onClose();
                  router.push("/settings/agent");
                }}
              />
            </div>
          </div>

          <div className="mb-6">
            <p className="text-[10px] uppercase tracking-[0.08em] text-neutral-500 mb-3">
              Your data
            </p>
            <div className="flex flex-col gap-px overflow-hidden rounded-[10px] border-[0.5px] border-neutral-200 bg-neutral-200">
              <Row
                icon={<Database className="size-4" strokeWidth={1.5} />}
                label="Manage sources"
                detail="Add more data, or remove what's connected."
                onClick={() => {
                  onClose();
                  router.push("/connect");
                }}
              />
            </div>
          </div>

          {error && (
            <p className="mb-4 rounded-[8px] border-[0.5px] border-red-500/30 bg-red-50 px-3 py-2 text-[12px] text-red-700">
              {error}
            </p>
          )}
        </div>

        <footer className="border-t-[0.5px] border-neutral-900/8 px-6 py-5">
          <button
            onClick={signOut}
            disabled={busy === "signout"}
            className="flex w-full items-center justify-center gap-2 rounded-full bg-neutral-100 px-4 py-2.5 text-[13px] font-medium text-neutral-900 transition-colors hover:bg-neutral-200 disabled:opacity-60"
          >
            <LogOut className="size-3.5" strokeWidth={1.75} />
            {busy === "signout" ? "Signing out…" : "Sign out"}
          </button>
        </footer>
      </aside>
    </>
  );
}

function Row({
  icon,
  label,
  detail,
  onClick,
  busy,
}: {
  icon: React.ReactNode;
  label: string;
  detail: string;
  onClick: () => void;
  busy?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={busy}
      className="group flex items-center gap-3.5 bg-white px-4 py-3.5 text-left transition-colors hover:bg-neutral-50 disabled:opacity-60"
    >
      <div className="flex size-7 items-center justify-center rounded-md bg-neutral-100 text-neutral-900">
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium leading-tight tracking-[-0.005em] text-neutral-900">
          {busy ? "Working…" : label}
        </div>
        <div className="mt-0.5 text-[11px] text-neutral-500">{detail}</div>
      </div>
      <ChevronRight
        className="size-3.5 text-neutral-300 group-hover:text-neutral-500 transition-colors"
        strokeWidth={1.75}
      />
    </button>
  );
}

/**
 * Tiny header-mount button. Use in any screen header where the user should
 * be able to reach settings. Renders the same gear that chat-screen.tsx
 * already has, but wired to the drawer this module owns.
 */
export function SettingsGear({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="text-neutral-500 transition-colors hover:text-neutral-900"
      aria-label="Settings"
    >
      <SettingsIcon className="size-[18px]" strokeWidth={1.5} />
    </button>
  );
}
