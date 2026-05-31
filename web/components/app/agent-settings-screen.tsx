"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { BrandMark } from "@/components/shared/brand-mark";
import {
  clearConfig,
  getConfig,
  listProviders,
  setConfig,
  validateKey,
  type AgentConfig,
  type ProviderInfo,
} from "@/lib/api/agent";

/**
 * /settings/agent — the BYOM configuration screen.
 *
 * Three states:
 *   1. NOT_CONFIGURED  — never set; show provider+model+key form
 *   2. CONFIGURED      — already saved; show summary + Edit / Disconnect
 *   3. EDITING         — same form as (1), with "Cancel" back to (2)
 *
 * Typed-prose aesthetic matching /sign-in and /right-now: centered
 * column, big white space, no banners.
 */

type View = "loading" | "summary" | "editing";

export default function AgentSettingsScreen() {
  const router = useRouter();
  const [view, setView] = useState<View>("loading");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [config, setConfigState] = useState<AgentConfig | null>(null);
  const [error, setError] = useState<string | undefined>();
  // True when the user arrived here without a configured agent — i.e.
  // this is the onboarding step, not a settings visit. We track it on
  // first load so Save can route them straight into /reading instead
  // of dumping them back at /right-now.
  const [isOnboarding, setIsOnboarding] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [provs, cfg] = await Promise.all([listProviders(), getConfig()]);
        setProviders(provs);
        setConfigState(cfg);
        setIsOnboarding(!cfg.configured);
        setView(cfg.configured ? "summary" : "editing");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Couldn't load settings.");
        setView("editing");
      }
    })();
  }, []);

  async function refresh() {
    const cfg = await getConfig();
    setConfigState(cfg);
  }

  async function handleSave(args: {
    provider: string;
    model: string;
    api_key: string;
  }) {
    setError(undefined);
    try {
      await setConfig(args);
      await refresh();
      // Onboarding path: straight into the reading flow once an agent
      // is configured. Settings visit (already had a config): land on
      // the summary view.
      if (isOnboarding) {
        router.push("/reading");
      } else {
        setView("summary");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save.");
    }
  }

  async function handleDisconnect() {
    setError(undefined);
    await clearConfig();
    await refresh();
    setView("editing");
  }

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <div className="mt-16">
          {view === "loading" && <SkeletonLine />}

          {view === "summary" && config?.configured && (
            <Summary
              config={config}
              providers={providers}
              onEdit={() => setView("editing")}
              onDisconnect={handleDisconnect}
            />
          )}

          {view === "editing" && (
            <Editor
              providers={providers}
              initial={config ?? undefined}
              encryptionConfigured={config?.encryption_configured ?? true}
              onSave={handleSave}
              onCancel={
                config?.configured ? () => setView("summary") : undefined
              }
              error={error}
            />
          )}
        </div>

        <div className="mt-auto pt-16">
          <Link
            href={isOnboarding ? "/reading" : "/right-now"}
            className="text-sm text-foreground/55 hover:text-foreground/85"
          >
            Done
          </Link>
        </div>
      </div>
    </main>
  );
}

function SkeletonLine() {
  return (
    <div className="text-base text-foreground/40">Loading your settings…</div>
  );
}

function Summary({
  config,
  providers,
  onEdit,
  onDisconnect,
}: {
  config: AgentConfig;
  providers: ProviderInfo[];
  onEdit: () => void;
  onDisconnect: () => void;
}) {
  const label =
    providers.find((p) => p.id === config.provider)?.label ?? config.provider;
  return (
    <div className="space-y-10">
      <div className="space-y-2 text-foreground/85">
        <div className="text-xl font-semibold text-foreground">
          Your agent is configured.
        </div>
        <div className="text-base text-foreground/55">
          {label} · <span className="font-mono text-foreground/70">{config.model}</span>
        </div>
        {config.updated_at && (
          <div className="text-sm text-foreground/40">
            Set {new Date(config.updated_at).toLocaleString()}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-8">
        <button
          type="button"
          onClick={onEdit}
          className="text-base text-foreground/80 hover:text-foreground"
        >
          Change
        </button>
        <button
          type="button"
          onClick={onDisconnect}
          className="text-base text-foreground/45 hover:text-foreground/75"
        >
          Disconnect
        </button>
      </div>
    </div>
  );
}

function Editor({
  providers,
  initial,
  encryptionConfigured,
  onSave,
  onCancel,
  error,
}: {
  providers: ProviderInfo[];
  initial?: AgentConfig;
  encryptionConfigured: boolean;
  onSave: (args: { provider: string; model: string; api_key: string }) => Promise<void>;
  onCancel?: () => void;
  error?: string;
}) {
  const [providerId, setProviderId] = useState<string>(
    initial?.provider ?? providers[0]?.id ?? "anthropic",
  );
  const [model, setModel] = useState<string>(initial?.model ?? "");
  const [apiKey, setApiKey] = useState<string>("");
  const [busy, setBusy] = useState<"none" | "validating" | "saving">("none");
  const [probeResult, setProbeResult] = useState<"ok" | "bad" | null>(null);

  const currentProvider = providers.find((p) => p.id === providerId);

  // When provider changes, default the model to the first one in the list
  useEffect(() => {
    if (!currentProvider) return;
    if (!model || !currentProvider.default_models.includes(model)) {
      setModel(currentProvider.default_models[0] ?? "");
    }
    setProbeResult(null);
  }, [providerId]); // eslint-disable-line react-hooks/exhaustive-deps

  async function testKey() {
    if (!apiKey.trim()) return;
    setBusy("validating");
    setProbeResult(null);
    const ok = await validateKey({ provider: providerId, api_key: apiKey.trim() });
    setProbeResult(ok ? "ok" : "bad");
    setBusy("none");
  }

  async function save() {
    if (!apiKey.trim() || !model.trim()) return;
    setBusy("saving");
    try {
      await onSave({ provider: providerId, model: model.trim(), api_key: apiKey.trim() });
    } finally {
      setBusy("none");
    }
  }

  return (
    <div className="space-y-10">
      <div className="space-y-2 text-foreground/85">
        <div className="text-xl font-semibold text-foreground">
          Pick your agent.
        </div>
        <div className="text-base text-foreground/55">
          Any frontier model. Bring your own key. The key stays
          encrypted on our server and never leaves your account.
        </div>
      </div>

      {!encryptionConfigured && (
        <div className="text-sm text-amber-700">
          This deploy isn&apos;t configured for at-rest key encryption yet
          (PMC_KEY_ENCRYPTION_SECRET). Saving is disabled until the
          operator sets it.
        </div>
      )}

      {/* Provider picker — radio-style row of buttons */}
      <div className="space-y-3">
        <div className="text-sm text-foreground/55">Provider</div>
        <div className="flex flex-wrap gap-2">
          {providers.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setProviderId(p.id)}
              className={`rounded-full px-4 py-1.5 text-sm transition-colors ${
                providerId === p.id
                  ? "bg-foreground text-background"
                  : "bg-foreground/5 text-foreground/70 hover:bg-foreground/10"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Model picker — datalist allows custom + suggested */}
      <div className="space-y-3">
        <div className="text-sm text-foreground/55">Model</div>
        <input
          list={`models-${providerId}`}
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="model id"
          spellCheck={false}
          className="w-full max-w-lg border-0 border-b border-foreground/15 bg-transparent pb-2 font-mono text-base text-foreground outline-none placeholder:text-foreground/25 focus:border-foreground/50"
        />
        <datalist id={`models-${providerId}`}>
          {currentProvider?.default_models.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </div>

      {/* API key */}
      <div className="space-y-3">
        <div className="flex items-baseline justify-between gap-4">
          <div className="text-sm text-foreground/55">API key</div>
          {currentProvider?.console_url && (
            <a
              href={currentProvider.console_url}
              target="_blank"
              rel="noreferrer"
              className="text-sm text-foreground/45 hover:text-foreground/75"
            >
              Get one →
            </a>
          )}
        </div>
        <input
          type="password"
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          value={apiKey}
          onChange={(e) => {
            setApiKey(e.target.value);
            setProbeResult(null);
          }}
          placeholder={currentProvider?.key_prefix_hint ?? "sk-..."}
          className="w-full max-w-lg border-0 border-b border-foreground/15 bg-transparent pb-2 font-mono text-base text-foreground outline-none placeholder:text-foreground/25 focus:border-foreground/50"
        />
      </div>

      {error && <div className="text-sm text-red-500">{error}</div>}
      {probeResult === "ok" && (
        <div className="text-sm text-emerald-600">Key validated.</div>
      )}
      {probeResult === "bad" && (
        <div className="text-sm text-red-500">
          That key didn&apos;t work against {currentProvider?.label}.
        </div>
      )}

      <div className="flex flex-wrap items-center gap-8">
        <button
          type="button"
          onClick={save}
          disabled={
            !apiKey.trim() ||
            !model.trim() ||
            busy !== "none" ||
            !encryptionConfigured
          }
          className={`text-base transition-opacity ${
            !apiKey.trim() || !model.trim() || busy !== "none" || !encryptionConfigured
              ? "cursor-default text-foreground/25"
              : "cursor-pointer text-foreground/85 hover:text-foreground"
          }`}
        >
          {busy === "saving" ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          onClick={testKey}
          disabled={!apiKey.trim() || busy !== "none"}
          className="text-base text-foreground/55 hover:text-foreground/85 disabled:cursor-default disabled:text-foreground/25"
        >
          {busy === "validating" ? "Testing…" : "Test connection"}
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={busy !== "none"}
            className="text-base text-foreground/45 hover:text-foreground/75 disabled:cursor-default disabled:text-foreground/25"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
