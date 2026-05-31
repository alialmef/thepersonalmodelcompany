"use client";

import { useEffect, useState } from "react";

import { BrandMark } from "@/components/shared/brand-mark";
import { useUser } from "@/hooks/use-user";
import {
  addRedaction,
  eraseEverything,
  forgetItem,
  getOverview,
  pauseSource,
  removeRedaction,
  resumeSource,
  search,
  type KnowledgeOverview,
  type Redaction,
  type SearchResult,
} from "@/lib/api/knowledge";

/**
 * /knowledge-update — the redact + manage surface.
 *
 * Auto-opt-in means PMC reads everything from the jump. This is the
 * contract on the other side: queryable, redactable, pauseable. Four
 * sections, in trust-order:
 *
 *   1. Pause: per-source on/off + last item count
 *   2. Private: mark a Person / Topic / Date range — agent never reads
 *   3. Search: substring scan over ingested raw items, with Forget
 *      buttons (V1: backend returns 501; UI shows a hint)
 *   4. Erase: nuclear, two-step confirm
 *
 * The aesthetic matches /sign-in, /reading, /right-now: typed prose,
 * generous whitespace, no banners. This screen is *not* a settings
 * panel; it's a calm declarative surface.
 */

export default function KnowledgeUpdatePage() {
  const { user } = useUser();
  const userId = user?.pmcUserId ?? "";
  const [overview, setOverview] = useState<KnowledgeOverview | null>(null);
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    if (!userId) return;
    refresh().catch(() => {});
  }, [userId]);

  async function refresh() {
    try {
      const o = await getOverview(userId);
      setOverview(o);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load.");
    }
  }

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <header className="mt-12 mb-16 space-y-3 text-foreground/85">
          <div className="text-xl font-semibold text-foreground">
            What I know about you.
          </div>
          <div className="text-base text-foreground/55">
            Pause any source. Mark a person, a topic, or a date range
            private. Search what&apos;s been ingested. Or erase it all.
          </div>
        </header>

        {error && (
          <div className="mb-8 text-sm text-red-500">{error}</div>
        )}

        <Sources overview={overview} userId={userId} onChange={refresh} />
        <PrivateList overview={overview} userId={userId} onChange={refresh} />
        <SearchAndForget userId={userId} />
        <Erase userId={userId} />
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Sources (pause / resume)
// ---------------------------------------------------------------------------

function Sources({
  overview,
  userId,
  onChange,
}: {
  overview: KnowledgeOverview | null;
  userId: string;
  onChange: () => void | Promise<void>;
}) {
  const sources = overview?.sources ?? [];
  if (sources.length === 0) {
    return (
      <Section title="Sources">
        <div className="text-sm text-foreground/40">
          {overview ? "Nothing ingested yet." : "Loading…"}
        </div>
      </Section>
    );
  }
  return (
    <Section title="Sources">
      <div className="space-y-1">
        {sources
          .slice()
          .sort((a, b) => b.item_count - a.item_count)
          .map((s) => (
            <div
              key={s.source_id}
              className="flex items-baseline justify-between gap-4 py-2"
            >
              <div className="min-w-0 flex-1">
                <div className="text-[15px] text-foreground/85">
                  {s.kind}
                  {s.paused && (
                    <span className="ml-2 text-xs uppercase tracking-wider text-foreground/40">
                      paused
                    </span>
                  )}
                </div>
                <div className="text-xs text-foreground/40 font-mono truncate">
                  {s.source_id} · {s.item_count.toLocaleString()} items
                </div>
              </div>
              <button
                type="button"
                onClick={async () => {
                  if (s.paused) await resumeSource(userId, s.source_id);
                  else await pauseSource(userId, s.source_id);
                  await onChange();
                }}
                className="text-sm text-foreground/55 hover:text-foreground/85"
              >
                {s.paused ? "Resume" : "Pause"}
              </button>
            </div>
          ))}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Private list
// ---------------------------------------------------------------------------

function PrivateList({
  overview,
  userId,
  onChange,
}: {
  overview: KnowledgeOverview | null;
  userId: string;
  onChange: () => void | Promise<void>;
}) {
  const [kind, setKind] = useState<Redaction["kind"]>("person");
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);

  const items = overview?.redactions ?? [];

  async function add() {
    const v = value.trim();
    if (!v) return;
    setBusy(true);
    try {
      await addRedaction(userId, { kind, value: v });
      setValue("");
      await onChange();
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section title="Private">
      <div className="space-y-1 mb-8">
        {items.length === 0 ? (
          <div className="text-sm text-foreground/40">Nothing marked private yet.</div>
        ) : (
          items.map((r) => (
            <div
              key={r.id}
              className="flex items-baseline justify-between gap-4 py-2"
            >
              <div className="min-w-0 flex-1">
                <div className="text-[15px] text-foreground/85">
                  <span className="text-xs uppercase tracking-wider text-foreground/40 mr-2">
                    {r.kind === "date_range" ? "dates" : r.kind}
                  </span>
                  <span className="font-mono">{r.value}</span>
                </div>
                <div className="text-xs text-foreground/40">
                  Added {new Date(r.added_at).toLocaleString()}
                </div>
              </div>
              <button
                type="button"
                onClick={async () => {
                  await removeRedaction(userId, r.id);
                  await onChange();
                }}
                className="text-sm text-foreground/55 hover:text-foreground/85"
              >
                Remove
              </button>
            </div>
          ))
        )}
      </div>

      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {(["person", "topic", "date_range"] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setKind(k)}
              className={`rounded-full px-3 py-1 text-xs transition-colors ${
                kind === k
                  ? "bg-foreground text-background"
                  : "bg-foreground/5 text-foreground/70 hover:bg-foreground/10"
              }`}
            >
              {k === "date_range" ? "Date range" : k.charAt(0).toUpperCase() + k.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex items-baseline gap-3">
          <input
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
            placeholder={
              kind === "person"
                ? "name, email, or phone"
                : kind === "topic"
                ? "topic substring — e.g. medical"
                : "2026-03-20/2026-03-27"
            }
            disabled={busy}
            className="flex-1 max-w-md border-0 border-b border-foreground/15 bg-transparent pb-2 text-base text-foreground outline-none placeholder:text-foreground/25 focus:border-foreground/50"
          />
          <button
            type="button"
            onClick={add}
            disabled={busy || !value.trim()}
            className={`text-sm transition-opacity ${
              busy || !value.trim()
                ? "cursor-default text-foreground/25"
                : "cursor-pointer text-foreground/80 hover:text-foreground"
            }`}
          >
            {busy ? "Adding…" : "Mark private"}
          </button>
        </div>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Search and forget
// ---------------------------------------------------------------------------

function SearchAndForget({ userId }: { userId: string }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const [forgetMessage, setForgetMessage] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  async function run() {
    const query = q.trim();
    if (!query) return;
    setBusy(true);
    setForgetMessage(null);
    try {
      const r = await search(userId, query);
      setResults(r.results);
      setTruncated(r.truncated);
      setSearched(true);
    } finally {
      setBusy(false);
    }
  }

  async function tryForget(id: string | undefined) {
    if (!id) return;
    const ok = await forgetItem(userId, id);
    if (!ok) {
      setForgetMessage(
        "Per-item forget lands with the next release. For now: pause the source, or mark the person/topic private.",
      );
    } else {
      setResults((rs) => rs.filter((r) => r.id !== id));
    }
  }

  return (
    <Section title="Search and forget">
      <div className="flex items-baseline gap-3">
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              run();
            }
          }}
          placeholder="name, phrase, anything"
          disabled={busy}
          className="flex-1 max-w-md border-0 border-b border-foreground/15 bg-transparent pb-2 text-base text-foreground outline-none placeholder:text-foreground/25 focus:border-foreground/50"
        />
        <button
          type="button"
          onClick={run}
          disabled={busy || !q.trim()}
          className={`text-sm transition-opacity ${
            busy || !q.trim()
              ? "cursor-default text-foreground/25"
              : "cursor-pointer text-foreground/80 hover:text-foreground"
          }`}
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </div>

      {forgetMessage && (
        <div className="mt-6 text-sm text-foreground/55">{forgetMessage}</div>
      )}

      <div className="mt-6 space-y-3">
        {searched && results.length === 0 && !busy && (
          <div className="text-sm text-foreground/40">
            No matches.
          </div>
        )}
        {results.map((r, i) => (
          <div
            key={r.id ?? i}
            className="flex items-baseline justify-between gap-4 py-2 border-t border-foreground/10"
          >
            <div className="min-w-0 flex-1">
              <div className="text-xs uppercase tracking-wider text-foreground/40">
                {r.kind ?? "item"}
                {r.timestamp && <span className="ml-2 text-foreground/30">{r.timestamp}</span>}
              </div>
              <div className="text-sm text-foreground/75 font-mono truncate">
                {r.preview}
              </div>
            </div>
            <button
              type="button"
              onClick={() => tryForget(r.id)}
              className="text-sm text-foreground/55 hover:text-foreground/85"
            >
              Forget
            </button>
          </div>
        ))}
        {truncated && (
          <div className="text-xs text-foreground/40">
            More matches exist — narrow your query.
          </div>
        )}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Erase
// ---------------------------------------------------------------------------

function Erase({ userId }: { userId: string }) {
  const [stage, setStage] = useState<"idle" | "confirm" | "done">("idle");
  const [busy, setBusy] = useState(false);
  return (
    <Section title="Erase everything">
      <div className="text-sm text-foreground/55 mb-6">
        Wipes every raw item, every memory, every adapter trained on
        your data. Cannot be undone.
      </div>
      {stage === "idle" && (
        <button
          type="button"
          onClick={() => setStage("confirm")}
          className="text-base text-red-500/80 hover:text-red-500"
        >
          Erase everything PMC knows about me
        </button>
      )}
      {stage === "confirm" && (
        <div className="space-y-3">
          <div className="text-base text-foreground/85">Are you sure?</div>
          <div className="flex items-baseline gap-6">
            <button
              type="button"
              onClick={async () => {
                setBusy(true);
                try {
                  await eraseEverything(userId);
                  setStage("done");
                } finally {
                  setBusy(false);
                }
              }}
              disabled={busy}
              className="text-base text-red-500/80 hover:text-red-500 disabled:cursor-default disabled:text-foreground/25"
            >
              {busy ? "Erasing…" : "Yes, erase everything"}
            </button>
            <button
              type="button"
              onClick={() => setStage("idle")}
              disabled={busy}
              className="text-base text-foreground/55 hover:text-foreground/85 disabled:cursor-default disabled:text-foreground/25"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {stage === "done" && (
        <div className="text-base text-foreground/55">
          Done. Reload the app to start over.
        </div>
      )}
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-16">
      <h2 className="mb-6 text-xs uppercase tracking-[0.18em] text-foreground/40">
        {title}
      </h2>
      {children}
    </section>
  );
}
