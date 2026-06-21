"""Thread synthesis — the agent reads the graph + names what's in motion.

A Thread is "something currently happening in your life that needs your
attention." The OpenLoop extractor on the Rust side scores excerpts by
liveness (how alive the conversation/decision still is). This module
takes the highest-liveness loops, asks the user's configured frontier
agent to name + categorize + cluster them, and writes the result to
graph/synth/threads.jsonl.

The first-boot screen pulls from threads.jsonl. That's the moment.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pmc.agent.prompts import base_system_prompt
from pmc.agent.providers.base import Message, ProviderConfig, ProviderError
from pmc.agent.providers.registry import get_provider
from pmc.storage.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Output schema — what /right-now reads
# ---------------------------------------------------------------------------


@dataclass
class ThreadEvidence:
    """Citation tying a Thread back to a source the user can verify."""
    source: str           # "iMessage" | "Mail" | "Slack · Reality Labs" | "Calendar" | ...
    excerpt: str          # the raw line from the graph that justifies the Thread


@dataclass
class Thread:
    """A named thing in motion that the agent thinks needs attention."""
    id: str
    headline: str                          # ≤80 chars, second-person, "Reply to Sam about the offsite"
    body: str                              # 1-2 sentences, more context
    kind: str                              # "reply" | "decision" | "follow_up" | "draft" | "appointment" | "research"
    urgency: str                           # "now" | "this_week" | "soon" | "someday"
    liveness: float                        # 0-1, copied from underlying open_loop
    related_loop_ids: list[str] = field(default_factory=list)
    related_person_ids: list[str] = field(default_factory=list)
    related_theme_labels: list[str] = field(default_factory=list)
    evidence: list[ThreadEvidence] = field(default_factory=list)
    created_at: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _synth_dir(storage_root: Path | str, user_id: str) -> Path:
    p = Path(storage_root) / "users" / user_id / "graph" / "synth"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _threads_path(storage_root: Path | str, user_id: str) -> Path:
    return _synth_dir(storage_root, user_id) / "threads.jsonl"


# ---------------------------------------------------------------------------
# Top-of-graph slice we send to the agent
# ---------------------------------------------------------------------------


def _select_live_loops(
    graph_store: GraphStore,
    user_id: str,
    *,
    limit: int = 30,
    min_liveness: float = 0.2,
) -> list[dict[str, Any]]:
    """Pull the most-alive open_loops from the graph. The Rust scoring
    decays liveness with age and boosts on recent activity; we trust
    that and just take the top N above a floor."""
    loops = list(graph_store.iter_entities(user_id, "open_loop"))
    loops = [l for l in loops if (l.get("liveness") or 0) >= min_liveness]
    loops.sort(key=lambda l: l.get("liveness") or 0, reverse=True)
    out = []
    for l in loops[:limit]:
        excerpt = (l.get("excerpt") or "").strip()
        if not excerpt:
            continue
        # Keep excerpts short to fit context budget
        if len(excerpt) > 400:
            excerpt = excerpt[:400] + "…"
        out.append({
            "id": l.get("id"),
            "kind": l.get("kind") or "undecided",
            "excerpt": excerpt,
            "last_touched": l.get("last_touched"),
            "liveness": l.get("liveness") or 0,
        })
    return out


def _select_people_index(
    graph_store: GraphStore, user_id: str, *, limit: int = 80,
) -> list[dict[str, Any]]:
    """A small directory of named people the agent can reference by name
    when naming threads (so it says 'Reply to Sam' not 'Reply to person_47')."""
    from pmc.storage.graph_store import _is_quality_person
    out = []
    for p in graph_store.iter_entities(user_id, "person"):
        if not _is_quality_person(p):
            continue
        name = (p.get("display_name") or "").strip()
        if not name:
            # Fall back to first non-phone alias
            aliases = [a for a in (p.get("aliases") or []) if a and "@" in a]
            if aliases:
                name = aliases[0]
            else:
                continue
        out.append({"id": p.get("id"), "name": name})
        if len(out) >= limit:
            break
    return out


def _select_themes(
    graph_store: GraphStore, user_id: str, *, limit: int = 30,
) -> list[dict[str, Any]]:
    """Top themes by mentions so the agent can tie threads to active themes."""
    themes = list(graph_store.iter_entities(user_id, "theme"))
    themes.sort(key=lambda t: t.get("mentions_180d") or 0, reverse=True)
    return [
        {"id": t.get("id"), "label": t.get("label"), "mentions": t.get("mentions_180d")}
        for t in themes[:limit]
        if t.get("label")
    ]


# ---------------------------------------------------------------------------
# Prompt for the synthesis pass
# ---------------------------------------------------------------------------


_THREADS_SYSTEM = """\
TASK: Synthesize "Threads" from the user's personal graph.

A Thread is a specific, in-motion piece of the user's life that
deserves attention right now. Examples:
  - A reply they owe to a specific person
  - A decision they've left undecided for too long
  - A draft they started but didn't finish
  - An appointment / event approaching
  - A recurring question they keep asking but haven't answered

You'll be given:
  1. A directory of named people the user knows (id → name)
  2. The user's most-active themes (label + mention count)
  3. The most-alive open loops in the graph, each with an excerpt
     from the originating source and a liveness score

Your job: produce 5-12 Threads. Each Thread groups one or more
open loops + names them + categorizes them.

RULES:
  - Headlines are second-person, ≤80 chars, plain, no exclamation,
    e.g. "Reply to Cliff about the Ford Mustang transport"
  - Cite at least one open_loop id per Thread (related_loop_ids)
  - Cite people by id when the excerpt mentions someone in the directory
  - urgency = "now" | "this_week" | "soon" | "someday" — based on
    liveness, recency, and any time-bound language in the excerpt
  - kind = "reply" | "decision" | "follow_up" | "draft" | "appointment" | "research"
  - Don't make things up. If the excerpt is ambiguous, name the
    Thread literally from the excerpt.
  - One evidence entry per Thread minimum, quoting the source

OUTPUT: JSON only, this exact shape:
{
  "threads": [
    {
      "headline": "…",
      "body": "…",
      "kind": "reply",
      "urgency": "this_week",
      "related_loop_ids": ["..."],
      "related_person_ids": ["..."],
      "related_theme_labels": ["..."],
      "evidence": [
        {"source": "open_loop", "excerpt": "…"}
      ]
    }
  ]
}
No prose, no markdown fences, no commentary.
"""


def _compose_threads_prompt(user_email: str) -> str:
    return base_system_prompt(user_email) + "\n\n" + _THREADS_SYSTEM


def _build_user_context(
    loops: list[dict[str, Any]],
    people: list[dict[str, Any]],
    themes: list[dict[str, Any]],
    voice_memos: list[dict[str, str]] | None = None,
) -> str:
    """The user-role message the agent gets back. Compact JSON so the
    agent can scan it cheaply."""
    parts = [
        "Here is the user's structured graph for this synthesis pass.",
        "",
        "## People directory (id → name)",
        json.dumps(people, indent=2),
        "",
        "## Active themes (label + 180d mention count)",
        json.dumps(themes, indent=2),
        "",
        "## Live open loops (highest-liveness first)",
        json.dumps(loops, indent=2),
    ]
    if voice_memos:
        parts.extend([
            "",
            "## Recent voice memos (user speaking to themselves — strongest signal of internal state)",
            json.dumps(voice_memos, indent=2),
        ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _select_recent_voice_memos(
    storage_root: Path | str, user_id: str, *, limit: int = 6,
) -> list[dict[str, str]]:
    """Pull recent voice memo transcripts as agent context. Memos are
    the most candid signal in the graph — the user talking to themselves.
    Limit to the most recent so we don't blow context budgets."""
    from pmc.synthesis.transcripts import load_transcripts
    items = load_transcripts(storage_root, user_id)
    # transcripts manifest order ≈ extractor order; take the tail (most
    # recently written) and reverse for newest-first.
    selected = list(reversed(items))[:limit]
    return [
        {
            "audio": (Path(t.audio_path).name if t.audio_path else "?"),
            "excerpt": t.text_excerpt or "",
        }
        for t in selected
        if t.text_excerpt
    ]


async def build_threads(
    *,
    graph_store: GraphStore,
    storage_root: Path | str,
    user_id: str,
    user_email: str,
    provider_config: dict[str, str],  # {"provider", "model", "api_key" (plaintext)}
    max_tokens: int = 4000,
) -> list[Thread]:
    """Run one synthesis pass. Returns the Threads it wrote. The result
    is also persisted to threads.jsonl.

    Raises ProviderError if the agent call fails."""
    loops = _select_live_loops(graph_store, user_id)
    people = _select_people_index(graph_store, user_id)
    themes = _select_themes(graph_store, user_id)
    voice_memos = _select_recent_voice_memos(storage_root, user_id)

    if not loops and not voice_memos:
        # No alive loops or voice memos → nothing for the agent to name.
        _write_threads(storage_root, user_id, [])
        return []

    system_prompt = _compose_threads_prompt(user_email)
    user_msg = _build_user_context(loops, people, themes, voice_memos)

    provider = get_provider(provider_config["provider"])
    if provider is None:
        raise ProviderError(
            f"unknown provider {provider_config['provider']!r}",
            kind="model",
        )

    cfg = ProviderConfig(
        provider=provider_config["provider"],
        model=provider_config["model"],
        api_key=provider_config["api_key"],
    )
    resp = await provider.chat(
        [Message(role="user", content=user_msg)],
        config=cfg,
        max_tokens=max_tokens,
        system=system_prompt,
    )

    threads = _parse_agent_threads(resp.text, loops)
    _write_threads(storage_root, user_id, threads)
    return threads


def load_threads(storage_root: Path | str, user_id: str) -> list[Thread]:
    """Read the last-written threads.jsonl. Empty list if nothing
    persisted yet."""
    p = _threads_path(storage_root, user_id)
    if not p.is_file():
        return []
    out: list[Thread] = []
    try:
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                evidence = [
                    ThreadEvidence(**e) for e in d.get("evidence", []) if isinstance(e, dict)
                ]
                d["evidence"] = evidence
                out.append(Thread(**d))
            except Exception:
                continue
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_agent_threads(
    raw_text: str,
    live_loops: list[dict[str, Any]],
) -> list[Thread]:
    """Parse the agent's JSON output into Thread objects.

    Tolerates: markdown code fences, leading prose ("Here are the
    threads:"), trailing prose, and ``` blocks. If parsing fails we
    return an empty list rather than raising — the caller treats
    "agent didn't produce valid threads" as a soft degrade.
    """
    text = raw_text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # Find the first { ... matching brace } block
    start = text.find("{")
    if start == -1:
        return []
    # Scan forward to find the matching close
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = start + i + 1
                break
    if end == -1:
        return []
    try:
        parsed = json.loads(text[start:end])
    except Exception:
        return []
    items = parsed.get("threads") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []

    now = datetime.utcnow().isoformat() + "Z"
    valid_loop_ids = {l.get("id") for l in live_loops if l.get("id")}
    out: list[Thread] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        headline = (item.get("headline") or "").strip()
        if not headline:
            continue
        related_loops = [
            lid for lid in (item.get("related_loop_ids") or [])
            if isinstance(lid, str) and lid in valid_loop_ids
        ]
        evidence_list = []
        for e in (item.get("evidence") or []):
            if isinstance(e, dict):
                src = (e.get("source") or "").strip() or "graph"
                ex = (e.get("excerpt") or "").strip()
                if ex:
                    evidence_list.append(ThreadEvidence(source=src, excerpt=ex))
        # Synthesize a stable id from the headline so re-runs deduplicate
        import hashlib
        thread_id = hashlib.sha1(headline.encode("utf-8")).hexdigest()[:16]
        out.append(Thread(
            id=thread_id,
            headline=headline[:140],
            body=(item.get("body") or "").strip()[:800],
            kind=(item.get("kind") or "follow_up").strip()[:32],
            urgency=(item.get("urgency") or "soon").strip()[:16],
            liveness=_avg_liveness(related_loops, live_loops),
            related_loop_ids=related_loops,
            related_person_ids=[
                pid for pid in (item.get("related_person_ids") or [])
                if isinstance(pid, str)
            ][:8],
            related_theme_labels=[
                t for t in (item.get("related_theme_labels") or [])
                if isinstance(t, str)
            ][:8],
            evidence=evidence_list[:4],
            created_at=now,
        ))
    # Sort by urgency then liveness so the user-facing list reads top-down
    URGENCY_RANK = {"now": 0, "this_week": 1, "soon": 2, "someday": 3}
    out.sort(key=lambda t: (URGENCY_RANK.get(t.urgency, 99), -t.liveness))
    return out


def _avg_liveness(related_loop_ids: list[str], live_loops: list[dict[str, Any]]) -> float:
    if not related_loop_ids:
        return 0.0
    by_id = {l.get("id"): float(l.get("liveness") or 0) for l in live_loops}
    scores = [by_id[lid] for lid in related_loop_ids if lid in by_id]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _write_threads(storage_root: Path | str, user_id: str, threads: list[Thread]) -> None:
    p = _threads_path(storage_root, user_id)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        for t in threads:
            f.write(json.dumps(t.to_json(), default=str))
            f.write("\n")
    import os
    os.replace(tmp, p)
