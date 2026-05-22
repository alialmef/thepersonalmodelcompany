"""Preview consolidation — heuristic summaries, no LLM, no API cost.

Useful for:
  * validating the embed + retrieve pipeline end-to-end before paying
    for Claude consolidation
  * showing the user a working "knows you" demo on day one even if
    they haven't set ANTHROPIC_API_KEY yet
  * a degraded-mode fallback when offline

The summaries it produces are mechanical (first N chars / first
sentence) and miss state-change detection entirely. Once a real
Consolidator pass runs over an episode, it overwrites the preview
summary with the Claude-produced one.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from pmc.memory.recall.embed import LocalEmbedder
from pmc.memory.recall.schema import Episode
from pmc.memory.recall.store import RecallStore


def preview_consolidate(
    store: RecallStore,
    raw_text_lookup: Callable[[Episode], str],
    limit: int = 5000,
) -> int:
    """Heuristic summary + embedding for every pending episode."""
    pending = store.pending_consolidation(limit=limit)
    now = datetime.now(timezone.utc)
    summaries: list[tuple[Episode, str]] = []
    for ep in pending:
        raw = raw_text_lookup(ep) or ""
        if not raw.strip():
            ep.summary = "[no recoverable content]"
            ep.summary_model = "preview/none"
            ep.consolidation_time = now
            store.upsert_episode(ep)
            continue
        s = _heuristic_summary(ep, raw)
        ep.summary = s
        ep.summary_model = "preview/heuristic-v1"
        ep.importance = _heuristic_importance(ep, raw)
        ep.consolidation_time = now
        store.upsert_episode(ep, raw_text_for_fts=raw[:2000])
        summaries.append((ep, s))

    if not summaries:
        return 0

    # Batch-embed everything we just produced.
    texts = [s for _, s in summaries]
    vectors = LocalEmbedder.embed(texts)
    for (ep, _), vec in zip(summaries, vectors):
        store.set_embedding(ep.id, vec, model=LocalEmbedder.name())
    return len(summaries)


def _heuristic_summary(ep: Episode, raw: str) -> str:
    kind = ep.kind.value
    # iMessage: keep first 3 exchanges or 360 chars.
    if ep.raw_source == "imessage":
        lines = [l for l in raw.splitlines() if l.strip()]
        first = lines[:6]
        joined = " / ".join(first)[:360]
        date = ep.time_start.date().isoformat()
        partner = ep.participant_ids[0] if ep.participant_ids else "someone"
        return f"Conversation with {partner} on {date}: {joined}"
    # Notes: title + first paragraph.
    if ep.raw_source == "notes":
        lines = [l for l in raw.splitlines() if l.strip()]
        title = lines[0] if lines else ""
        rest = " ".join(lines[1:6])[:400]
        return f"Note '{title}'. {rest}".strip()
    # Calendar / photos: synthesize from event title + metadata.
    if ep.raw_source in ("calendar", "photos"):
        return f"{kind.replace('_', ' ').title()} — {raw[:240]}"
    if ep.raw_source == "safari":
        return f"Web session — {raw[:240]}"
    return raw[:300]


def _heuristic_importance(ep: Episode, raw: str) -> float:
    # Quick proxy: length + presence of question marks + names.
    score = 0.3
    n_words = len(raw.split())
    if n_words > 100:
        score += 0.15
    if n_words > 500:
        score += 0.10
    if "?" in raw:
        score += 0.10
    if re.search(r"\b(decide|decision|plan|need|important|tomorrow|tonight|today)\b", raw, re.I):
        score += 0.10
    if ep.kind.value == "calendar_event":
        score += 0.05
    return min(0.95, score)
