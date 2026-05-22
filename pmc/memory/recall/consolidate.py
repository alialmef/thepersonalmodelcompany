"""Frontier-model consolidation worker — with parallel LLM calls.

This is the heart of the recall layer. Raw episode bundles go in;
structured, summarized, state-aware memory comes out.

We use Anthropic's Claude Sonnet 4.6 by default because, as of mid-2026,
it is the best model on the market for:
  * dense structured-output generation (JSON with multi-field schemas)
  * extracting state changes from longitudinal text
  * holding emotional register without flattening it
  * prompt caching — critical for nightly batches where the system
    prompt is large and stable

Configurable via the `PMC_CONSOLIDATION_MODEL` env var. We deliberately
do NOT default to a small local model. A poorly-tagged episode pollutes
the graph permanently and propagates noise through retrieval. The cost
of a single Claude call is ~$0.003 per episode at the prompt sizes we
use; for a user with 30 new episodes a day that's ~$0.09 a day in
consolidation — well within any sane pricing.

For each episode we extract:
  * a 1-3 sentence summary in third person
  * topics (2-5 short labels)
  * entity_states (subject/predicate/object/temporal scope) — these
    become bi-temporal Facts
  * emotional_tone (one of ~12 stable categories)
  * importance (0-1 — relative weight for retrieval ranking)
  * open_loops created or closed by this episode
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from pmc.memory.recall.embed import LocalEmbedder
from pmc.memory.recall.schema import Episode, Fact
from pmc.memory.recall.store import RecallStore


DEFAULT_MODEL = "claude-sonnet-4-6"

CONSOLIDATION_SYSTEM_PROMPT = """You are a memory consolidator for a personal AI.

You read one episode from the user's life — a conversation thread, a
photo cluster, a calendar event, a written note, a call, a browsing
session — and produce a structured JSON record that another agent will
later retrieve when the user asks "what's going on with X" or "remind
me about that Y."

Your output is the user's own memory of this moment. Treat it that way.
Write the summary from the user's third-person perspective ("Sarah told
the user that..." not "I learned that..." and not "you discussed...").

Be specific. Names, dates, decisions, places. Avoid generic verbs like
"discussed", "talked about" — say what was actually said.

Be honest about state changes. If Sarah announces she's pregnant, that
is a state change — surface it as an entity_state. If the user agrees
to visit in October, that is a planned event — surface it as an
open_loop. If something prior is contradicted (Sarah was going to move
to Berlin, now she's staying), invalidate the prior assumption.

NEVER fabricate detail. If the episode doesn't say something, don't
infer it. Confidence reflects how clearly the episode states the fact.

You return ONLY valid JSON in this schema (no commentary, no markdown
fence):

{
  "summary": "one to three sentences, third person",
  "topics": ["short", "labels"],
  "emotional_tone": "one of: warm | tense | celebratory | grieving | practical | playful | conflicted | tender | excited | drained | reflective | neutral",
  "importance": 0.0-1.0,
  "entity_states": [
    {
      "subject": "entity name as it appears in the episode",
      "predicate": "lives_in | works_at | pregnant_with | feels_about | plans_to | knows_about | relationship_with | health | role | other",
      "object": "string value or entity name",
      "object_kind": "entity | literal",
      "confidence": 0.0-1.0,
      "valid_from": "ISO-8601 or null",
      "valid_until": "ISO-8601 or null"
    }
  ],
  "open_loops_opened": [
    {
      "kind": "unanswered_question | undecided | unsent_draft | missed_followup | planned_unscheduled",
      "summary": "one-line description",
      "related_subject": "entity name or null"
    }
  ],
  "open_loops_closed": [
    {
      "prior_loop_excerpt": "snippet of the loop that just resolved"
    }
  ]
}
"""


@dataclass
class ConsolidationResult:
    episode_id: str
    summary: str
    topics: list[str]
    emotional_tone: Optional[str]
    importance: float
    entity_states: list[dict]
    open_loops_opened: list[dict]
    open_loops_closed: list[dict]
    model: str
    duration_ms: int


class Consolidator:
    """Calls a frontier model to consolidate episodes.

    Usage:
        c = Consolidator(api_key=os.environ['ANTHROPIC_API_KEY'])
        store = RecallStore(...)
        c.consolidate_pending(store)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_episodes_per_run: int = 200,
        max_concurrency: int = 12,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package is required for consolidation. "
                "Install it via `uv pip install anthropic`."
            ) from e
        self._anthropic = anthropic
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for consolidation. "
                "Set it in the Railway environment for hosted runs, or "
                "in your local shell for dev."
            )
        # The anthropic SDK's sync client is thread-safe; we share one
        # client across worker threads and rely on the SDK's internal
        # HTTP pooling.
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_episodes_per_run = max_episodes_per_run
        self.max_concurrency = max_concurrency

    # ------------------------------------------------------------------

    def consolidate_one(self, episode: Episode, raw_text: str) -> ConsolidationResult:
        started = time.time()
        user_prompt = self._format_user_prompt(episode, raw_text)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": CONSOLIDATION_SYSTEM_PROMPT,
                # Cache the (large, stable) system prompt across all
                # episodes in this run — cuts cost ~5x on batches.
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        parsed = _parse_json_strict(text)
        return ConsolidationResult(
            episode_id=episode.id,
            summary=parsed.get("summary", "").strip(),
            topics=list(parsed.get("topics", []))[:8],
            emotional_tone=parsed.get("emotional_tone"),
            importance=float(parsed.get("importance", 0.5) or 0.5),
            entity_states=list(parsed.get("entity_states", [])),
            open_loops_opened=list(parsed.get("open_loops_opened", [])),
            open_loops_closed=list(parsed.get("open_loops_closed", [])),
            model=self.model,
            duration_ms=int((time.time() - started) * 1000),
        )

    def apply_result(
        self,
        store: RecallStore,
        episode: Episode,
        raw_text: str,
        result: ConsolidationResult,
    ) -> None:
        """Write the consolidation output back into the store.

        Idempotent: re-running on the same episode overwrites prior
        consolidation. Bi-temporal facts get net-new IDs so prior facts
        with the same subject/predicate are *invalidated*, never
        overwritten.
        """
        now = datetime.now(timezone.utc)

        # 1. Update the episode row with summary, topics, tone, importance.
        episode.summary = result.summary
        episode.summary_model = result.model
        episode.topics = result.topics
        episode.emotional_tone = result.emotional_tone
        episode.importance = max(0.0, min(1.0, result.importance))
        episode.consolidation_time = now
        store.upsert_episode(episode, raw_text_for_fts=raw_text)

        # 2. Embed the summary locally.
        if result.summary:
            vec = LocalEmbedder.embed_one(result.summary)
            store.set_embedding(episode.id, vec, model=LocalEmbedder.name())

        # 3. Apply entity_states as bi-temporal facts.
        for state in result.entity_states:
            self._apply_entity_state(store, episode, state, now, result.model)

        # 4. Open loops — currently we just log them through the schema;
        #    full open-loop graph integration happens in synthesis.
        #    Left as a structured payload in the episode topics for now.

    def _apply_entity_state(
        self,
        store: RecallStore,
        episode: Episode,
        state: dict,
        now: datetime,
        model: str,
    ) -> None:
        subj = (state.get("subject") or "").strip()
        pred = (state.get("predicate") or "").strip()
        obj  = state.get("object") or ""
        if not (subj and pred):
            return

        valid_from = _parse_iso(state.get("valid_from")) or episode.time_start
        valid_until = _parse_iso(state.get("valid_until"))

        # Subject id resolution: best-effort. If the subject string
        # exactly matches an entity name we already have in episode
        # participants, use that id; otherwise hash the name. The
        # entity-resolution synthesis pass will reconcile later.
        subject_id = subj  # we keep the name as the id stem; downstream
                            # synthesis can map name -> canonical id.

        # Bi-temporal supersession: any currently-active fact with the
        # same (subject, predicate) gets invalidated as of valid_from.
        for prior in store.active_facts_for(subject_id):
            if prior.predicate == pred:
                store.invalidate_fact(prior.id, invalidated_by="pending", at=valid_from or now)

        fact_id = f"{subject_id}|{pred}|{episode.id}|{int(now.timestamp()*1000)}"
        fact = Fact(
            id=fact_id,
            subject_id=subject_id,
            predicate=pred,
            object_value=str(obj),
            object_kind=str(state.get("object_kind", "literal")),
            confidence=float(state.get("confidence", 0.7) or 0.7),
            valid_from=valid_from,
            valid_until=valid_until,
            invalidated_by=None,
            source_episode_ids=[episode.id],
            ingestion_time=now,
            summary_model=model,
        )
        store.upsert_fact(fact)

        # Update the placeholder invalidated_by we set above.
        store.conn.execute(
            "UPDATE facts SET invalidated_by = ? WHERE invalidated_by = 'pending'",
            (fact_id,),
        )

    # ------------------------------------------------------------------

    def consolidate_pending(
        self,
        store: RecallStore,
        raw_text_lookup,  # callable(Episode) -> str
        progress: Optional[callable] = None,  # progress(done, total)
    ) -> list[ConsolidationResult]:
        """Consolidate every un-consolidated episode (up to the run cap).

        Runs LLM calls concurrently via a thread pool. SQLite writes
        stay on the main thread because the store's connection isn't
        shared-thread-safe by default.

        `raw_text_lookup` is a callback the caller provides; it knows
        how to look up the raw content for a given episode (since that
        lives outside the recall DB — in the user's raw/*.jsonl files).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pending = store.pending_consolidation(limit=self.max_episodes_per_run)
        total = len(pending)
        results: list[ConsolidationResult] = []
        done = 0

        # Pre-fetch raw text serially (cheap I/O) so worker threads
        # only do the LLM call.
        episodes_with_text: list[tuple[Episode, str]] = []
        for ep in pending:
            raw_text = raw_text_lookup(ep) or ""
            if not raw_text.strip():
                ep.summary = ep.summary or "[no raw content recoverable]"
                ep.consolidation_time = datetime.now(timezone.utc)
                store.upsert_episode(ep)
                done += 1
                if progress:
                    progress(done, total)
                continue
            episodes_with_text.append((ep, raw_text))

        # Fire LLM calls in parallel. The Anthropic SDK's HTTP client
        # pool handles connection reuse.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futures = {
                ex.submit(self.consolidate_one, ep, raw_text): (ep, raw_text)
                for ep, raw_text in episodes_with_text
            }
            for fut in as_completed(futures):
                ep, raw_text = futures[fut]
                try:
                    r = fut.result()
                    # SQLite writes serially — safe and fast.
                    self.apply_result(store, ep, raw_text, r)
                    results.append(r)
                except Exception as e:
                    ep.consolidation_time = datetime.now(timezone.utc)
                    ep.summary = ep.summary or f"[consolidation failed: {type(e).__name__}]"
                    store.upsert_episode(ep)
                done += 1
                if progress:
                    progress(done, total)

        return results

    # ------------------------------------------------------------------

    def _format_user_prompt(self, episode: Episode, raw_text: str) -> str:
        return (
            f"EPISODE TYPE: {episode.kind.value}\n"
            f"TIME: {episode.time_start.isoformat()}"
            f"{(' to ' + episode.time_end.isoformat()) if episode.time_end else ''}\n"
            f"SOURCE: {episode.raw_source}\n"
            f"PLACE_ID: {episode.place_id or '(none)'}\n"
            f"PARTICIPANTS: {episode.participant_ids or '(none)'}\n\n"
            f"RAW CONTENT (truncated to 6000 chars):\n"
            f"{raw_text[:6000]}\n\n"
            f"Produce the JSON record."
        )


# ----------------------------------------------------------------------


def _parse_json_strict(text: str) -> dict:
    # Claude returns clean JSON when asked to, but defensively strip
    # any stray markdown fence.
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[: -3]
    return json.loads(t)


def _parse_iso(s) -> Optional[datetime]:
    if not s:
        return None
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None
