"""Daily working-memory snapshot.

Working memory is the "what's live right now" view the agent reads on
every turn — before any per-query retrieval — so it never opens a
conversation cold. It's small, deliberately, so it can sit in the
agent's system prompt without bloating context.

The frontier model assembles it: we feed Claude the most recent N
episodes, the active facts, and recent open loops, and ask it to
produce a tight, ranked working set with anticipation items ("three
things the user might want surfaced today").
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from pmc.memory.recall.schema import WorkingMemorySnapshot
from pmc.memory.recall.store import RecallStore


WORKING_MEMORY_SYSTEM_PROMPT = """You assemble the user's working memory for today.

You receive the user's most recent episodes (already summarized), their
active open loops, and their hottest relationships. Produce a small,
ranked working set the agent will read on every interaction.

Be specific. Surface live decisions, recent inflections, planned things
that haven't happened. Drop anything older than ~30 days unless it's
unresolved.

Most importantly: produce 3-5 *anticipation items* — things the agent
might proactively surface to the user today. Examples:
  - "Dad's birthday is in 9 days — usually start planning around 2 weeks out"
  - "You drafted a reply to Ali on the funding round 3 days ago and didn't send"
  - "You've mentioned LA 4 times in the last 2 weeks but haven't acted"

Return ONLY valid JSON, no markdown fence:

{
  "top_open_loops": [{"summary": "...", "kind": "...", "liveness": 0.0-1.0}],
  "hot_people": [{"name": "...", "context": "one-line why they're hot"}],
  "rising_themes": [{"label": "...", "context": "what's driving the rise"}],
  "upcoming_events": [{"label": "...", "when": "ISO date or natural"}],
  "recent_episodes": [{"episode_id": "...", "summary": "...", "weight": 0.0-1.0}],
  "anticipation": ["item 1", "item 2", "item 3"]
}
"""


def build_working_memory(
    store: RecallStore,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    recent_episode_limit: int = 40,
) -> WorkingMemorySnapshot:
    """Build today's working-memory snapshot with a frontier model."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic is required") from e

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY required for working memory")
    client = anthropic.Anthropic(api_key=api_key)

    # Gather inputs.
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent_eps = [
        e for e in store.recent_episodes(limit=recent_episode_limit * 2)
        if e.time_start >= cutoff and e.summary
    ][:recent_episode_limit]

    active_facts = store.all_active_facts()[:200]

    payload = {
        "recent_episodes": [
            {
                "id": e.id,
                "kind": e.kind.value,
                "time": e.time_start.isoformat(),
                "summary": e.summary,
                "topics": e.topics,
                "tone": e.emotional_tone,
                "importance": e.importance,
                "participants": e.participant_ids,
            }
            for e in recent_eps
        ],
        "active_facts": [
            {
                "subject": f.subject_id,
                "predicate": f.predicate,
                "object": f.object_value,
                "valid_from": f.valid_from.isoformat() if f.valid_from else None,
                "confidence": f.confidence,
            }
            for f in active_facts
        ],
    }

    started = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": WORKING_MEMORY_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": json.dumps(payload, default=str)[:60000]}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    data = json.loads(text)

    snap = WorkingMemorySnapshot(
        snapshot_date=datetime.now(timezone.utc),
        top_open_loops=list(data.get("top_open_loops", [])),
        hot_people=list(data.get("hot_people", [])),
        rising_themes=list(data.get("rising_themes", [])),
        upcoming_events=list(data.get("upcoming_events", [])),
        recent_episodes=list(data.get("recent_episodes", [])),
        anticipation=list(data.get("anticipation", []))[:5],
        produced_by=model,
        produced_at=datetime.now(timezone.utc),
    )
    store.set_working_memory(snap)
    snap.__dict__["_duration_ms"] = int((time.time() - started) * 1000)
    return snap
