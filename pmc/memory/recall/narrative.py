"""Monthly narrative / era detection.

Narrative memory is the user's life as a story — eras, arcs, identity
turns. We refresh it monthly because narrative shifts slowly and a
full LLM pass over a year of episodes is expensive enough that we
shouldn't run it every day.

The frontier model is the right call here too: era boundaries and
identity arcs require synthesis across hundreds of episodes; a small
model would produce vague, generic eras ("Spring 2026", "Summer 2026")
that aren't worth surfacing.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from pmc.memory.recall.schema import NarrativeSnapshot
from pmc.memory.recall.store import RecallStore


NARRATIVE_SYSTEM_PROMPT = """You are writing the user's life narrative as eras.

You receive a chronological digest of episode summaries — months or
years of the user's life. Identify the meaningful eras: periods when
the user's main project, primary relationships, or emotional register
was stable enough to be one thing. Eras are not seasons — they're the
shape of a stretch of life.

For each era produce:
  * label  — a short distinctive name (NOT a date range)
  * start  — ISO date
  * end    — ISO date or null for current
  * primary_themes — 2-4 things this era was about
  * key_people    — 2-5 people who defined it
  * summary — 2-3 sentences of texture

Then identify the **identity arcs** — multi-era threads that show
who the user is becoming: ambitions that recur, relationships that
deepen or fade, recurring tensions, taste shifts.

Finally, **current era** — special: name it, point to its turning
point, and write what feels new vs. carried over.

Return ONLY valid JSON, no markdown fence:

{
  "eras": [
    {
      "label": "...",
      "start": "ISO date",
      "end": "ISO date or null",
      "primary_themes": ["..."],
      "key_people": ["..."],
      "summary": "..."
    }
  ],
  "current_era": { ...same shape... },
  "identity_arcs": [
    {"label": "...", "spanning_eras": ["..."], "summary": "..."}
  ],
  "trajectory_notes": ["1-line observations about where things are going"]
}
"""


def build_narrative(
    store: RecallStore,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    episode_limit: int = 600,
) -> NarrativeSnapshot:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic is required") from e

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY required for narrative")
    client = anthropic.Anthropic(api_key=api_key)

    # Gather a sampled chronology — too many episodes will blow the
    # token budget, so we down-sample by date bucket.
    all_eps = [e for e in store.recent_episodes(limit=10_000) if e.summary]
    all_eps.sort(key=lambda e: e.time_start)
    digest = _down_sample_chronology(all_eps, target=episode_limit)

    payload = [
        {
            "time": e.time_start.isoformat(),
            "kind": e.kind.value,
            "summary": e.summary,
            "topics": e.topics,
            "tone": e.emotional_tone,
            "participants": e.participant_ids[:4],
            "importance": e.importance,
        }
        for e in digest
    ]

    started = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": NARRATIVE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": json.dumps(payload, default=str)[:160_000]}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    data = json.loads(text)

    now = datetime.now(timezone.utc)
    snap = NarrativeSnapshot(
        snapshot_month=now.strftime("%Y-%m"),
        eras=list(data.get("eras", [])),
        current_era=data.get("current_era"),
        identity_arcs=list(data.get("identity_arcs", [])),
        trajectory_notes=list(data.get("trajectory_notes", [])),
        produced_by=model,
        produced_at=now,
    )
    store.set_narrative(snap)
    snap.__dict__["_duration_ms"] = int((time.time() - started) * 1000)
    return snap


def _down_sample_chronology(eps, target: int):
    """Even sampling across time with importance-weighted retention."""
    if len(eps) <= target:
        return eps
    # Sort by importance descending and take a soft mix of high-importance
    # plus uniform-time samples to ensure coverage.
    by_imp = sorted(eps, key=lambda e: e.importance, reverse=True)
    top_n = by_imp[: target // 2]
    rest = [e for e in eps if e not in set(top_n)]
    if not rest:
        return top_n
    stride = max(1, len(rest) // (target - len(top_n)))
    uniform = rest[::stride]
    out = sorted(top_n + uniform, key=lambda e: e.time_start)
    return out[:target]
