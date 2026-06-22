"""The agenda — the actual output the agent uses to help the user.

Time + reading + activity get cross-referenced into two structured
lists:

  ACTIONABLE — concrete pieces of work / decisions / threads the agent
               could help with NOW. Each item carries: what it is,
               evidence (where in the data it came from), suggested
               next move.

  LEARNABLE  — patterns the user does repeatedly. Each item is a task
               the agent could LEARN to do on the user's behalf
               (drafting similar replies, filing similar things,
               running similar searches). The seed for personal RL.

This is universal — no assumptions about coder / creator / parent /
student. The LLM reads everything we know about the user and produces
items shaped to THAT user's actual life.

Output: portrait/agenda.md + portrait/agenda.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from pmc.agent.providers.base import Message, ProviderConfig
from pmc.agent.providers.registry import get_provider
from pmc.cli.local_config import LocalConfig


log = logging.getLogger("pmc.consolidator.agenda")


@dataclass
class AgendaItem:
    kind: str                    # "actionable" | "learnable"
    label: str                   # short headline
    detail: str                  # 1-2 sentences
    evidence: list[str]          # citations: file refs / message excerpts / URLs
    confidence: str              # "high" | "medium" | "low"
    suggested_action: str = ""   # for actionable: what the agent could do
    automation_hint: str = ""    # for learnable: how the agent might learn it


@dataclass
class Agenda:
    actionable: list[AgendaItem] = field(default_factory=list)
    learnable: list[AgendaItem] = field(default_factory=list)
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_agenda(
    cfg: LocalConfig,
    *,
    user_id: str,
    self_md: str,
    time_md: str,
    reading_md: str,
    recent_md: str,
) -> Agenda:
    """Compose the agenda from the portrait layers. One LLM call.
    Reads:
      - self.md           — who the user is
      - time.md           — how they spend time
      - reading.md        — what they've been thinking about
      - time_recent.md    — what they've literally done lately
    Asks the model to surface concrete items, not interpretation."""

    evidence_block = "\n\n---\n\n".join([
        f"WHO THEY ARE:\n{self_md.strip()}",
        f"HOW THEY SPEND TIME:\n{time_md.strip()}",
        f"WHAT THEY'VE BEEN READING / THINKING ABOUT:\n{reading_md.strip()}",
        f"WHAT THEY ACTUALLY DID, LAST 7 DAYS:\n{recent_md.strip()}",
    ])

    text = _ask_llm(
        cfg, system=_AGENDA_SYSTEM, user=evidence_block, max_tokens=4000,
    )
    agenda = _parse_agenda(text)
    if not agenda.actionable and not agenda.learnable:
        # Retry once with a stricter JSON-only instruction.
        text2 = _ask_llm(
            cfg,
            system=_AGENDA_SYSTEM + "\n\nIMPORTANT: respond with VALID JSON ONLY.",
            user=evidence_block, max_tokens=4000,
        )
        agenda = _parse_agenda(text2)
    from datetime import datetime, timezone
    agenda.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_agenda(cfg, user_id, agenda)
    return agenda


_AGENDA_SYSTEM = """\
You are looking at the user's structured digital life and extracting two
concrete lists:

  ACTIONABLE — things the user has on their plate RIGHT NOW that an
               agent could help with. Each one must be specific enough
               that the user would recognize it. Not "be more productive"
               — but "draft the reply to the June 11 message" or
               "decide whether to bind the Progressive auto policy" or
               "evaluate the Cohere offer."

  LEARNABLE — repeating patterns the user does, where an agent could
              learn to do that task on their behalf. The seed of
              personal RL environments. Examples: "user repeatedly looks
              up X kind of doc" → agent could learn to fetch + summarize.
              "user processes calendar invites a certain way" → agent
              could learn to triage.

Rules:
  - Be specific to THIS person. Generic productivity advice is failure.
  - Cite evidence inline (which page, which thread, which pattern).
  - If the data doesn't support an item, don't include it. Better to
    have 3 sharp items than 10 hand-wavy ones.
  - Confidence: "high" only when the evidence is unambiguous.
  - This must work for ANY user — don't assume they're a coder, parent,
    student, etc. Let the evidence dictate the shape.

Return ONLY valid JSON in this exact shape (no code fences, no prose):
{
  "actionable": [
    {
      "label": "short headline (max ~70 chars)",
      "detail": "1-2 sentences explaining what this is and why it's open",
      "evidence": ["specific citation 1", "specific citation 2"],
      "confidence": "high|medium|low",
      "suggested_action": "what an agent could DO about this, one line"
    }
  ],
  "learnable": [
    {
      "label": "short pattern name",
      "detail": "what the user does repeatedly",
      "evidence": ["citation 1", "citation 2"],
      "confidence": "high|medium|low",
      "automation_hint": "how an agent could learn to do this — one line"
    }
  ]
}

5-12 items total across both lists. Quality over quantity.
"""


def _parse_agenda(text: str) -> Agenda:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("agenda: JSON parse failed: %s", e)
        return Agenda()
    out = Agenda()
    for row in data.get("actionable", []) or []:
        if not isinstance(row, dict):
            continue
        out.actionable.append(AgendaItem(
            kind="actionable",
            label=(row.get("label") or "").strip(),
            detail=(row.get("detail") or "").strip(),
            evidence=row.get("evidence") or [],
            confidence=(row.get("confidence") or "medium").lower(),
            suggested_action=(row.get("suggested_action") or "").strip(),
        ))
    for row in data.get("learnable", []) or []:
        if not isinstance(row, dict):
            continue
        out.learnable.append(AgendaItem(
            kind="learnable",
            label=(row.get("label") or "").strip(),
            detail=(row.get("detail") or "").strip(),
            evidence=row.get("evidence") or [],
            confidence=(row.get("confidence") or "medium").lower(),
            automation_hint=(row.get("automation_hint") or "").strip(),
        ))
    return out


def _write_agenda(cfg: LocalConfig, user_id: str, agenda: Agenda) -> None:
    root = cfg.effective_storage_root()
    portrait_dir = root / "users" / user_id / "graph" / "synth" / "portrait"
    portrait_dir.mkdir(parents=True, exist_ok=True)

    # JSONL — machine-readable
    jl = portrait_dir / "agenda.jsonl"
    with jl.open("w") as f:
        for item in agenda.actionable + agenda.learnable:
            f.write(json.dumps(asdict(item)) + "\n")

    # Markdown — human-readable
    md_lines: list[str] = []
    md_lines.append("# Agenda")
    md_lines.append("")
    md_lines.append(f"_generated at {agenda.generated_at}_")
    md_lines.append("")
    md_lines.append("## Actionable — things on the user's plate right now")
    md_lines.append("")
    if not agenda.actionable:
        md_lines.append("_(none surfaced)_")
    else:
        for it in agenda.actionable:
            md_lines.append(f"### {it.label}")
            md_lines.append("")
            md_lines.append(it.detail)
            md_lines.append("")
            if it.suggested_action:
                md_lines.append(f"_Suggested action:_ {it.suggested_action}")
                md_lines.append("")
            if it.evidence:
                md_lines.append("Evidence:")
                for e in it.evidence:
                    md_lines.append(f"  - {e}")
                md_lines.append("")
            md_lines.append(f"_Confidence: {it.confidence}_")
            md_lines.append("")
    md_lines.append("## Learnable — patterns an agent could automate")
    md_lines.append("")
    if not agenda.learnable:
        md_lines.append("_(none surfaced)_")
    else:
        for it in agenda.learnable:
            md_lines.append(f"### {it.label}")
            md_lines.append("")
            md_lines.append(it.detail)
            md_lines.append("")
            if it.automation_hint:
                md_lines.append(f"_How an agent could learn this:_ {it.automation_hint}")
                md_lines.append("")
            if it.evidence:
                md_lines.append("Evidence:")
                for e in it.evidence:
                    md_lines.append(f"  - {e}")
                md_lines.append("")
            md_lines.append(f"_Confidence: {it.confidence}_")
            md_lines.append("")

    (portrait_dir / "agenda.md").write_text("\n".join(md_lines))
    log.info(
        "agenda: wrote %d actionable + %d learnable",
        len(agenda.actionable), len(agenda.learnable),
    )


def _ask_llm(cfg: LocalConfig, *, system: str, user: str, max_tokens: int) -> str:
    provider = get_provider(cfg.provider)
    if provider is None:
        raise RuntimeError(f"unknown provider {cfg.provider!r}")
    pcfg = ProviderConfig(
        provider=cfg.provider, model=cfg.model, api_key=cfg.api_key,
    )
    async def _call() -> str:
        resp = await provider.chat(
            messages=[Message(role="user", content=user)],
            config=pcfg, system=system, max_tokens=max_tokens,
        )
        return resp.text
    return asyncio.run(_call())


__all__ = ["build_agenda", "Agenda", "AgendaItem"]
