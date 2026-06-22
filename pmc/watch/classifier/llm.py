"""LLM tier — the gate's second pass.

The rules layer handles the 95% case in microseconds. Anything tagged
DEFER with reason "no rule matched" goes into a small queue that this
module drains once a minute. The LLM looks at a batch of ambiguous
events at once and decides for each:

  - PROMOTE → which extractor to run
  - DEFER   → keep in queue for next pass (still ambiguous, but maybe
              a future signal will clarify)
  - DROP    → fully reject

Uses the user's configured provider via pmc.agent.providers. Cheap
model preferred (Haiku, gpt-mini) — this runs continuously and the
rules already catch the easy stuff.

The prompt is tight and structured-output:

  "Here are 20 ambiguous events that just happened on the user's Mac.
  For each, return a single-word verdict: PROMOTE_X (where X is one
  of the known extractors), DEFER, or DROP. One per line."

We parse and dispatch. Anything unparseable defaults to DROP.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
)
from pmc.agent.providers.registry import get_provider
from pmc.cli.local_config import LocalConfig
from pmc.watch.classifier.rules import HOME, WATCHED_PATHS
from pmc.watch.event import Classified, Decision, Event


log = logging.getLogger("pmc.watch.llm")


BATCH_INTERVAL_SECS = 60     # once per minute
MAX_BATCH = 30               # 30 events per LLM call ceiling


KNOWN_EXTRACTORS: list[str] = sorted({e for _, e, _ in WATCHED_PATHS})


@dataclass
class _Pending:
    event: Event
    received_at: float = 0.0
    attempts: int = 0


class LLMClassifier:
    """Buffers ambiguous events and drains them on a timer."""

    def __init__(self, cfg: LocalConfig) -> None:
        self.cfg = cfg
        self._buf: list[_Pending] = []
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._drain_loop())

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()

    async def enqueue(self, c: Classified) -> None:
        # Called by the router for any rules-DEFER event with reason
        # "no rule matched". We accumulate; the drain loop calls the
        # LLM on a slow tick.
        if c.decision != Decision.DEFER:
            return
        async with self._lock:
            self._buf.append(_Pending(event=c.event))

    async def _drain_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=BATCH_INTERVAL_SECS)
                break
            except asyncio.TimeoutError:
                pass
            await self._drain_one_batch()

    async def _drain_one_batch(self) -> list[Classified]:
        """Pull up to MAX_BATCH events, classify, return per-event decisions."""
        async with self._lock:
            if not self._buf:
                return []
            batch = self._buf[:MAX_BATCH]
            self._buf = self._buf[MAX_BATCH:]

        decisions = await self._ask_llm(batch)
        return decisions

    async def _ask_llm(self, batch: list[_Pending]) -> list[Classified]:
        """One provider call, structured JSON output, parse per-event."""
        provider = get_provider(self.cfg.provider)
        if provider is None:
            log.warning("llm: unknown provider %r", self.cfg.provider)
            return []
        pcfg = ProviderConfig(
            provider=self.cfg.provider,
            model=self.cfg.model,
            api_key=self.cfg.api_key,
        )

        prompt_lines: list[str] = []
        for i, p in enumerate(batch):
            short_path = p.event.path
            # Make paths $HOME-relative for prompt brevity.
            try:
                short_path = "~/" + str(p.event.path).removeprefix(str(HOME) + "/")
            except Exception:  # noqa: BLE001
                pass
            prompt_lines.append(
                f"{i+1}. [{p.event.source.value}/{p.event.kind.value}] {short_path}"
            )

        system = _SYSTEM_PROMPT.format(
            extractors="\n  - ".join(KNOWN_EXTRACTORS),
        )
        user_msg = (
            "Events to classify (one per line). Return ONLY a JSON array, "
            "one object per event in the same order:\n\n"
            + "\n".join(prompt_lines)
        )

        try:
            resp = await provider.chat(
                messages=[Message(role="user", content=user_msg)],
                config=pcfg,
                system=system,
                max_tokens=1024,
            )
        except ProviderError as e:
            log.warning("llm: provider error %s: %s", e.kind, e)
            return []
        except Exception as e:  # noqa: BLE001
            log.warning("llm: unexpected error: %s", e)
            return []

        return _parse_response(resp.text, batch)


_SYSTEM_PROMPT = """\
You are the second-tier classifier in a personal-data ingestion gate.
The user's Mac is being watched for filesystem and OS events. The
rules layer handles obvious cases. You get the ambiguous ones.

For each event, decide ONE of:
  - PROMOTE  (and which extractor to run) — when the event is signal
             we want to ingest into the personal graph.
  - DEFER    — when it might become interesting later but isn't now.
  - DROP     — when it's not worth keeping.

Known extractors:
  - {extractors}

Return ONLY a JSON array, one object per input event in the same
order. Each object must have these keys:
  {{"verdict": "PROMOTE"|"DEFER"|"DROP",
   "extractor": "<name or null>",
   "reason": "<short string, <60 chars>"}}

Be aggressive about DROP. The graph already has signal; we don't
need every random file write. If you cannot identify what an event
means, prefer DROP over DEFER.
"""


def _parse_response(text: str, batch: list[_Pending]) -> list[Classified]:
    """Best-effort JSON parse. Anything unparseable defaults to DROP."""
    text = text.strip()
    # Tolerate code fences.
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        log.warning("llm: response not JSON; dropping batch")
        return [
            Classified(event=p.event, decision=Decision.DROP,
                       reason="llm response unparseable")
            for p in batch
        ]
    if not isinstance(arr, list):
        return []

    out: list[Classified] = []
    for p, row in zip(batch, arr):
        if not isinstance(row, dict):
            out.append(Classified(event=p.event, decision=Decision.DROP,
                                  reason="llm row malformed"))
            continue
        verdict = (row.get("verdict") or "").upper()
        extractor = row.get("extractor") or None
        reason = (row.get("reason") or "")[:80]
        if verdict == "PROMOTE" and extractor in KNOWN_EXTRACTORS:
            out.append(Classified(
                event=p.event, decision=Decision.PROMOTE,
                reason=reason, extractor=extractor, confidence=0.7,
            ))
        elif verdict == "DEFER":
            out.append(Classified(event=p.event, decision=Decision.DEFER,
                                  reason=reason, confidence=0.5))
        else:
            out.append(Classified(event=p.event, decision=Decision.DROP,
                                  reason=reason or "llm dropped"))
    return out


__all__ = ["LLMClassifier"]
