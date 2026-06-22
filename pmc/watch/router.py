"""Router — turns Classified decisions into actions.

PROMOTE: dispatch the relevant extractor against the user's graph.
         Implemented by shelling out to the pmc-ingest binary with a
         `--only <source>` flag (so we re-run just that source rather
         than the whole graph). Coalesces rapid bursts: if the same
         extractor is already running or was last run <COOLDOWN_SECS ago,
         we skip the new call but mark a `dirty` flag so the next tick
         picks it up.

DEFER:   push onto a JSONL queue at `~/.pmc/watch-queue.jsonl`. The
         consolidator / batch LLM picks these up later.

DROP:    log a counter (telemetry only); nothing persisted.

This module knows about the pmc-ingest binary's location (reuses
`pmc.cli.ingest._find_or_build` discovery logic). It doesn't know
anything about which extractors exist — the classifier names them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from pmc.cli.local_config import CONFIG_DIR, LocalConfig
from pmc.watch.event import Classified, Decision


log = logging.getLogger("pmc.watch.router")


QUEUE_FILE = CONFIG_DIR / "watch-queue.jsonl"
COOLDOWN_SECS = 30          # min seconds between same-extractor runs
COALESCE_WINDOW_SECS = 5    # bursts within this window collapse to one run


class Router:
    """Owns the action side: extractor dispatch, defer queue, counters."""

    def __init__(self, cfg: LocalConfig, *, binary: Path) -> None:
        self.cfg = cfg
        self.binary = binary
        self.last_run: dict[str, float] = defaultdict(float)
        self.dirty: dict[str, bool] = defaultdict(bool)
        self.counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def handle(self, c: Classified) -> None:
        self.counts[c.decision.value] += 1
        if c.decision == Decision.PROMOTE:
            if c.extractor:
                await self._promote(c.extractor)
        elif c.decision == Decision.DEFER:
            await self._defer(c)
        elif c.decision == Decision.DROP:
            # Telemetry only — already counted above.
            log.debug("drop  %s  · %s", c.event.short(), c.reason)

    # ----- PROMOTE ----------------------------------------------------

    async def _promote(self, extractor: str) -> None:
        now = time.time()
        last = self.last_run[extractor]
        if now - last < COOLDOWN_SECS:
            # Mark dirty so the cooldown loop picks it up later.
            self.dirty[extractor] = True
            log.debug("promote %-15s · within cooldown; marked dirty", extractor)
            return
        self.last_run[extractor] = now
        self.dirty[extractor] = False
        # Launch the extractor in the background.
        asyncio.create_task(self._run_extractor(extractor))

    async def _run_extractor(self, extractor: str) -> None:
        log.info("run     %-15s", extractor)
        user_id = self.cfg.user_id or "local"
        storage = str(self.cfg.effective_storage_root())
        # pmc-ingest doesn't yet support --only; for v1 we just run the
        # full ingest. The cooldown + coalescing prevents this from
        # being expensive in practice. Day-2 work adds a --only flag.
        cmd = [str(self.binary),
               "--user", user_id,
               "--root", storage,
               "--json"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env={**os.environ},
            )
            await proc.wait()
            log.info("done    %-15s · rc=%s", extractor, proc.returncode)
        except Exception as e:  # noqa: BLE001
            log.warning("error   %-15s · %s", extractor, e)

    # ----- DEFER ------------------------------------------------------

    async def _defer(self, c: Classified) -> None:
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": c.event.ts,
            "source": c.event.source.value,
            "kind": c.event.kind.value,
            "path": c.event.path,
            "reason": c.reason,
            "extractor_hint": c.extractor,
        }
        # Append-only, single-writer (the daemon).
        async with self._lock:
            with QUEUE_FILE.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        log.debug("defer   %s", c.event.short())

    # ----- Cooldown sweep ---------------------------------------------

    async def sweep_dirty(self) -> None:
        """Called on a slow tick (every COOLDOWN_SECS). Any extractor
        marked dirty since its last run gets fired now, respecting the
        cooldown that's elapsed."""
        now = time.time()
        for extractor, was_dirty in list(self.dirty.items()):
            if not was_dirty:
                continue
            if now - self.last_run[extractor] < COOLDOWN_SECS:
                continue
            self.last_run[extractor] = now
            self.dirty[extractor] = False
            asyncio.create_task(self._run_extractor(extractor))


__all__ = ["Router", "QUEUE_FILE"]
