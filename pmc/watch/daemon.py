"""`pmc watch` daemon.

A long-running asyncio process. Three concurrent tasks:

  1. FSWatcher (and later SQLite/distnotif sources) push Events onto
     a shared asyncio.Queue.
  2. The classifier loop drains the queue, applies rules, hands off
     to the Router.
  3. The cooldown sweep runs every COOLDOWN_SECS and re-fires any
     extractor that got marked dirty during a cooldown.

Lifecycle:
  - Caught SIGINT/SIGTERM trigger a clean shutdown — sources stop,
    queue drains, in-flight extractor tasks are awaited briefly
    (with a 5s timeout), then exit 0.
  - Logs go to ~/.pmc/watch.log when running detached, stderr when
    running interactively.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from pmc.cli.ingest import _find_or_build
from pmc.cli.local_config import CONFIG_DIR, LocalConfig, load
from pmc.watch.classifier import classify
from pmc.watch.classifier.llm import LLMClassifier
from pmc.watch.event import Decision, Event
from pmc.watch.router import COOLDOWN_SECS, Router
from pmc.watch.sources.fs import FSWatcher
from pmc.watch.sources.sqlite import SQLiteWatcher
from pmc.watch.sources.workspace import WorkspaceWatcher


LOG_FILE = CONFIG_DIR / "watch.log"


log = logging.getLogger("pmc.watch")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(*, log_to_file: bool = False, verbose: bool = False) -> int:
    cfg = load()
    if cfg is None:
        sys.stderr.write("no pmc config — run `pmc configure` first.\n")
        return 1

    from rich.console import Console
    console = Console()
    _setup_logging(log_to_file=log_to_file, verbose=verbose)

    try:
        asyncio.run(_main(cfg, console))
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
    return 0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _main(cfg: LocalConfig, console) -> None:
    from pmc.cli import ui

    # Discover the pmc-ingest binary up-front so we fail fast if it's
    # missing rather than after a watcher event.
    binary = _find_or_build(console, allow_build=True)
    if binary is None:
        log.error("pmc-ingest binary not found and could not be built; exiting")
        return

    queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10_000)
    router = Router(cfg, binary=binary)

    fs = FSWatcher(queue)
    fs.start()

    sqlite = SQLiteWatcher(queue)
    sqlite.start()

    workspace = WorkspaceWatcher(queue)
    workspace.start()

    llm = LLMClassifier(cfg)
    llm.start()

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    # Print the running banner (so it's obvious the daemon is alive).
    console.clear()
    ui.banner_top(console, title="pmc watch",
                  subtitle=f"user {(cfg.user_id or 'local')[:8]}…  ·  watching")

    classifier_task = asyncio.create_task(_classifier_loop(queue, router, llm))
    llm_drain_task = asyncio.create_task(_llm_drain_loop(llm, router, stop))
    sweep_task = asyncio.create_task(_sweep_loop(router, stop))
    status_task = asyncio.create_task(_status_loop(router, console, stop))
    consol_task = asyncio.create_task(_consolidator_loop(cfg, stop))

    await stop.wait()

    log.info("stopping")
    fs.stop()
    sqlite.stop()
    workspace.stop()
    llm.stop()
    classifier_task.cancel()
    llm_drain_task.cancel()
    sweep_task.cancel()
    status_task.cancel()
    consol_task.cancel()
    for t in (classifier_task, llm_drain_task, sweep_task, status_task, consol_task):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


async def _classifier_loop(
    queue: asyncio.Queue[Event],
    router: Router,
    llm: LLMClassifier,
) -> None:
    while True:
        evt = await queue.get()
        try:
            c = classify(evt)
            # Ambiguous DEFERs (rules couldn't match) → LLM tier.
            # Other DEFERs (e.g. Mail, photos batch) go straight to the queue.
            if c.decision == Decision.DEFER and "no rule matched" in c.reason:
                await llm.enqueue(c)
            else:
                await router.handle(c)
        except Exception as e:  # noqa: BLE001
            log.warning("classifier error on %s: %s", evt.short(), e)
        finally:
            queue.task_done()


async def _llm_drain_loop(
    llm: LLMClassifier,
    router: Router,
    stop: asyncio.Event,
) -> None:
    """The LLM internally batches on a timer; we just wait on it.
    The LLM returns classified decisions; route each."""
    from pmc.watch.classifier.llm import BATCH_INTERVAL_SECS
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=BATCH_INTERVAL_SECS)
            break
        except asyncio.TimeoutError:
            pass
        try:
            results = await llm._drain_one_batch()
            for c in results:
                await router.handle(c)
        except Exception as e:  # noqa: BLE001
            log.warning("llm drain error: %s", e)


async def _sweep_loop(router: Router, stop: asyncio.Event) -> None:
    """Periodic dirty-flag sweep."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=COOLDOWN_SECS)
            break
        except asyncio.TimeoutError:
            await router.sweep_dirty()


async def _consolidator_loop(cfg: LocalConfig, stop: asyncio.Event) -> None:
    """Run the consolidator daily. Checks every hour whether a pass
    is due; if so, runs in a worker thread so it doesn't block the
    event loop."""
    import time as _time
    from pmc.consolidator import run_consolidation
    from pmc.consolidator.run import last_run_at

    CHECK_INTERVAL = 60 * 60       # check every hour
    DAILY = 24 * 60 * 60           # 24h cadence

    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=CHECK_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass
        last = last_run_at(cfg) or 0
        if _time.time() - last < DAILY:
            continue
        log.info("consolidator: daily cadence reached; firing")
        try:
            await asyncio.to_thread(run_consolidation, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("consolidator failed: %s", e)


async def _status_loop(router: Router, console, stop: asyncio.Event) -> None:
    """Periodic status print — keeps the terminal feeling alive."""
    from pmc.cli import ui
    last_total = -1
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
            break
        except asyncio.TimeoutError:
            pass
        total = sum(router.counts.values())
        if total == last_total:
            continue
        last_total = total
        line = (
            f"events  {total}    "
            f"promoted {router.counts.get('promote', 0)}    "
            f"deferred {router.counts.get('defer', 0)}    "
            f"dropped  {router.counts.get('drop', 0)}"
        )
        console.print(f"{ui.margin()}[{ui.DIM}]{line}[/]")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows / restricted env — skip.
            pass


def _setup_logging(*, log_to_file: bool, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-5s  %(name)s  %(message)s"
    handlers: list[logging.Handler] = []
    if log_to_file:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    # Quiet watchdog's chatty INFO logs at our INFO level.
    logging.getLogger("watchdog").setLevel(logging.WARNING)


__all__ = ["run"]
