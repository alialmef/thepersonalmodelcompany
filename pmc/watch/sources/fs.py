"""FSEvents source — wraps `watchdog` for macOS.

Watches each of the data-source directories listed in
`pmc.watch.classifier.rules.WATCHED_PATHS`. Pushes one Event onto a
shared asyncio queue per filesystem change. The daemon drains the
queue and runs each event through the classifier.

We don't filter here — the classifier owns the promote/drop logic.
The source's only job is "convert raw watchdog events into our Event
shape, push them on the bus." That keeps the rules in one place and
lets the LLM tier (later) see every event the OS reported.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Iterable

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from pmc.watch.event import Event, Kind, Source


log = logging.getLogger("pmc.watch.fs")


# Roots we want to subscribe to. Coalesced and deduped at startup;
# `watchdog` complains if you register overlapping subtrees.
HOME = Path(os.path.expanduser("~"))

DEFAULT_ROOTS: list[Path] = [
    # Messaging
    HOME / "Library/Messages",
    # Mail
    HOME / "Library/Mail",
    # Voice memos
    HOME / "Library/Application Support/com.apple.voicememos",
    HOME / "Library/Group Containers/group.com.apple.VoiceMemos.shared",
    # Notes (CloudKit)
    HOME / "Library/Group Containers/group.com.apple.notes",
    # Calendar
    HOME / "Library/Calendars",
    # Reminders
    HOME / "Library/Group Containers/group.com.apple.reminders",
    # Safari
    HOME / "Library/Safari",
    # Chrome
    HOME / "Library/Application Support/Google/Chrome/Default",
    # Wallet
    HOME / "Library/Passes",
    # iCloud Drive
    HOME / "Library/Mobile Documents",
    # Shell history — polled separately by `pmc.watch.sources.sqlite` style
    # poll (added in a follow-up). Watching HOME is too broad — every
    # third-party app's Library/ subdir would fire events.
]


class _Handler(FileSystemEventHandler):
    """Translates watchdog events into pmc Events on the queue."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.queue = queue
        self.loop = loop

    def on_created(self, e: FileSystemEvent) -> None:
        self._push(e, Kind.CREATED)

    def on_modified(self, e: FileSystemEvent) -> None:
        self._push(e, Kind.MODIFIED)

    def on_deleted(self, e: FileSystemEvent) -> None:
        self._push(e, Kind.DELETED)

    def on_moved(self, e: FileSystemEvent) -> None:
        self._push(e, Kind.MOVED, extra={"dest": getattr(e, "dest_path", "")})

    def _push(self, e: FileSystemEvent, kind: Kind, extra: dict | None = None) -> None:
        # Skip directory events — too noisy. The actual file write
        # events for contents inside will fire as well.
        if getattr(e, "is_directory", False):
            return
        evt = Event(
            source=Source.FS,
            kind=kind,
            path=e.src_path,
            extra=extra or {},
        )
        # Cross-thread queue push — watchdog calls us from its own thread.
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, evt)
        except RuntimeError:
            # Loop may be shutting down. Drop quietly.
            pass


class FSWatcher:
    """Holds the watchdog Observer + lifecycle."""

    def __init__(
        self,
        queue: asyncio.Queue,
        *,
        roots: Iterable[Path] | None = None,
    ) -> None:
        self.queue = queue
        self.roots = list(roots) if roots is not None else list(DEFAULT_ROOTS)
        self.observer: Observer | None = None

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        handler = _Handler(self.queue, loop)
        obs = Observer()
        for root in _coalesce_roots(self.roots):
            if not root.exists():
                log.debug("fs: skipping non-existent root %s", root)
                continue
            # recursive=True so we get events for files nested under
            # the root (e.g. Mail's mboxes, Messages' attachments).
            try:
                obs.schedule(handler, str(root), recursive=True)
                log.info("fs: watching %s", root)
            except (OSError, PermissionError) as e:
                log.warning("fs: can't watch %s: %s", root, e)
        obs.start()
        self.observer = obs

    def stop(self) -> None:
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None


def _coalesce_roots(roots: list[Path]) -> list[Path]:
    """Drop any root that's a subdirectory of another root in the
    list — watchdog errors on overlapping schedules."""
    sorted_roots = sorted({Path(p) for p in roots}, key=lambda p: len(str(p)))
    keep: list[Path] = []
    for r in sorted_roots:
        if any(str(r).startswith(str(k) + "/") for k in keep):
            continue
        keep.append(r)
    return keep


__all__ = ["FSWatcher", "DEFAULT_ROOTS"]
