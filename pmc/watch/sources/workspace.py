"""NSWorkspace source — emits Events when the foreground app changes,
when the user locks/unlocks the screen, and when the machine sleeps
or wakes.

This is the "what are you doing right now" signal that FSEvents and
SQLite watching cannot see — it's about attention, not data.

Implementation: PyObjC. We register as an observer on
`NSWorkspace.sharedWorkspace().notificationCenter()` for a small set
of notifications and forward each one as an Event onto the shared
queue. The Cocoa runloop is driven by spinning it from a background
thread (PyObjC handles the GIL coordination).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from pmc.watch.event import Event, Kind, Source


log = logging.getLogger("pmc.watch.workspace")


# Notifications we subscribe to. Names per Apple docs.
NOTIFS_TO_EXTRACTOR = {
    "NSWorkspaceDidActivateApplicationNotification":   "app_usage",
    "NSWorkspaceDidDeactivateApplicationNotification": "app_usage",
    "NSWorkspaceWillSleepNotification":                "app_usage",
    "NSWorkspaceDidWakeNotification":                  "app_usage",
    "NSWorkspaceScreensDidSleepNotification":          "app_usage",
    "NSWorkspaceScreensDidWakeNotification":           "app_usage",
    # Screen lock/unlock are on the *distributed* notification center,
    # not workspace — wired separately below.
}


class WorkspaceWatcher:
    """Subscribes to NSWorkspace + NSDistributedNotificationCenter and
    posts Events to the shared queue."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._observer = None  # the PyObjC delegate object

    def start(self) -> None:
        try:
            import AppKit  # noqa: F401  — just probe the import
            import Foundation  # noqa: F401
        except Exception as e:  # noqa: BLE001
            log.warning("workspace: PyObjC unavailable — %s", e)
            return
        self.loop = asyncio.get_event_loop()
        self._thread = threading.Thread(
            target=self._run_runloop, daemon=True, name="pmc-workspace"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # The runloop will exit when CFRunLoopStop is signaled; we let
        # it die with the daemon process since it's a daemon thread.

    # ------------------------------------------------------------------

    def _run_runloop(self) -> None:
        import AppKit
        from Foundation import (
            NSObject,
            NSDistributedNotificationCenter,
        )

        queue = self.queue
        loop = self.loop

        class _Observer(NSObject):
            def handle_(self, notif):
                try:
                    name = str(notif.name())
                    info = notif.userInfo() or {}
                    # Extract bundle id of the activated app if present.
                    bundle = ""
                    app_obj = info.get("NSWorkspaceApplicationKey")
                    if app_obj is not None:
                        try:
                            bundle = str(app_obj.bundleIdentifier() or "")
                        except Exception:  # noqa: BLE001
                            bundle = ""
                    extractor = NOTIFS_TO_EXTRACTOR.get(name, "app_usage")
                    evt = Event(
                        source=Source.DISTNOTIF,
                        kind=Kind.APP_EVENT,
                        path=name,
                        extra={
                            "bundle_id": bundle,
                            "extractor": extractor,
                        },
                    )
                    # Push onto the asyncio queue via the loop.
                    if loop is not None:
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, evt)
                        except RuntimeError:
                            pass
                except Exception as e:  # noqa: BLE001
                    log.debug("workspace handler error: %s", e)

        obs = _Observer.alloc().init()
        self._observer = obs

        # Workspace notifications
        ws_center = AppKit.NSWorkspace.sharedWorkspace().notificationCenter()
        for name in NOTIFS_TO_EXTRACTOR.keys():
            ws_center.addObserver_selector_name_object_(
                obs, b"handle:", name, None
            )

        # Distributed notifications — screen lock/unlock
        dist_center = NSDistributedNotificationCenter.defaultCenter()
        for name in (
            "com.apple.screenIsLocked",
            "com.apple.screenIsUnlocked",
        ):
            dist_center.addObserver_selector_name_object_(
                obs, b"handle:", name, None
            )

        log.info("workspace: subscribed to %d NSWorkspace + 2 distributed notifications",
                 len(NOTIFS_TO_EXTRACTOR))

        # Spin the runloop until stopped.
        from Foundation import NSRunLoop, NSDate
        runloop = NSRunLoop.currentRunLoop()
        while not self._stop.is_set():
            runloop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.5))


__all__ = ["WorkspaceWatcher"]
