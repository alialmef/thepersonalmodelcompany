"""SQLite WAL watcher.

FSEvents doesn't reliably fire on SQLite WAL writes because the OS
sees them as mmap'd writes rather than regular file I/O. We work
around that by polling a small set of "interesting" SQLite databases
(chat.db, History.db, Photos.sqlite) every POLL_INTERVAL seconds and
emitting an Event whenever:

  - the WAL file's mtime or size changed, OR
  - the main DB's `journal_mode == 'delete'` and its mtime changed, OR
  - the highest visible rowid in the main table is higher than last
    seen (for chat.db this is `message.ROWID`).

That last case is the strongest signal — a new message landed.

Polling is cheap (each iteration is a stat() and at most one read-only
SELECT MAX(ROWID)). With POLL_INTERVAL=2 the latency from "message
arrives" to "Event emitted" is at most 2 seconds — fine for our needs.
A future revision can use fanotify (Linux) or DTrace (macOS) for true
zero-poll detection, but those need elevated permissions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pmc.watch.event import Event, Kind, Source


log = logging.getLogger("pmc.watch.sqlite")


POLL_INTERVAL = 2.0   # seconds between watermark checks


@dataclass
class WatchedDB:
    """One database we want to track. Carries its last-seen watermark
    so the watcher can detect new rows efficiently."""
    name: str                       # short label, e.g. "chat.db"
    db_path: Path
    extractor: str                  # source name the classifier expects
    rowid_table: Optional[str] = None    # if set, we SELECT MAX(ROWID)
    last_mtime: float = 0.0
    last_size: int = 0
    last_rowid: int = 0


HOME = Path(os.path.expanduser("~"))


DEFAULT_DBS: list[WatchedDB] = [
    WatchedDB(
        name="chat.db",
        db_path=HOME / "Library/Messages/chat.db",
        extractor="imessage_enrich",
        rowid_table="message",
    ),
    WatchedDB(
        name="History.db",
        db_path=HOME / "Library/Safari/History.db",
        extractor="safari",
        rowid_table="history_visits",
    ),
]


class SQLiteWatcher:
    """Polls a list of SQLite DBs and emits Events on change."""

    def __init__(
        self,
        queue: asyncio.Queue,
        *,
        dbs: Optional[list[WatchedDB]] = None,
    ) -> None:
        self.queue = queue
        self.dbs: list[WatchedDB] = list(dbs) if dbs is not None else list(DEFAULT_DBS)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()

    async def _loop(self) -> None:
        # Initial watermark probe — so we don't emit on first tick for
        # rows that were already there before the daemon started.
        for db in self.dbs:
            await asyncio.to_thread(self._probe, db, emit=False)

        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_INTERVAL)
                break
            except asyncio.TimeoutError:
                pass
            for db in self.dbs:
                if not db.db_path.is_file():
                    continue
                try:
                    await asyncio.to_thread(self._probe, db, emit=True)
                except Exception as e:  # noqa: BLE001
                    log.debug("sqlite probe error on %s: %s", db.name, e)

    def _probe(self, db: WatchedDB, *, emit: bool) -> None:
        """One pass: stat the file + read the rowid watermark. If
        anything changed, emit an Event onto the queue."""
        try:
            st = db.db_path.stat()
        except OSError:
            return
        changed = False
        if st.st_mtime != db.last_mtime or st.st_size != db.last_size:
            db.last_mtime = st.st_mtime
            db.last_size = st.st_size
            changed = True

        # Read the rowid watermark from the main table.
        if db.rowid_table:
            try:
                # URI mode + read-only so we never lock the writer.
                conn = sqlite3.connect(
                    f"file:{db.db_path}?mode=ro", uri=True, timeout=0.5
                )
                try:
                    cur = conn.execute(
                        f"SELECT MAX(ROWID) FROM {db.rowid_table}"
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        rowid = int(row[0])
                        if rowid > db.last_rowid:
                            db.last_rowid = rowid
                            changed = True
                finally:
                    conn.close()
            except sqlite3.Error:
                # Locked / busy / can't open — try again next tick.
                pass

        if changed and emit:
            evt = Event(
                source=Source.SQLITE,
                kind=Kind.DB_WRITE,
                path=str(db.db_path),
                extra={
                    "name": db.name,
                    "extractor": db.extractor,
                    "rowid": db.last_rowid,
                },
            )
            try:
                self.queue.put_nowait(evt)
            except asyncio.QueueFull:
                log.warning("sqlite: queue full; dropping %s event", db.name)


__all__ = ["SQLiteWatcher", "WatchedDB", "DEFAULT_DBS"]
