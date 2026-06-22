"""The Gate — `pmc watch` event-driven ingestion.

A long-running process that subscribes to OS-level signals (filesystem,
SQLite WAL, distributed notifications) and routes every event through
a classifier that decides promote / defer / drop.

This is the inversion of the polling scheduler we have today. The
Rust scheduler asks "what's new?" every N minutes per source. The
watch daemon is told "something changed" and reacts in seconds.

Module layout:
    daemon.py            - long-running asyncio process + signal handling
    event.py             - the Event dataclass + Decision enum
    router.py            - dispatches decisions to actions
    sources/
      fs.py              - watchdog/FSEvents subscriber
      sqlite.py          - WAL-aware DB watchers (day 2)
      distnotif.py       - NSDistributedNotificationCenter (day 3)
    classifier/
      rules.py           - the deterministic first pass
      llm.py             - batched fuzzy-case pass (day 4)
      learn.py           - per-user weight adjustments (day 5)
"""
