"""Rule-based classifier — the gate's first pass.

Every filesystem/DB/notif event hits these rules. The rules answer:
   - is this from a source we know?
   - is it a kind of change that produces signal?
   - should it run an extractor now, queue for batch, or be dropped?

Rules are deterministic — same input always gives the same Classified
output. Ambiguous cases (new senders, novel files, unknown directories
under the user's HOME) are tagged `DEFER` so the LLM tier can look at
them in a batch. Truly junk events (Spotlight indexing temp files,
.DS_Store writes) are dropped outright.

The mapping from rule-match to extractor name uses the same names the
Rust binary uses (`messages`, `mail`, `voice_memos`, etc.) so the
router can dispatch by name without translation.
"""

from __future__ import annotations

import os
from pathlib import Path

from pmc.watch.classifier.learn import FeedbackOverlay
from pmc.watch.event import Classified, Decision, Event, Kind, Source


# Shared overlay instance — refreshed when feedback is recorded.
_OVERLAY: FeedbackOverlay | None = None


def _overlay() -> FeedbackOverlay:
    global _OVERLAY
    if _OVERLAY is None:
        _OVERLAY = FeedbackOverlay()
    return _OVERLAY


def refresh_overlay() -> None:
    """Call after recording feedback to pick up the new overrides."""
    _overlay().refresh()


# ---------------------------------------------------------------------------
# Path → extractor map
# ---------------------------------------------------------------------------
#
# Each row: (path-prefix relative to $HOME, extractor-name, decision).
# First match wins. Ordering matters — put the more specific prefixes
# first so e.g. `Library/Messages/Attachments` doesn't shadow
# `Library/Messages`.
#
# DECISION default for known paths is PROMOTE — when something changes
# in a watched directory we know about, run that source's extractor.
# A few are DEFER (e.g. large iCloud Drive walks shouldn't fire on
# every file touch; better to batch).

HOME = Path(os.path.expanduser("~"))


WATCHED_PATHS: list[tuple[str, str, Decision]] = [
    # iMessage chat DB + WAL + SHM files
    ("Library/Messages/chat.db",                          "imessage_enrich", Decision.PROMOTE),
    ("Library/Messages",                                  "imessage_enrich", Decision.PROMOTE),

    # Mail — large, batches well
    ("Library/Mail",                                      "mail_enrich",     Decision.DEFER),

    # Voice memos — promote immediately; new memo is high signal
    ("Library/Application Support/com.apple.voicememos",  "voice_memos",     Decision.PROMOTE),
    ("Library/Group Containers/group.com.apple.VoiceMemos.shared",
                                                          "voice_memos",     Decision.PROMOTE),

    # Notes (CloudKit container — files change often, defer to batch)
    ("Library/Group Containers/group.com.apple.notes",    "notes_enrich",    Decision.DEFER),

    # Calendar
    ("Library/Calendars",                                 "calendar",        Decision.PROMOTE),

    # Reminders
    ("Library/Group Containers/group.com.apple.reminders",
                                                          "reminders",       Decision.PROMOTE),

    # Photos library — huge; defer everything to batch
    ("Pictures/Photos Library.photoslibrary",             "photos",          Decision.DEFER),

    # Safari history + bookmarks
    ("Library/Safari/History.db",                         "safari",          Decision.PROMOTE),
    ("Library/Safari/Bookmarks.plist",                    "bookmarks",       Decision.PROMOTE),

    # Chrome
    ("Library/Application Support/Google/Chrome/Default/History",
                                                          "chrome",          Decision.PROMOTE),

    # Shell history
    (".zsh_history",                                      "shell",           Decision.PROMOTE),
    (".bash_history",                                     "shell",           Decision.PROMOTE),

    # iCloud Drive — defer, batches well
    ("Library/Mobile Documents",                          "icloud_drive",    Decision.DEFER),

    # Wallet passes
    ("Library/Passes",                                    "wallet",          Decision.PROMOTE),

    # Locations (Significant Locations) — huge defer
    ("Library/Caches/com.apple.routined",                 "locations",       Decision.DEFER),

    # Notifications
    ("Library/DoNotDisturb",                              "notifications",   Decision.DEFER),
]


# ---------------------------------------------------------------------------
# Junk patterns — events to DROP without ever running classifier logic.
# These appear constantly under HOME and would otherwise burn cycles.
# ---------------------------------------------------------------------------

JUNK_SUFFIXES = (
    ".DS_Store",
    ".tmp",
    ".swp",
    ".lock",
    "~",                # editor backup files
)

JUNK_DIR_SEGMENTS = (
    "/.Trash/",
    "/Library/Caches/com.apple.LaunchServices",
    "/Library/Caches/CloudKit",
    "/Library/Caches/Metadata/CoreSpotlight",
    "/Library/Caches/com.apple.iCloudHelper",
    "/Library/Caches/com.apple.bird",
    "/.git/",           # git internals fire constantly
    "/node_modules/",
    "/__pycache__/",
    "/.venv/",
    "/target/",         # rust builds
    "/build/",
    "/dist/",
)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def classify(event: Event) -> Classified:
    """Apply the rules. Always returns a Classified; never raises."""

    # User-feedback overlay — pmc remember/forget overrides everything.
    if event.source == Source.FS:
        forced = _overlay().override_for(event.path)
        if forced is not None:
            return Classified(
                event=event,
                decision=forced,
                reason="user feedback override",
            )

    # Junk filter — fast path, drops 90% of FS noise
    if event.source == Source.FS:
        if _is_junk_path(event.path):
            return Classified(
                event=event,
                decision=Decision.DROP,
                reason="junk path (build artifact / cache / dotfile)",
            )

    # FS events on known data-source paths → use the path map
    if event.source == Source.FS:
        match = _match_watched_path(event.path)
        if match is not None:
            extractor, decision = match
            return Classified(
                event=event,
                decision=decision,
                reason=f"matched {extractor} path",
                extractor=extractor,
            )

    # SQLite WAL events — the source already knows which DB it's for
    if event.source == Source.SQLITE:
        extractor = event.extra.get("extractor")
        if extractor:
            return Classified(
                event=event,
                decision=Decision.PROMOTE,
                reason=f"sqlite WAL update → {extractor}",
                extractor=extractor,
            )

    # Distributed notifications — the source maps notif name → extractor
    if event.source == Source.DISTNOTIF:
        extractor = event.extra.get("extractor")
        if extractor:
            return Classified(
                event=event,
                decision=Decision.PROMOTE,
                reason=f"distributed notification {event.path}",
                extractor=extractor,
            )

    # Shell preexec events — promote any command for now
    if event.source == Source.SHELL:
        return Classified(
            event=event,
            decision=Decision.PROMOTE,
            reason="shell preexec",
            extractor="shell",
        )

    # Unknown FS path — be conservative. If it's under ~/Library/ but
    # didn't match any watched path, it's almost certainly an app's
    # private state we don't care about. Drop. If it's elsewhere under
    # HOME (Documents, Desktop, code repos, etc.) it might be real
    # user content — defer to the LLM tier.
    if event.source == Source.FS:
        rel = event.path.replace(str(HOME), "", 1)
        if rel.startswith("/Library/"):
            return Classified(
                event=event,
                decision=Decision.DROP,
                reason="unknown ~/Library/ subdir (app private state)",
            )
    return Classified(
        event=event,
        decision=Decision.DEFER,
        reason="no rule matched — queueing for LLM classifier",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_junk_path(p: str) -> bool:
    if any(seg in p for seg in JUNK_DIR_SEGMENTS):
        return True
    return any(p.endswith(suf) for suf in JUNK_SUFFIXES)


def _match_watched_path(p: str) -> tuple[str, Decision] | None:
    """First-match wins over WATCHED_PATHS, checking each rule's prefix
    expanded under $HOME."""
    for rel, extractor, decision in WATCHED_PATHS:
        full = str(HOME / rel)
        if p == full or p.startswith(full + "/") or p.startswith(full):
            return (extractor, decision)
    return None


__all__ = ["classify"]
