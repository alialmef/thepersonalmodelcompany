"""Build Episodes from the existing graph + raw substrate.

The Rust extractors already produced:

    <storage>/users/<uid>/raw/{imessage,notes,mail,...}.jsonl
    <storage>/users/<uid>/graph/{people,events,places,...}.jsonl

This module reads those and creates Episode rows in the recall.db that
the consolidation worker can then enrich with Claude.

Strategy per source:

  iMessage   — group by (handle, day) → one Episode per conversation
               cluster per thread per day with enough message density
  Notes      — one Episode per note authored (already a discrete unit)
  Calendar   — one Episode per event with attendees / location resolved
  Photos     — already clustered by the photos extractor; we pass
               those Event rows through as Episodes
  Mail       — group by (correspondent, day) → one Episode per exchange
  Web        — one Episode per (domain, day) with >=3 visits

We never include raw content directly in the Episode — only pointers
back to the source jsonl/database. The consolidation worker re-fetches
content via `raw_text_lookup`.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pmc.memory.recall.schema import Episode, EpisodeKind
from pmc.memory.recall.store import RecallStore


def storage_paths(user_storage_root: Path | str, user_id: str) -> dict[str, Path]:
    root = Path(user_storage_root) / "users" / user_id
    return {
        "raw":    root / "raw",
        "graph":  root / "graph",
        "recall": root / "recall.db",
        "root":   root,
    }


def build_episodes(user_storage_root: Path | str, user_id: str) -> dict[str, int]:
    """Produce Episodes for every supported source. Returns counts."""
    paths = storage_paths(user_storage_root, user_id)
    store = RecallStore(paths["recall"])
    counts: dict[str, int] = {}

    counts["imessage"]  = _from_imessage(store, paths["raw"])
    counts["notes"]     = _from_notes(store, paths["raw"])
    counts["calendar"]  = _from_calendar_graph(store, paths["graph"])
    counts["photos"]    = _from_photos_graph(store, paths["graph"])
    counts["mail"]      = _from_mail(store, paths["raw"])
    counts["web"]       = _from_web_graph(store, paths["graph"])

    store.close()
    return counts


# ----------------------------------------------------------------------
# iMessage — group by (handle, day)
# ----------------------------------------------------------------------


def _from_imessage(store: RecallStore, raw_dir: Path) -> int:
    f = raw_dir / "imessage.jsonl"
    if not f.is_file():
        return 0
    by_thread_day: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = row.get("timestamp")
            thread = row.get("thread_id") or row.get("author_identifier") or "unknown"
            if not ts:
                continue
            day = ts[:10]
            by_thread_day[(thread, day)].append(row)

    now = datetime.now(timezone.utc)
    count = 0
    for (thread, day), msgs in by_thread_day.items():
        if len(msgs) < 2:
            continue
        msgs_sorted = sorted(msgs, key=lambda m: m.get("timestamp", ""))
        first_ts = _parse_iso(msgs_sorted[0].get("timestamp"))
        last_ts = _parse_iso(msgs_sorted[-1].get("timestamp"))
        if not first_ts:
            continue
        ep_id = _stable("imessage", thread, day)
        ep = Episode(
            id=ep_id,
            kind=EpisodeKind.conversation,
            time_start=first_ts,
            time_end=last_ts,
            place_id=None,
            participant_ids=[thread],
            raw_source="imessage",
            raw_pointers=[{
                "source": "imessage",
                "thread": thread,
                "day": day,
                "n_messages": len(msgs_sorted),
                "first_source_id": msgs_sorted[0].get("source_id"),
                "last_source_id": msgs_sorted[-1].get("source_id"),
            }],
            ingestion_time=now,
        )
        store.upsert_episode(ep, raw_text_for_fts="")
        count += 1
    return count


# ----------------------------------------------------------------------
# Notes — one Episode per note
# ----------------------------------------------------------------------


def _from_notes(store: RecallStore, raw_dir: Path) -> int:
    f = raw_dir / "notes.jsonl"
    if not f.is_file():
        return 0
    now = datetime.now(timezone.utc)
    count = 0
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = _parse_iso(row.get("timestamp")) or now
            ep_id = _stable("notes", row.get("source_id") or "")
            ep = Episode(
                id=ep_id,
                kind=EpisodeKind.note_authored,
                time_start=ts,
                time_end=ts,
                place_id=None,
                participant_ids=[],
                raw_source="notes",
                raw_pointers=[{
                    "source": "notes",
                    "source_id": row.get("source_id"),
                    "subject": row.get("subject"),
                }],
                ingestion_time=now,
            )
            store.upsert_episode(ep, raw_text_for_fts="")
            count += 1
    return count


# ----------------------------------------------------------------------
# Calendar — one Episode per Event row in graph
# ----------------------------------------------------------------------


def _from_calendar_graph(store: RecallStore, graph_dir: Path) -> int:
    return _events_from_source(store, graph_dir, source_filter="calendar",
                                kind=EpisodeKind.calendar_event)


def _from_photos_graph(store: RecallStore, graph_dir: Path) -> int:
    return _events_from_source(store, graph_dir, source_filter="photos",
                                kind=EpisodeKind.photo_cluster)


def _events_from_source(store: RecallStore, graph_dir: Path, source_filter: str,
                         kind: EpisodeKind) -> int:
    f = graph_dir / "events.jsonl"
    if not f.is_file():
        return 0
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    # Calendar.app preloads recurring holidays into the future. Those
    # aren't lived events; they pollute working memory and retrieval.
    future_cutoff = now + timedelta(days=60)
    count = 0
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            sources = ev.get("sources") or []
            if source_filter not in sources:
                continue
            start = _parse_iso(ev.get("start"))
            if not start:
                continue
            if start > future_cutoff:
                continue
            end = _parse_iso(ev.get("end"))
            ep_id = _stable(kind.value, ev.get("id") or "")
            ep = Episode(
                id=ep_id,
                kind=kind,
                time_start=start,
                time_end=end,
                place_id=ev.get("place_id"),
                participant_ids=list(ev.get("attendee_ids") or []),
                raw_source=source_filter,
                raw_pointers=[{
                    "source": source_filter,
                    "event_id": ev.get("id"),
                    "title": ev.get("title"),
                }],
                ingestion_time=now,
            )
            store.upsert_episode(ep, raw_text_for_fts=ev.get("title", ""))
            count += 1
    return count


# ----------------------------------------------------------------------
# Mail — group by (correspondent, day). Only headers available from
# raw, no body. So Episodes are header-only; consolidation can later
# fold in subject lines.
# ----------------------------------------------------------------------


def _from_mail(store: RecallStore, raw_dir: Path) -> int:
    f = raw_dir / "mail.jsonl"
    if not f.is_file():
        # mail_enrich extractor writes through graph not raw; nothing
        # to do here unless the user uploaded sent-mail bodies.
        return 0
    now = datetime.now(timezone.utc)
    by_corr_day: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = row.get("timestamp")
            if not ts:
                continue
            corr = row.get("author_identifier") or row.get("thread_id") or "unknown"
            day = ts[:10]
            by_corr_day[(corr, day)].append(row)
    count = 0
    for (corr, day), msgs in by_corr_day.items():
        if not msgs:
            continue
        msgs_sorted = sorted(msgs, key=lambda m: m.get("timestamp", ""))
        first = _parse_iso(msgs_sorted[0].get("timestamp"))
        last  = _parse_iso(msgs_sorted[-1].get("timestamp"))
        if not first:
            continue
        ep = Episode(
            id=_stable("mail", corr, day),
            kind=EpisodeKind.mail_exchange,
            time_start=first,
            time_end=last,
            place_id=None,
            participant_ids=[corr],
            raw_source="mail",
            raw_pointers=[{"source": "mail", "corr": corr, "day": day, "n": len(msgs_sorted)}],
            ingestion_time=now,
        )
        store.upsert_episode(ep, raw_text_for_fts="")
        count += 1
    return count


# ----------------------------------------------------------------------
# Web — one Episode per (domain, recent peak)
# ----------------------------------------------------------------------


def _from_web_graph(store: RecallStore, graph_dir: Path) -> int:
    f = graph_dir / "web.jsonl"
    if not f.is_file():
        return 0
    now = datetime.now(timezone.utc)
    count = 0
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line)
            except Exception:
                continue
            if (w.get("visits_30d") or 0) < 3:
                continue
            last = _parse_iso(w.get("last_visit"))
            if not last:
                continue
            ep = Episode(
                id=_stable("web", w.get("domain") or ""),
                kind=EpisodeKind.web_session,
                time_start=last,
                time_end=last,
                place_id=None,
                participant_ids=[],
                raw_source="safari",
                raw_pointers=[{
                    "source": "safari",
                    "domain": w.get("domain"),
                    "visits_30d": w.get("visits_30d"),
                    "category": w.get("category"),
                }],
                ingestion_time=now,
            )
            store.upsert_episode(ep, raw_text_for_fts=w.get("domain", ""))
            count += 1
    return count


# ----------------------------------------------------------------------


def _stable(*parts: str) -> str:
    seed = "|".join(str(p) for p in parts)
    return str(uuid.uuid5(uuid.UUID(int=0), seed))


def _parse_iso(s):
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def raw_text_for_episode(user_storage_root: Path | str, user_id: str, ep: Episode) -> str:
    """Hydrate the raw content for one Episode at consolidation time.

    Reads the original raw jsonl on demand. Kept separate from migration
    so we don't bloat the recall.db with duplicated text.
    """
    paths = storage_paths(user_storage_root, user_id)
    src = ep.raw_source
    if src == "imessage":
        return _hydrate_imessage(paths["raw"] / "imessage.jsonl", ep)
    if src == "notes":
        return _hydrate_notes(paths["raw"] / "notes.jsonl", ep)
    if src in ("calendar", "photos"):
        # The graph events.jsonl is the source of truth — pull title +
        # whatever metadata we wrote for that event id.
        return _hydrate_event(paths["graph"] / "events.jsonl", ep)
    if src == "mail":
        return _hydrate_mail(paths["raw"] / "mail.jsonl", ep)
    if src == "safari":
        return _hydrate_web(paths["graph"] / "web.jsonl", ep)
    return ""


def _hydrate_imessage(f: Path, ep: Episode) -> str:
    if not f.is_file() or not ep.raw_pointers:
        return ""
    thread = ep.raw_pointers[0].get("thread")
    day = ep.raw_pointers[0].get("day")
    if not thread or not day:
        return ""
    lines: list[str] = []
    with f.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("thread_id") != thread and row.get("author_identifier") != thread:
                continue
            ts = (row.get("timestamp") or "")[:10]
            if ts != day:
                continue
            who = "user" if row.get("is_user") else (row.get("author_identifier") or "them")
            text = (row.get("content") or "").strip()
            if text:
                lines.append(f"{who}: {text}")
    return "\n".join(lines[:200])


def _hydrate_notes(f: Path, ep: Episode) -> str:
    if not f.is_file() or not ep.raw_pointers:
        return ""
    target = ep.raw_pointers[0].get("source_id")
    with f.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("source_id") == target:
                return (row.get("content") or "")[:6000]
    return ""


def _hydrate_event(f: Path, ep: Episode) -> str:
    if not f.is_file() or not ep.raw_pointers:
        return ""
    target = ep.raw_pointers[0].get("event_id")
    with f.open() as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("id") == target:
                bits = []
                bits.append(f"Title: {ev.get('title', '')}")
                if ev.get("kind"):
                    bits.append(f"Kind: {ev['kind']}")
                if ev.get("notes"):
                    bits.append(f"Notes: {ev['notes']}")
                if ev.get("attendee_ids"):
                    bits.append(f"Attendees: {len(ev['attendee_ids'])}")
                return "\n".join(bits)
    return ""


def _hydrate_mail(f: Path, ep: Episode) -> str:
    if not f.is_file() or not ep.raw_pointers:
        return ""
    corr = ep.raw_pointers[0].get("corr")
    day = ep.raw_pointers[0].get("day")
    lines: list[str] = []
    with f.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = (row.get("timestamp") or "")[:10]
            if (row.get("author_identifier") != corr and row.get("thread_id") != corr) or ts != day:
                continue
            subj = row.get("subject") or ""
            text = (row.get("content") or "").strip()
            lines.append(f"Subject: {subj}\n{text}")
    return "\n\n".join(lines[:50])


def _hydrate_web(f: Path, ep: Episode) -> str:
    if not f.is_file() or not ep.raw_pointers:
        return ""
    target = ep.raw_pointers[0].get("domain")
    with f.open() as fh:
        for line in fh:
            try:
                w = json.loads(line)
            except Exception:
                continue
            if w.get("domain") == target:
                return (
                    f"Domain: {w.get('domain')}\n"
                    f"Category: {w.get('category')}\n"
                    f"Visits last 30d: {w.get('visits_30d')}\n"
                    f"Visits last 180d: {w.get('visits_180d')}"
                )
    return ""
