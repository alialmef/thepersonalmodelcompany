"""Time model — how the user actually spends their time.

The bedrock of the portrait. A person is how they spend their time;
this module turns the graph's timestamped events into a structured
24-hour day, weekly rhythm, and monthly aggregate.

Primary input: `app_usage.jsonl` rows, which carry `by_hour` (24-element
array) and `by_dow` (7-element array) from Apple Screen Time. Apple
stores these in *local time*, so no TZ conversion is needed for app
data — `by_hour[8]` means 8am the user's time.

Augmented with:
  - git log timestamps per repo (when coding actually happens, by clock)
  - iMessage send timestamps (when comms happens) — via chat.db direct read
  - web visit aggregates by domain (what the reading is about)
  - voice memo recording timestamps (when memos happen)

Output: portrait/time_day.md, portrait/time_week.md, portrait/time_month.md,
and the `TimeModel` dataclass which feeds `compose_time_md` for the
top-level synthesis.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from pmc.consolidator.categorize import (
    app_label,
    categorize_app,
    categorize_domain,
)
from pmc.storage.graph_store import GraphStore


log = logging.getLogger("pmc.consolidator.time_model")


def _detect_local_tz() -> ZoneInfo:
    """Best-effort detection of the user's local timezone.

    Order:
      1. PMC_LOCAL_TZ env var (explicit override)
      2. macOS `systemsetup -gettimezone` output
      3. UTC fallback
    """
    env = os.environ.get("PMC_LOCAL_TZ")
    if env:
        try:
            return ZoneInfo(env)
        except Exception:  # noqa: BLE001
            pass
    # macOS — read from /etc/localtime symlink
    try:
        link = os.readlink("/etc/localtime")
        # e.g. /var/db/timezone/zoneinfo/America/Los_Angeles
        if "zoneinfo/" in link:
            name = link.split("zoneinfo/", 1)[1]
            return ZoneInfo(name)
    except OSError:
        pass
    # Last resort
    try:
        from datetime import datetime as _dt
        local_name = _dt.now().astimezone().tzinfo
        if local_name is not None:
            return ZoneInfo(str(local_name))
    except Exception:  # noqa: BLE001
        pass
    return ZoneInfo("UTC")


LOCAL_TZ = _detect_local_tz()

# Categories we report, in display order.
CATEGORIES = (
    "coding", "comms", "reading", "ai-assist",
    "creating", "video-editing", "media",
    "navigation", "system", "other",
)

DOW_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class HourBin:
    """One hour of a typical day (aggregated across 30-180 days)."""
    hour: int                                # 0-23 local time
    minutes: int                             # total minutes in this hour
    by_category: dict[str, int] = field(default_factory=dict)


@dataclass
class DayShape:
    hours: list[HourBin]                     # 24 entries
    total_minutes: int
    by_category: dict[str, int]              # category → total minutes
    peak_hour: int                           # the busiest hour
    sleep_window: Optional[tuple[int, int]]  # (start_hour, end_hour) of longest gap


@dataclass
class WeekShape:
    by_dow: list[int]                        # 7 entries — minutes per weekday
    by_dow_category: dict[str, list[int]]    # category → 7-entry list
    weekday_avg_min: int                     # mean weekday minutes
    weekend_avg_min: int                     # mean weekend minutes


@dataclass
class CommitsByHour:
    """Coding-time-of-day from git log, not app_usage."""
    repo_name: str
    by_hour: list[int]                       # 24 entries — commit count
    total_30d: int


@dataclass
class WebReading:
    """What the user's been reading, by subcategory of domain."""
    by_subcat: dict[str, int]                # subcat → visits_30d (or estimate)
    top_domains: list[tuple[str, int, str]]  # (domain, visits, subcat)


@dataclass
class DayLog:
    """One specific day's reconstructed activity from precise timestamps."""
    date: str                                # YYYY-MM-DD in local TZ
    commits: int = 0
    commit_repos: dict[str, int] = field(default_factory=dict)
    messages_sent_estimate: int = 0          # iMessage rows on this day
    photo_files: int = 0
    first_event_hour: Optional[int] = None   # earliest hour with activity
    last_event_hour: Optional[int] = None    # latest hour with activity
    active_hours: int = 0                    # distinct hours with any event


@dataclass
class YearScale:
    """Long-horizon (last 365 days) aggregates, per source. Used by the
    picture composer to surface 'over the last year you...' framing."""
    commits_by_repo_year: dict[str, int] = field(default_factory=dict)
    commits_by_month: dict[str, int] = field(default_factory=dict)  # "YYYY-MM" → count
    messages_total_year: Optional[int] = None
    months_with_activity: int = 0


@dataclass
class TimeModel:
    """The complete time picture. Composed into prose by characterize.py."""
    day: DayShape
    week: WeekShape
    commits: list[CommitsByHour]
    web: WebReading
    total_minutes_30d: int
    rhythm_label: str                        # "night owl" / "early bird" / "split"
    sleep_summary: str                       # "likely sleep window 3am–10am"
    messages_by_hour: Optional[list[int]] = None
    photos_by_hour: Optional[list[int]] = None
    creations_by_hour: Optional[list[int]] = None
    narrative_by_hour: list[str] = field(default_factory=list)
    recent_days: list[DayLog] = field(default_factory=list)
    year_scale: Optional[YearScale] = None
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_time_model(store: GraphStore, *, user_id: str) -> TimeModel:
    """Compose the full picture from the graph + augmenting sources."""
    apps = list(store.iter_entities(user_id, "app"))
    day = _build_day_shape(apps)
    week = _build_week_shape(apps)

    repos = list(store.iter_entities(user_id, "repo"))
    commits = [_build_commits_by_hour(r) for r in repos]
    commits = [c for c in commits if c.total_30d > 0]

    web_entities = list(store.iter_entities(user_id, "web"))
    web = _build_web_reading(web_entities)

    files = list(store.iter_entities(user_id, "file"))
    photos_h = _files_by_hour(files, kinds=("image", "video"))
    creations_h = _files_by_hour(
        files, kinds=("note", "document", "text", "code"),
    )

    msgs_h = messages_by_hour(days=30)

    recent_days = _build_recent_days(repos, files, days=7)

    year_scale = _build_year_scale(repos)

    narrative = _build_narrative(day)

    total = sum(int(a.get("minutes_30d") or 0) for a in apps)
    rhythm = _rhythm_label(day)
    sleep = _sleep_summary(day)

    return TimeModel(
        day=day,
        week=week,
        commits=commits,
        web=web,
        total_minutes_30d=total,
        rhythm_label=rhythm,
        sleep_summary=sleep,
        messages_by_hour=msgs_h,
        photos_by_hour=photos_h if any(photos_h) else None,
        creations_by_hour=creations_h if any(creations_h) else None,
        narrative_by_hour=narrative,
        recent_days=recent_days,
        year_scale=year_scale,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Year-scale aggregator
# ---------------------------------------------------------------------------


def _build_year_scale(repos: list[dict[str, Any]]) -> YearScale:
    """Walk git log over the last 365 days for each repo. Cheap enough
    to do every pass."""
    by_repo: dict[str, int] = {}
    by_month: dict[str, int] = {}
    for r in repos:
        path = r.get("path") or ""
        name = r.get("name") or "?"
        p = Path(path).expanduser()
        if not (p / ".git").is_dir():
            continue
        try:
            res = subprocess.run(
                ["git", "-C", str(p), "log",
                 "--since=365.days", "--pretty=%aI", "--all"],
                capture_output=True, text=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if res.returncode != 0:
            continue
        count = 0
        for line in res.stdout.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                dt = datetime.fromisoformat(s).astimezone(LOCAL_TZ)
            except ValueError:
                continue
            count += 1
            month_key = dt.strftime("%Y-%m")
            by_month[month_key] = by_month.get(month_key, 0) + 1
        if count:
            by_repo[name] = count
    return YearScale(
        commits_by_repo_year=by_repo,
        commits_by_month=by_month,
        months_with_activity=len(by_month),
    )


# ---------------------------------------------------------------------------
# Per-day timeline reconstruction (last N days, from precise timestamps)
# ---------------------------------------------------------------------------


def _build_recent_days(
    repos: list[dict[str, Any]],
    files: list[dict[str, Any]],
    *,
    days: int = 7,
) -> list[DayLog]:
    """For each of the last `days` days, count events from precise
    sources (git log, message timestamps, photo file modifications).
    Returns newest-first."""
    now_local = datetime.now(timezone.utc).astimezone(LOCAL_TZ)
    today = now_local.date()
    day_keys = []
    day_logs: dict[str, DayLog] = {}
    for i in range(days):
        d = today.fromordinal(today.toordinal() - i)
        key = d.isoformat()
        day_keys.append(key)
        day_logs[key] = DayLog(date=key)

    cutoff = (now_local - timedelta(days=days)).timestamp()

    # 1. Git commits per day per repo
    for repo in repos:
        path = repo.get("path") or ""
        name = repo.get("name") or "?"
        if not path:
            continue
        p = Path(path).expanduser()
        if not (p / ".git").is_dir():
            continue
        try:
            res = subprocess.run(
                ["git", "-C", str(p), "log",
                 f"--since={days}.days", "--pretty=%aI", "--all"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if res.returncode != 0:
            continue
        for line in res.stdout.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                dt = datetime.fromisoformat(s).astimezone(LOCAL_TZ)
            except ValueError:
                continue
            key = dt.date().isoformat()
            if key not in day_logs:
                continue
            log_row = day_logs[key]
            log_row.commits += 1
            log_row.commit_repos[name] = log_row.commit_repos.get(name, 0) + 1
            log_row.active_hours = max(log_row.active_hours, 0)
            log_row.first_event_hour = (
                dt.hour if log_row.first_event_hour is None
                else min(log_row.first_event_hour, dt.hour)
            )
            log_row.last_event_hour = (
                dt.hour if log_row.last_event_hour is None
                else max(log_row.last_event_hour, dt.hour)
            )

    # 2. iMessage timestamps per day (best effort)
    chat_db = Path("~/Library/Messages/chat.db").expanduser()
    if chat_db.is_file():
        try:
            conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True, timeout=1.0)
            try:
                epoch_offset = 978307200  # 2001-01-01 - 1970-01-01 in seconds
                for (apple_date,) in conn.execute(
                    "SELECT date FROM message ORDER BY ROWID DESC LIMIT 200000"
                ):
                    if apple_date is None:
                        continue
                    ts = apple_date / 1e9 if apple_date > 1e12 else apple_date
                    unix = ts + epoch_offset
                    if unix < cutoff:
                        break
                    dt = datetime.fromtimestamp(unix, tz=timezone.utc).astimezone(LOCAL_TZ)
                    key = dt.date().isoformat()
                    if key not in day_logs:
                        continue
                    log_row = day_logs[key]
                    log_row.messages_sent_estimate += 1
                    log_row.first_event_hour = (
                        dt.hour if log_row.first_event_hour is None
                        else min(log_row.first_event_hour, dt.hour)
                    )
                    log_row.last_event_hour = (
                        dt.hour if log_row.last_event_hour is None
                        else max(log_row.last_event_hour, dt.hour)
                    )
            finally:
                conn.close()
        except sqlite3.Error:
            pass

    # 3. Photo / video files modified per day
    for f in files:
        kind = f.get("kind") or ""
        ext = (f.get("extension") or "").lower()
        is_photo = (
            kind in ("image", "video")
            or ext in ("jpg", "jpeg", "png", "heic", "mov", "mp4", "m4v")
        )
        if not is_photo:
            continue
        ts_str = f.get("modified")
        if not ts_str:
            continue
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(LOCAL_TZ)
        except (ValueError, AttributeError):
            continue
        if dt.timestamp() < cutoff:
            continue
        key = dt.date().isoformat()
        if key not in day_logs:
            continue
        day_logs[key].photo_files += 1

    # 4. Compute active_hours per day — number of distinct hours with any
    #    of {commit, message, photo}. Simple proxy.
    for key, log_row in day_logs.items():
        if log_row.first_event_hour is None:
            continue
        span = log_row.last_event_hour - log_row.first_event_hour + 1
        log_row.active_hours = max(1, span)

    return [day_logs[k] for k in day_keys]


# ---------------------------------------------------------------------------
# File-creation timing (photos, notes, docs)
# ---------------------------------------------------------------------------


def _files_by_hour(
    files: list[dict[str, Any]], *, kinds: tuple[str, ...],
) -> list[int]:
    """Bin file modification timestamps by hour-of-day. Files matching
    `kinds` only (e.g. image+video for photo activity)."""
    by_hour = [0] * 24
    kindset = set(kinds)
    cutoff = datetime.now(timezone.utc).timestamp() - 30 * 86400
    for f in files:
        kind = f.get("kind") or ""
        if kind not in kindset:
            # Also accept by file extension
            ext = (f.get("extension") or "").lower()
            if ext in ("jpg", "jpeg", "png", "heic", "heif", "gif", "tiff",
                       "mov", "mp4", "m4v", "webp"):
                if "image" not in kindset and "video" not in kindset:
                    continue
            elif ext in ("md", "txt", "rtf", "docx", "pdf"):
                if "document" not in kindset and "note" not in kindset and "text" not in kindset:
                    continue
            elif ext in ("py", "ts", "tsx", "js", "rs", "go", "swift"):
                if "code" not in kindset:
                    continue
            else:
                continue
        ts_str = f.get("modified")
        if not ts_str:
            continue
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if dt.timestamp() < cutoff:
            continue
        local = dt.astimezone(LOCAL_TZ)
        by_hour[local.hour] += 1
    return by_hour


# ---------------------------------------------------------------------------
# Narrative — what's the dominant category in each hour
# ---------------------------------------------------------------------------


def _build_narrative(day: DayShape) -> list[str]:
    """For each hour 0-23, return a short label like 'coding' / 'reading' /
    'idle'. The dominant category, or 'idle' if total minutes are below
    a threshold."""
    out: list[str] = []
    if not day.hours:
        return ["idle"] * 24
    median_active = sorted(
        [h.minutes for h in day.hours if h.minutes > 0]
    )
    idle_threshold = max(2, (median_active[len(median_active) // 4]
                              if median_active else 0))
    for h in day.hours:
        if h.minutes < idle_threshold:
            out.append("idle")
            continue
        if not h.by_category:
            out.append("active")
            continue
        top_cat = max(h.by_category, key=lambda c: h.by_category[c])
        out.append(top_cat)
    return out


# ---------------------------------------------------------------------------
# Day shape — by_hour aggregated across apps, weighted by category
# ---------------------------------------------------------------------------


def _build_day_shape(apps: list[dict[str, Any]]) -> DayShape:
    """Aggregate the per-app by_hour arrays into one typical-day shape."""
    hours: list[HourBin] = [HourBin(hour=h, minutes=0) for h in range(24)]
    by_category_total: dict[str, int] = defaultdict(int)

    for a in apps:
        bh = a.get("by_hour") or []
        if not isinstance(bh, list) or len(bh) != 24:
            continue
        cat = categorize_app(
            bundle_id=a.get("bundle_id"),
            display_name=a.get("display_name"),
            fallback=a.get("category"),
        )
        mins_30d = int(a.get("minutes_30d") or 0)
        # by_hour is reported by Apple as the count over the full 180-day
        # window (or some Apple-defined window). We rescale to 30d using
        # the ratio between its sum and minutes_30d, so the numbers are
        # comparable across apps and the daily totals match the per-app
        # minutes_30d we already display.
        bh_sum = sum(v for v in bh if isinstance(v, (int, float)))
        if bh_sum <= 0 or mins_30d <= 0:
            continue
        scale = mins_30d / bh_sum
        for h, v in enumerate(bh):
            if not isinstance(v, (int, float)) or v <= 0:
                continue
            scaled = v * scale
            hours[h].minutes += int(round(scaled))
            hours[h].by_category[cat] = hours[h].by_category.get(cat, 0) + int(round(scaled))
            by_category_total[cat] += int(round(scaled))

    peak = max(range(24), key=lambda h: hours[h].minutes)
    sleep = _find_sleep_window(hours)
    total_mins = sum(h.minutes for h in hours)

    return DayShape(
        hours=hours,
        total_minutes=total_mins,
        by_category=dict(by_category_total),
        peak_hour=peak,
        sleep_window=sleep,
    )


def _find_sleep_window(hours: list[HourBin]) -> Optional[tuple[int, int]]:
    """Find the longest run of low-activity hours wrapping around midnight.
    Returns `(start_hour, end_hour)` inclusive, or None if no clear gap."""
    minutes = [h.minutes for h in hours]
    if not minutes:
        return None
    median = sorted(minutes)[len(minutes) // 2]
    threshold = max(2, median // 4)  # quiet = below 25% of median, min 2 min

    # Look across two days (circular) so a 2am-9am gap is one run.
    doubled = minutes + minutes
    best_start = -1
    best_len = 0
    run_start = -1
    run_len = 0
    for i, m in enumerate(doubled):
        if m <= threshold:
            if run_start < 0:
                run_start = i
                run_len = 0
            run_len += 1
            if run_len > best_len:
                best_len = run_len
                best_start = run_start
        else:
            run_start = -1
            run_len = 0
    if best_len < 4 or best_len > 14:  # too short = noise; too long = no data
        return None
    start_h = best_start % 24
    end_h = (best_start + best_len - 1) % 24
    return (start_h, end_h)


# ---------------------------------------------------------------------------
# Week shape — by_dow aggregated across apps
# ---------------------------------------------------------------------------


def _build_week_shape(apps: list[dict[str, Any]]) -> WeekShape:
    """Aggregate by_dow across apps. Apple's by_dow is Sunday-first; we
    re-order to Monday-first to match DOW_LABELS."""
    by_dow_total = [0] * 7
    by_dow_cat: dict[str, list[int]] = defaultdict(lambda: [0] * 7)

    for a in apps:
        bd = a.get("by_dow") or []
        if not isinstance(bd, list) or len(bd) != 7:
            continue
        mins_30d = int(a.get("minutes_30d") or 0)
        bd_sum = sum(v for v in bd if isinstance(v, (int, float)))
        if bd_sum <= 0 or mins_30d <= 0:
            continue
        scale = mins_30d / bd_sum
        cat = categorize_app(
            bundle_id=a.get("bundle_id"),
            display_name=a.get("display_name"),
            fallback=a.get("category"),
        )
        # Reorder Sun-first → Mon-first.
        # Apple's by_dow[0] = Sunday, by_dow[1] = Monday, etc.
        # We want Mon=0..Sun=6, so dow_mon_idx = (sun_idx + 6) % 7
        for sun_idx, v in enumerate(bd):
            if not isinstance(v, (int, float)) or v <= 0:
                continue
            mon_idx = (sun_idx + 6) % 7
            scaled = int(round(v * scale))
            by_dow_total[mon_idx] += scaled
            by_dow_cat[cat][mon_idx] += scaled

    weekday = by_dow_total[:5]
    weekend = by_dow_total[5:]
    weekday_avg = sum(weekday) // 5 if weekday else 0
    weekend_avg = sum(weekend) // 2 if weekend else 0

    return WeekShape(
        by_dow=by_dow_total,
        by_dow_category=dict(by_dow_cat),
        weekday_avg_min=weekday_avg,
        weekend_avg_min=weekend_avg,
    )


# ---------------------------------------------------------------------------
# Commits — actual git log, more precise than app_usage
# ---------------------------------------------------------------------------


def _build_commits_by_hour(repo: dict[str, Any]) -> CommitsByHour:
    """Pull commit timestamps from the repo's git log over the last 30d
    and bin by local hour-of-day."""
    name = repo.get("name") or "(repo)"
    path = repo.get("path") or ""
    by_hour = [0] * 24
    total = 0
    if not path:
        return CommitsByHour(repo_name=name, by_hour=by_hour, total_30d=0)
    p = Path(path).expanduser()
    if not (p / ".git").is_dir():
        return CommitsByHour(repo_name=name, by_hour=by_hour, total_30d=0)
    try:
        res = subprocess.run(
            ["git", "-C", str(p), "log", "--since=30.days",
             "--pretty=%aI", "--all"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return CommitsByHour(repo_name=name, by_hour=by_hour, total_30d=0)
    if res.returncode != 0:
        return CommitsByHour(repo_name=name, by_hour=by_hour, total_30d=0)
    for line in res.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            dt = datetime.fromisoformat(s).astimezone(LOCAL_TZ)
        except ValueError:
            continue
        by_hour[dt.hour] += 1
        total += 1
    return CommitsByHour(repo_name=name, by_hour=by_hour, total_30d=total)


# ---------------------------------------------------------------------------
# Web reading — domain → subcategory aggregate
# ---------------------------------------------------------------------------


def _build_web_reading(web_entities: list[dict[str, Any]]) -> WebReading:
    by_subcat: dict[str, int] = defaultdict(int)
    by_domain: dict[str, tuple[int, str]] = {}
    for w in web_entities:
        domain = (w.get("domain") or w.get("host") or "").lower()
        if not domain:
            continue
        v30 = int(w.get("visits_30d") or 0)
        if v30 <= 0:
            continue
        subcat = categorize_domain(domain)
        by_subcat[subcat] += v30
        prev_v, _prev_s = by_domain.get(domain, (0, subcat))
        by_domain[domain] = (prev_v + v30, subcat)

    top = sorted(
        ((d, v, s) for d, (v, s) in by_domain.items()),
        key=lambda t: t[1], reverse=True,
    )[:30]
    return WebReading(by_subcat=dict(by_subcat), top_domains=top)


# ---------------------------------------------------------------------------
# Rhythm / sleep description
# ---------------------------------------------------------------------------


def _rhythm_label(day: DayShape) -> str:
    """Classify the user's daily rhythm based on where peak activity lands."""
    if day.total_minutes == 0:
        return "unknown"
    # Morning = 5-11, Afternoon = 12-17, Evening = 18-22, Night = 23-4
    morning = sum(day.hours[h].minutes for h in range(5, 12))
    afternoon = sum(day.hours[h].minutes for h in range(12, 18))
    evening = sum(day.hours[h].minutes for h in range(18, 23))
    night = sum(day.hours[h].minutes for h in [23, 0, 1, 2, 3, 4])
    bands = {
        "morning person":  morning,
        "afternoon focus": afternoon,
        "evening worker":  evening,
        "night owl":       night,
    }
    top = max(bands, key=lambda k: bands[k])
    top_val = bands[top]
    if top_val < day.total_minutes * 0.35:
        return "spread across the day"
    return top


def _sleep_summary(day: DayShape) -> str:
    if not day.sleep_window:
        return "no clear sleep window detected"
    start, end = day.sleep_window
    return f"likely sleep window {start:02d}:00–{end:02d}:59"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_hour_bars(day: DayShape, *, width_per_bar: int = 2) -> str:
    """ASCII histogram of the typical day."""
    if day.total_minutes == 0:
        return "(no time data)"
    maxv = max(h.minutes for h in day.hours) or 1
    blocks = " ▁▂▃▄▅▆▇█"  # 9 levels including blank
    out_chars: list[str] = []
    for h in range(24):
        level = round(day.hours[h].minutes / maxv * 8)
        out_chars.append(blocks[level] * width_per_bar)
    bar = "".join(out_chars)
    # Hour ruler below
    ruler_parts: list[str] = []
    for h in range(24):
        label = f"{h:02d}" if h % 3 == 0 else "  "
        ruler_parts.append(label[:width_per_bar].ljust(width_per_bar))
    ruler = "".join(ruler_parts)
    return f"{bar}\n{ruler}"


def render_dow_bars(week: WeekShape, *, width_per_bar: int = 6) -> str:
    if not any(week.by_dow):
        return "(no week data)"
    maxv = max(week.by_dow) or 1
    lines: list[str] = []
    for i, m in enumerate(week.by_dow):
        bars = "█" * int(round(m / maxv * width_per_bar))
        lines.append(f"  {DOW_LABELS[i]} {bars:<{width_per_bar}}  {_fmt_min(m)}")
    return "\n".join(lines)


def _fmt_min(m: int) -> str:
    if m < 60:
        return f"{m}m"
    h = m / 60
    return f"{h:.1f}h"


# ---------------------------------------------------------------------------
# Markdown output (for the consolidator)
# ---------------------------------------------------------------------------


def render_time_day_md(tm: TimeModel) -> str:
    lines = ["# Typical 24-hour day (aggregated, last 30 days)", ""]
    lines.append(f"rhythm: **{tm.rhythm_label}**  ·  {tm.sleep_summary}")
    lines.append("")
    lines.append("```")
    lines.append(render_hour_bars(tm.day))
    lines.append("```")
    lines.append("")
    lines.append("## by category, total minutes per typical day's worth of activity")
    lines.append("")
    sorted_cats = sorted(tm.day.by_category.items(), key=lambda t: t[1], reverse=True)
    for cat, m in sorted_cats:
        pct = (m / max(1, tm.day.total_minutes)) * 100
        lines.append(f"- {cat:14}  {_fmt_min(m):>6}  ({pct:>4.1f}%)")
    lines.append("")
    # Commit-time-of-day overlay (when the user actually codes)
    if tm.commits:
        lines.append("## coding hours, from git log (last 30 days)")
        lines.append("")
        for c in tm.commits:
            peak = max(range(24), key=lambda h: c.by_hour[h]) if c.total_30d else 0
            count_at_peak = c.by_hour[peak] if c.total_30d else 0
            lines.append(
                f"- {c.repo_name}: {c.total_30d} commits, "
                f"peak hour {peak:02d}:00 ({count_at_peak})"
            )
        lines.append("")
    return "\n".join(lines)


def render_time_week_md(tm: TimeModel) -> str:
    lines = ["# Weekly rhythm (last ~30 days, by day-of-week)", ""]
    lines.append("```")
    lines.append(render_dow_bars(tm.week))
    lines.append("```")
    lines.append("")
    lines.append(
        f"weekday average: {_fmt_min(tm.week.weekday_avg_min)}  ·  "
        f"weekend average: {_fmt_min(tm.week.weekend_avg_min)}"
    )
    lines.append("")
    return "\n".join(lines)


def render_time_recent_md(tm: TimeModel) -> str:
    """Per-day breakdown of the last 7 days."""
    lines = ["# Last 7 days (reconstructed from precise timestamps)", ""]
    if not tm.recent_days:
        lines.append("(no per-day data)")
        return "\n".join(lines)
    for d in tm.recent_days:
        bits = [f"- **{d.date}**"]
        if d.commits:
            top_repo = max(d.commit_repos, key=d.commit_repos.get) if d.commit_repos else ""
            bits.append(f"{d.commits} commits ({top_repo})")
        if d.messages_sent_estimate:
            bits.append(f"{d.messages_sent_estimate} messages")
        if d.photo_files:
            bits.append(f"{d.photo_files} photos/video")
        if d.first_event_hour is not None:
            bits.append(
                f"active {d.first_event_hour:02d}–{d.last_event_hour:02d}"
            )
        lines.append("  " + "  ·  ".join(bits) if len(bits) > 1 else bits[0])
    return "\n".join(lines)


def render_time_month_md(tm: TimeModel) -> str:
    total_h = tm.total_minutes_30d / 60.0
    daily_avg_h = total_h / 30.0
    lines = ["# Monthly time (last 30 days)", ""]
    lines.append(
        f"**{_fmt_min(tm.total_minutes_30d)} on Mac**  ·  "
        f"~{daily_avg_h:.1f}h/day average"
    )
    lines.append("")
    lines.append("## by category")
    lines.append("")
    sorted_cats = sorted(tm.day.by_category.items(), key=lambda t: t[1], reverse=True)
    for cat, m in sorted_cats:
        pct = (m / max(1, tm.total_minutes_30d)) * 100
        lines.append(f"- {cat:14}  {_fmt_min(m):>6}  ({pct:>4.1f}%)")
    lines.append("")
    lines.append("## reading — what's the browser actually about")
    lines.append("")
    if tm.web.by_subcat:
        for subcat, v in sorted(
            tm.web.by_subcat.items(), key=lambda t: t[1], reverse=True
        )[:10]:
            lines.append(f"- {subcat:18}  {v:>4} visits")
        lines.append("")
        lines.append("top domains:")
        for d, v, s in tm.web.top_domains[:15]:
            lines.append(f"  - {d:30}  {v:>4} visits  ({s})")
    else:
        lines.append("(no web visit data with `visits_30d > 0`)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct chat.db read for message-time-of-day (optional, best effort)
# ---------------------------------------------------------------------------


def messages_by_hour(*, days: int = 30) -> Optional[list[int]]:
    """Read iMessage send/receive timestamps directly from chat.db and
    bin by local hour-of-day. Returns None if the DB isn't readable
    (e.g. FDA not granted from this process)."""
    chat_db = Path("~/Library/Messages/chat.db").expanduser()
    if not chat_db.is_file():
        return None
    by_hour = [0] * 24
    try:
        # Apple's `date` column is Mach absolute time in nanoseconds since
        # 2001-01-01. Convert with the offset.
        epoch_offset = 978307200  # seconds between 1970-01-01 and 2001-01-01
        cutoff = (datetime.now(timezone.utc).timestamp() - days * 86400)
        # Apple stores date in seconds-since-2001 multiplied by 1e9 (some
        # builds use seconds, some nanoseconds — handle both).
        conn = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True, timeout=1.0)
        try:
            for (apple_date,) in conn.execute(
                "SELECT date FROM message ORDER BY ROWID DESC LIMIT 100000"
            ):
                if apple_date is None:
                    continue
                # Heuristic: if > 1e12 it's nanoseconds.
                ts = apple_date / 1e9 if apple_date > 1e12 else apple_date
                unix = ts + epoch_offset
                if unix < cutoff:
                    break
                dt = datetime.fromtimestamp(unix, tz=timezone.utc).astimezone(LOCAL_TZ)
                by_hour[dt.hour] += 1
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return by_hour


__all__ = [
    "TimeModel", "DayShape", "WeekShape", "HourBin", "DayLog",
    "CommitsByHour", "WebReading",
    "CATEGORIES", "DOW_LABELS", "LOCAL_TZ",
    "build_time_model",
    "render_hour_bars", "render_dow_bars",
    "render_time_day_md", "render_time_week_md",
    "render_time_recent_md", "render_time_month_md",
    "messages_by_hour",
]
