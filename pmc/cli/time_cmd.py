"""`pmc time` — inspect the time model.

Three views, same data:
    pmc time              today's shape (typical day, aggregated)
    pmc time week         weekly rhythm
    pmc time month        monthly aggregate

Renders the TimeModel using the existing reading-column UI: clean
ASCII histogram for the day, weekday bars for the week, category
table for the month.
"""

from __future__ import annotations

import argparse
from typing import Optional

from rich.console import Console
from rich.text import Text

from pmc.cli import ui
from pmc.cli.local_config import load
from pmc.consolidator.time_model import (
    DOW_LABELS,
    TimeModel,
    build_time_model,
    render_hour_bars,
)
from pmc.storage.graph_store import GraphStore


def cmd_time(args: argparse.Namespace) -> int:
    cfg = load()
    if cfg is None:
        print("no pmc config — run `pmc configure` first.")
        return 1

    store = GraphStore(cfg.effective_storage_root())
    tm = build_time_model(store, user_id=cfg.user_id or "local")

    console = Console()
    console.clear() if args.clear else None
    view = args.view or "day"

    if view == "day":
        _render_day(console, tm)
    elif view == "week":
        _render_week(console, tm)
    elif view == "month":
        _render_month(console, tm)
    elif view == "recent":
        _render_recent_days(console, tm)
    elif view == "all":
        _render_day(console, tm)
        _render_recent_days(console, tm)
        _render_week(console, tm)
        _render_month(console, tm)
    else:
        ui.say(console, f"unknown view {view!r}. try: day / recent / week / month / all",
               style=ui.WARN)
        return 2
    return 0


# ---------------------------------------------------------------------------
# Day view
# ---------------------------------------------------------------------------


def _render_day(console: Console, tm: TimeModel) -> None:
    ui.banner_top(
        console,
        title="time · typical day",
        subtitle=f"{tm.rhythm_label}  ·  {tm.sleep_summary}",
    )

    # Histogram with ruler.
    bars_text = render_hour_bars(tm.day, width_per_bar=3)
    for line in bars_text.splitlines():
        console.print(f"{ui.margin()}[{ui.ACCENT}]{line}[/]")
    console.print()

    # By-category breakdown.
    ui.card_title(console, "by category (minutes per typical day)")
    sorted_cats = sorted(
        tm.day.by_category.items(), key=lambda t: t[1], reverse=True
    )
    for cat, m in sorted_cats:
        if m <= 0:
            continue
        pct = (m / max(1, tm.day.total_minutes)) * 100
        bar_len = min(20, int(round(pct / 5)))
        bar = "█" * bar_len
        line = Text(ui.margin(), style="")
        line.append(f"{cat:<14}", style=ui.WHITE)
        line.append(f"{_fmt_min(m):>7}  ", style=ui.WHITE)
        line.append(bar.ljust(20), style=ui.ACCENT)
        line.append(f"  {pct:>4.1f}%", style=ui.DIM)
        console.print(line, highlight=False)
    console.print()

    # Commit overlay
    if tm.commits:
        ui.card_title(console, "coding hours (from git log, last 30 days)")
        for c in tm.commits:
            peak = max(range(24), key=lambda h: c.by_hour[h])
            count_peak = c.by_hour[peak]
            spread = sum(1 for v in c.by_hour if v > 0)
            line = Text(ui.margin(), style="")
            line.append(f"{c.repo_name:<26}", style=ui.WHITE)
            line.append(f"{c.total_30d:>3} commits  ", style=ui.WHITE)
            line.append(
                f"peak {peak:02d}:00 ({count_peak})  "
                f"across {spread} hours of the day",
                style=ui.DIM,
            )
            console.print(line, highlight=False)
        console.print()

    # Hour-by-hour narrative
    if tm.narrative_by_hour:
        ui.card_title(console, "what each hour typically is")
        # Group consecutive same-narrative hours
        last = None
        run_start = 0
        for h in range(24):
            label = tm.narrative_by_hour[h]
            if label != last:
                if last is not None:
                    _print_narrative_band(console, run_start, h - 1, last)
                last = label
                run_start = h
        if last is not None:
            _print_narrative_band(console, run_start, 23, last)
        console.print()

    # Messaging by hour (from chat.db, if FDA granted)
    if tm.messages_by_hour and any(tm.messages_by_hour):
        ui.card_title(console, "messages — when you actually communicate")
        msg_peak = max(range(24), key=lambda h: tm.messages_by_hour[h])
        msg_total = sum(tm.messages_by_hour)
        line = Text(ui.margin(), style="")
        line.append(f"{msg_total} messages  ", style=ui.WHITE)
        line.append(
            f"peak hour {msg_peak:02d}:00 ({tm.messages_by_hour[msg_peak]})",
            style=ui.DIM,
        )
        console.print(line, highlight=False)
        console.print()

    # Photo creation by hour
    if tm.photos_by_hour and any(tm.photos_by_hour):
        ui.card_title(console, "photos / video — when you create")
        peak = max(range(24), key=lambda h: tm.photos_by_hour[h])
        total = sum(tm.photos_by_hour)
        line = Text(ui.margin(), style="")
        line.append(f"{total} files in 30d  ", style=ui.WHITE)
        line.append(f"peak {peak:02d}:00 ({tm.photos_by_hour[peak]})",
                    style=ui.DIM)
        console.print(line, highlight=False)
        console.print()


def _render_recent_days(console: Console, tm: TimeModel) -> None:
    """`pmc time recent` — the last 7 days reconstructed from precise
    timestamps (git, chat.db, photos). Today first."""
    if not tm.recent_days:
        ui.say_dim(console, "(no per-day data — no precise timestamps available)")
        return
    ui.banner_top(
        console,
        title="time · last 7 days (reconstructed)",
        subtitle="commits · messages · photos, from precise timestamps",
    )
    from datetime import date as _date
    today_key = _date.today().isoformat()
    for d in tm.recent_days:
        is_today = d.date == today_key
        label = "today" if is_today else d.date
        line = Text(ui.margin(), style="")
        line.append(f"{label:<10}", style=ui.ACCENT_BOLD if is_today else ui.WHITE)
        # commits
        line.append(f" {d.commits:>3} commits", style=ui.WHITE if d.commits else ui.DIM)
        # repos
        if d.commit_repos:
            top_repo = max(d.commit_repos, key=lambda r: d.commit_repos[r])
            line.append(f" ({top_repo[:18]})", style=ui.DIM)
        else:
            line.append(" " * 21, style=ui.DIM)
        # messages
        if d.messages_sent_estimate > 0:
            line.append(f"  {d.messages_sent_estimate:>4} msg", style=ui.WHITE)
        else:
            line.append("  " + " " * 8, style=ui.DIM)
        # photos
        if d.photo_files > 0:
            line.append(f"  {d.photo_files:>2} photo/vid", style=ui.WHITE)
        # active window
        if d.first_event_hour is not None and d.last_event_hour is not None:
            line.append(
                f"  · active {d.first_event_hour:02d}:00–{d.last_event_hour:02d}:59",
                style=ui.DIM,
            )
        console.print(line, highlight=False)
    console.print()


def _print_narrative_band(console, h_start: int, h_end: int, label: str) -> None:
    """One narrative band — e.g. '11:00 – 02:00  coding'."""
    if label == "idle":
        style = ui.DIM
    else:
        style = ui.ACCENT
    span = f"{h_start:02d}:00 – {h_end:02d}:59" if h_start != h_end else f"{h_start:02d}:00"
    line = Text(ui.margin(), style="")
    line.append(f"{span:<14}", style=ui.DIM)
    line.append(label, style=style)
    console.print(line, highlight=False)


# ---------------------------------------------------------------------------
# Week view
# ---------------------------------------------------------------------------


def _render_week(console: Console, tm: TimeModel) -> None:
    ui.banner_top(
        console,
        title="time · weekly rhythm",
        subtitle=(
            f"weekday avg {_fmt_min(tm.week.weekday_avg_min)}  ·  "
            f"weekend avg {_fmt_min(tm.week.weekend_avg_min)}"
        ),
    )
    by_dow = tm.week.by_dow
    maxv = max(by_dow) or 1
    for i, m in enumerate(by_dow):
        bar_len = int(round(m / maxv * 28))
        bar = "█" * bar_len
        line = Text(ui.margin(), style="")
        line.append(f"{DOW_LABELS[i]:<4}", style=ui.DIM)
        line.append(bar.ljust(28), style=ui.ACCENT)
        line.append(f"  {_fmt_min(m)}", style=ui.WHITE)
        console.print(line, highlight=False)
    console.print()


# ---------------------------------------------------------------------------
# Month view
# ---------------------------------------------------------------------------


def _render_month(console: Console, tm: TimeModel) -> None:
    total_h = tm.total_minutes_30d / 60.0
    daily_h = total_h / 30.0
    ui.banner_top(
        console,
        title="time · last 30 days",
        subtitle=f"{total_h:.1f}h on Mac  ·  ~{daily_h:.1f}h/day",
    )

    # Category breakdown
    ui.card_title(console, "by category")
    sorted_cats = sorted(
        tm.day.by_category.items(), key=lambda t: t[1], reverse=True
    )
    total = max(1, tm.total_minutes_30d)
    for cat, m in sorted_cats:
        if m <= 0:
            continue
        pct = m / total * 100
        bar_len = min(28, int(round(pct / 3)))
        line = Text(ui.margin(), style="")
        line.append(f"{cat:<14}", style=ui.WHITE)
        line.append(f"{_fmt_min(m):>7}  ", style=ui.WHITE)
        line.append(("█" * bar_len).ljust(28), style=ui.ACCENT)
        line.append(f"  {pct:>4.1f}%", style=ui.DIM)
        console.print(line, highlight=False)
    console.print()

    # Web reading
    if tm.web.by_subcat:
        ui.card_title(console, "reading — what the browser is about")
        for subcat, v in sorted(
            tm.web.by_subcat.items(), key=lambda t: t[1], reverse=True
        )[:10]:
            line = Text(ui.margin(), style="")
            line.append(f"{subcat:<18}", style=ui.WHITE)
            line.append(f"{v:>5} visits", style=ui.DIM)
            console.print(line, highlight=False)
        console.print()
        ui.card_title(console, "top domains")
        for d, v, s in tm.web.top_domains[:12]:
            line = Text(ui.margin(), style="")
            line.append(f"{d:<32}", style=ui.WHITE)
            line.append(f"{v:>4}  ", style=ui.WHITE)
            line.append(f"({s})", style=ui.DIM)
            console.print(line, highlight=False)
        console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_min(m: int) -> str:
    if m < 60:
        return f"{m}m"
    h = m / 60
    return f"{h:.1f}h"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "time",
        help="Inspect the time model — how the user actually spends time",
    )
    p.add_argument(
        "view", nargs="?", default="day",
        choices=("day", "recent", "week", "month", "all"),
        help="which view to render (default: day)",
    )
    p.add_argument("--no-clear", dest="clear", action="store_false",
                   default=True, help="don't clear the screen")
    p.set_defaults(func=cmd_time)


__all__ = ["cmd_time", "register"]
