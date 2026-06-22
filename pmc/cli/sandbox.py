"""`pmc sandbox` — the autonomous build harness.

A single command that:
  1. Wipes ALL portrait outputs (no prior-self.md feedback, no stale data)
  2. Wipes the page-content cache (next reading pass re-fetches)
  3. Runs the full consolidator from cold
  4. Builds the agenda (actionable + learnable items)
  5. Saves the entire output to `portrait/iterations/<n>-<timestamp>/`
  6. Reports a compact summary

Used both interactively (`pmc sandbox`) and as the iteration loop the
assistant runs during autonomous development sessions.

NOTHING in this module assumes a particular kind of user. The substrate
+ the LLM characterization do the per-user work.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

from pmc.cli import ui
from pmc.cli.local_config import load


def cmd_sandbox(args: argparse.Namespace) -> int:
    cfg = load()
    if cfg is None:
        print("no config — run pmc configure first.")
        return 1
    console = Console()
    if args.clear:
        console.clear()
    ui.banner_top(
        console,
        title="pmc sandbox",
        subtitle="fresh-start build  ·  every layer from zero",
    )

    user_id = cfg.user_id or "local"
    root = cfg.effective_storage_root()
    portrait_dir = root / "users" / user_id / "graph" / "synth" / "portrait"
    cache_dir = root / "users" / user_id / "graph" / "raw" / "page_content"
    iter_root = portrait_dir / "iterations"
    iter_root.mkdir(parents=True, exist_ok=True)

    next_idx = _next_iteration_index(iter_root)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    iter_dir = iter_root / f"{next_idx:03d}-{ts}"

    # Wipe portrait files (but keep iterations subdir).
    if portrait_dir.is_dir():
        for f in portrait_dir.iterdir():
            if f.is_file():
                f.unlink()
        ui.say_dim(console, f"wiped {portrait_dir}/*.{{md,jsonl,txt}}")

    # Wipe page content cache so reading layer runs cold.
    if cache_dir.is_dir() and not args.keep_pages:
        shutil.rmtree(cache_dir)
        ui.say_dim(console, f"wiped page content cache ({cache_dir.name})")
    elif args.keep_pages:
        ui.say_dim(console, "keeping page content cache (--keep-pages)")

    console.print()

    # Run the consolidator from scratch.
    started = time.time()
    from pmc.consolidator import run_consolidation
    result = run_consolidation(cfg, rebuild=False, diff=False)

    # Run the agenda layer on top.
    ui.say_dim(console, "building agenda (actionable + learnable items)…")
    from pmc.consolidator.agenda import build_agenda
    self_md = _read_md(portrait_dir / "self.md")
    time_md = _read_md(portrait_dir / "time.md")
    reading_md = _read_md(portrait_dir / "reading.md")
    recent_md = _read_md(portrait_dir / "time_recent.md")
    try:
        agenda = build_agenda(
            cfg, user_id=user_id,
            self_md=self_md, time_md=time_md,
            reading_md=reading_md, recent_md=recent_md,
        )
    except Exception as e:  # noqa: BLE001
        ui.say(console, f"agenda step failed: {e}", style=ui.WARN)
        agenda = None

    # Compose the canonical top-level picture.
    # If a prior iteration exists, pass its picture.md so the new pass
    # DEEPENS rather than restarts (the user's "continuously strengthen
    # the graph" principle).
    ui.say_dim(console, "composing picture.md (high-level → drilled-down)…")
    from pmc.consolidator.characterize import compose_picture_md
    agenda_md = _read_md(portrait_dir / "agenda.md") if agenda else ""
    prior_picture = _read_latest_prior_picture(iter_root)
    if prior_picture:
        ui.say_dim(console, "(deepening from prior picture.md)")
    # Pull broader extractor signal into the LLM context — calendar,
    # photos, locations, wallet, music — so the picture isn't blind to
    # life dimensions that aren't on the user's screen all day.
    other_signals = _gather_other_signals(cfg, user_id)
    try:
        compose_picture_md(
            cfg, user_id=user_id,
            self_md=self_md, time_md=time_md,
            reading_md=reading_md, recent_md=recent_md,
            agenda_md=agenda_md,
            prior_picture=prior_picture,
            other_signals=other_signals,
        )
    except Exception as e:  # noqa: BLE001
        ui.say(console, f"picture.md step failed: {e}", style=ui.WARN)

    duration = time.time() - started

    # Save this iteration.
    iter_dir.mkdir(parents=True, exist_ok=True)
    for f in portrait_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, iter_dir / f.name)
    note = {
        "iteration": next_idx,
        "ts": ts,
        "duration_s": round(duration, 2),
        "projects_count": result.projects_count,
        "attention_count": result.attention_count,
        "interests_count": result.interests_count,
        "actionable_count": len(agenda.actionable) if agenda else 0,
        "learnable_count": len(agenda.learnable) if agenda else 0,
    }
    (iter_dir / "_meta.json").write_text(json.dumps(note, indent=2))

    console.print()
    ui.say(console,
           f"{ui.GLYPH_DONE} iteration {next_idx:03d} complete  ·  "
           f"{duration:.1f}s",
           style=ui.OK)
    ui.say_dim(console, f"saved to {iter_dir.relative_to(root)}")
    console.print()

    # Print a compact summary.
    if agenda:
        ui.card_title(console, "agenda — actionable")
        if not agenda.actionable:
            ui.say_dim(console, "(none surfaced)")
        else:
            for it in agenda.actionable[:8]:
                ui.say(console, f"• {it.label}", style=ui.WHITE)
                if it.detail:
                    ui.say_dim(console, f"  {it.detail}")
                if it.suggested_action:
                    ui.say_dim(console, f"  → {it.suggested_action}")
                console.print()

        ui.card_title(console, "agenda — learnable")
        if not agenda.learnable:
            ui.say_dim(console, "(none surfaced)")
        else:
            for it in agenda.learnable[:8]:
                ui.say(console, f"• {it.label}", style=ui.WHITE)
                if it.detail:
                    ui.say_dim(console, f"  {it.detail}")
                if it.automation_hint:
                    ui.say_dim(console, f"  → {it.automation_hint}")
                console.print()

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_iteration_index(iter_root: Path) -> int:
    if not iter_root.is_dir():
        return 1
    indices = []
    for d in iter_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name.split("-", 1)[0]
        try:
            indices.append(int(name))
        except ValueError:
            continue
    return (max(indices) + 1) if indices else 1


def _read_md(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text()


def _gather_other_signals(cfg, user_id: str) -> str:
    """Sweep the graph for extractor outputs the picture would
    otherwise miss — calendar events, photos with people, significant
    locations, wallet passes, music, voice memos count. Universal
    across users: we only count what's there, the LLM characterizes.
    """
    from pmc.storage.graph_store import GraphStore
    store = GraphStore(cfg.effective_storage_root())
    lines: list[str] = []

    # Calendar / events
    events = list(store.iter_entities(user_id, "event"))
    if events:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        upcoming = []
        recent = []
        for e in events:
            s = e.get("start") or ""
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            days = (dt - now).days
            if 0 <= days <= 30:
                upcoming.append((dt, e.get("title") or "?"))
            elif -7 <= days < 0:
                recent.append((dt, e.get("title") or "?"))
        if upcoming or recent:
            lines.append(f"CALENDAR: {len(events)} total events. "
                         f"{len(upcoming)} in next 30 days. "
                         f"{len(recent)} in last 7 days.")
            for dt, title in sorted(upcoming, key=lambda t: t[0])[:5]:
                lines.append(f"  upcoming: {dt.date()} — {title[:60]}")
            for dt, title in sorted(recent, key=lambda t: t[0], reverse=True)[:3]:
                lines.append(f"  recent: {dt.date()} — {title[:60]}")
            lines.append("")

    # Photos — top photographed places (already a pattern, surface here for picture)
    places = [p for p in store.iter_entities(user_id, "place")
              if (p.get("visit_count") or 0) > 0]
    if places:
        top_places = sorted(places,
                            key=lambda p: p.get("visit_count") or 0,
                            reverse=True)[:8]
        lines.append(f"PLACES: {len(places)} distinct, top photographed:")
        for p in top_places:
            lbl = p.get("label") or "?"
            v = p.get("visit_count") or 0
            lines.append(f"  - {lbl[:50]} (×{v})")
        lines.append("")

    # Wallet passes — recent boarding passes / event tickets surface what's happening IRL
    projects = list(store.iter_entities(user_id, "project"))
    wallet = [p for p in projects
              if any("wallet:" in s for s in (p.get("sources") or []))]
    if wallet:
        flights = [p for p in wallet
                   if any("wallet:boardingPass" in s for s in p.get("sources") or [])]
        tickets = [p for p in wallet
                   if any("wallet:eventTicket" in s for s in p.get("sources") or [])]
        lines.append(f"WALLET: {len(flights)} flights, {len(tickets)} event tickets in graph.")
        # Most recent of each
        for kind, group in [("flights", flights), ("tickets", tickets)]:
            recent = sorted(
                [(p.get("last_activity") or "", p.get("name") or "?")
                 for p in group],
                reverse=True,
            )[:3]
            for ts, name in recent:
                lines.append(f"  recent {kind[:-1]}: {ts[:10]} — {name[:60]}")
        lines.append("")

    return "\n".join(lines).strip()


def _read_latest_prior_picture(iter_root: Path) -> str:
    """Find the most recent prior iteration's picture.md, if any.
    Returns its text so the next pass can DEEPEN rather than restart."""
    if not iter_root.is_dir():
        return ""
    iters = sorted(
        [p for p in iter_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in iters:
        pic = d / "picture.md"
        if pic.is_file():
            return pic.read_text()
    return ""


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "sandbox",
        help="Fresh-start build: wipe portrait + page cache, rebuild "
             "everything from zero, run the agenda layer, save as a "
             "numbered iteration for inspection.",
    )
    p.add_argument("--keep-pages", action="store_true",
                   help="don't wipe the page-content cache "
                        "(faster but reading layer won't re-fetch)")
    p.add_argument("--no-clear", dest="clear", action="store_false",
                   default=True, help="don't clear the screen")
    p.set_defaults(func=cmd_sandbox)


__all__ = ["cmd_sandbox", "register"]
