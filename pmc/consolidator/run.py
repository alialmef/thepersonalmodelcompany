"""One full consolidator pass — turns the graph into a continuous
picture of how the user spends their time.

Order matters: structured rows first (pure compute), then the LLM
calls that synthesize them. self.md reads time.md; whoami reads
self.md. Each pass is idempotent — re-running overwrites with the
latest version.

Cost per pass at current scope:
  - characterize_projects: 1 LLM call per repo (typically 2-5)
  - compute_attention:     0 LLM calls
  - find_active_interests: 0 LLM calls
  - compose_time_md:       1 LLM call
  - compose_self_md:       1 LLM call
  - compose_whoami:        0 LLM calls
  → roughly 5-10 cheap calls per pass. ~$0.05-0.20 depending on model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import shutil

from pmc.cli.local_config import LocalConfig
from pmc.consolidator.characterize import (
    SELF_MD,
    characterize_projects,
    compose_self_md,
    compose_time_md,
    compose_whoami,
    compute_attention,
    find_active_interests,
)
from pmc.consolidator.reading import build_reading_layer
from pmc.consolidator.time_model import (
    build_time_model,
    render_time_day_md,
    render_time_month_md,
    render_time_recent_md,
    render_time_week_md,
)
from pmc.storage.graph_store import GraphStore


log = logging.getLogger("pmc.consolidator")


@dataclass
class ConsolidationResult:
    started_at: float
    duration_s: float
    user_id: str
    notes: str
    projects_count: int = 0
    attention_count: int = 0
    interests_count: int = 0


def _state_path(cfg: LocalConfig) -> Path:
    root = cfg.effective_storage_root()
    return (
        root / "users" / (cfg.user_id or "local") / "graph" / "synth"
        / "consolidator.json"
    )


def last_run_at(cfg: LocalConfig) -> Optional[float]:
    p = _state_path(cfg)
    if not p.is_file():
        return None
    try:
        return float(json.loads(p.read_text()).get("last_run_at", 0))
    except Exception:  # noqa: BLE001
        return None


def _self_md_path(cfg: LocalConfig) -> Path:
    root = cfg.effective_storage_root()
    return root / "users" / (cfg.user_id or "local") / "graph" / "synth" / SELF_MD


LAYERS = ("time", "work", "attention", "interests", "reading",
          "time_md", "self_md", "whoami")


def run_consolidation(
    cfg: LocalConfig,
    *,
    only: Optional[list[str]] = None,
    rebuild: bool = False,
    diff: bool = False,
) -> ConsolidationResult:
    """One full pass — or a targeted re-run.

    Args:
      only:    if given, only run these layers (others are read from
               existing portrait files where possible).
      rebuild: if True, wipe `graph/synth/portrait/*` before running.
      diff:    if True, save a snapshot of the current portrait BEFORE
               running, so we can diff after.
    """
    started = time.time()
    user_id = cfg.user_id or "local"
    log.info("consolidator: pass start for user %s (only=%s rebuild=%s diff=%s)",
             user_id, only, rebuild, diff)

    portrait_dir = (
        cfg.effective_storage_root()
        / "users" / user_id / "graph" / "synth" / "portrait"
    )

    # Snapshot for diff
    if diff:
        _snapshot_portrait(portrait_dir)

    # Rebuild — nuke existing portrait files (NOT the page-content cache,
    # which is expensive to re-fetch).
    if rebuild:
        if portrait_dir.is_dir():
            for f in portrait_dir.iterdir():
                if f.is_file():
                    f.unlink()
            log.info("consolidator: wiped portrait/ for rebuild")

    want = set(only) if only else set(LAYERS)
    unknown = want - set(LAYERS)
    if unknown:
        raise ValueError(f"unknown layer(s): {unknown}. valid: {LAYERS}")

    store = GraphStore(cfg.effective_storage_root())

    # 0. TIME MODEL — the bedrock.
    if "time" in want:
        tm = build_time_model(store, user_id=user_id)
        _write_md(cfg, user_id, "portrait/time_day.md",    render_time_day_md(tm))
        _write_md(cfg, user_id, "portrait/time_recent.md", render_time_recent_md(tm))
        _write_md(cfg, user_id, "portrait/time_week.md",   render_time_week_md(tm))
        _write_md(cfg, user_id, "portrait/time_month.md",  render_time_month_md(tm))
        log.info("consolidator: built time model — %d min/30d, rhythm=%s",
                 tm.total_minutes_30d, tm.rhythm_label)
    else:
        # Need tm for downstream layers — build cheaply, don't write files.
        tm = build_time_model(store, user_id=user_id)

    # 1. work
    if "work" in want:
        projects = characterize_projects(cfg, store, user_id=user_id, limit=10)
        log.info("consolidator: characterized %d projects", len(projects))
    else:
        projects = _load_projects(cfg)

    # 2. attention
    if "attention" in want:
        attention = compute_attention(cfg, store, user_id=user_id, limit=20)
        log.info("consolidator: computed attention rows: %d", len(attention))
    else:
        attention = _load_attention(cfg)

    # 3. interests
    if "interests" in want:
        interests = find_active_interests(cfg, store, user_id=user_id, limit=15)
        log.info("consolidator: found %d active interests", len(interests))
    else:
        interests = []

    # 3.5 READING
    reading_topics: list = []
    if "reading" in want:
        try:
            reading_pages, reading_topics = build_reading_layer(
                cfg, user_id=user_id,
            )
            log.info(
                "consolidator: reading layer — %d pages, %d topics",
                len(reading_pages), len(reading_topics),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("consolidator: reading layer failed: %s", e)
            reading_topics = []

    # 4. time.md
    if "time_md" in want:
        time_md = compose_time_md(
            cfg, user_id=user_id,
            projects=projects, attention=attention, interests=interests,
            time_model=tm, reading_topics=reading_topics,
        )
    else:
        time_md_path = portrait_dir / "time.md"
        time_md = time_md_path.read_text() if time_md_path.is_file() else ""

    # 5. self.md — read prior + revise
    if "self_md" in want:
        prior_self = None
        sp = _self_md_path(cfg)
        if sp.is_file() and not rebuild:
            prior_self = sp.read_text()
        self_md = compose_self_md(
            cfg, user_id=user_id, time_md=time_md, prior_self=prior_self,
        )
    else:
        sp = _self_md_path(cfg)
        self_md = sp.read_text() if sp.is_file() else ""

    # 6. whoami.txt
    if "whoami" in want:
        compose_whoami(
            cfg, user_id=user_id,
            self_md=self_md, projects=projects, attention=attention,
        )

    duration = time.time() - started

    p = _state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "last_run_at": started,
        "duration_s": duration,
        "version": "time-oriented-v1",
        "projects_count": len(projects),
        "attention_count": len(attention),
        "interests_count": len(interests),
    }, indent=2))

    log.info("consolidator: pass complete in %.1fs", duration)
    return ConsolidationResult(
        started_at=started,
        duration_s=duration,
        user_id=user_id,
        notes="time-oriented portrait v1",
        projects_count=len(projects),
        attention_count=len(attention),
        interests_count=len(interests),
    )


def _write_md(cfg: LocalConfig, user_id: str, rel: str, body: str) -> None:
    root = cfg.effective_storage_root()
    p = root / "users" / user_id / "graph" / "synth" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body.strip() + "\n")


def _snapshot_portrait(portrait_dir: Path) -> None:
    """Save a copy of the current portrait under
    `portrait/_snapshots/<iso-ts>/` for diffing."""
    if not portrait_dir.is_dir():
        return
    snap_root = portrait_dir / "_snapshots"
    snap_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    target = snap_root / ts
    target.mkdir()
    for f in portrait_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, target / f.name)
    log.info("consolidator: snapshot saved to %s", target)


def _load_projects(cfg: LocalConfig) -> list:
    """Re-read work.jsonl into ProjectCharacterization objects so
    downstream layers (time_md, whoami) can run without re-running
    the expensive characterize_projects pass."""
    from pmc.consolidator.characterize import ProjectCharacterization
    p = cfg.effective_storage_root() / "users" / (cfg.user_id or "local") \
        / "graph" / "synth" / "portrait" / "work.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.open():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out.append(ProjectCharacterization(**d))
        except Exception:  # noqa: BLE001
            continue
    return out


def _load_attention(cfg: LocalConfig) -> list:
    from pmc.consolidator.characterize import AttentionRow
    p = cfg.effective_storage_root() / "users" / (cfg.user_id or "local") \
        / "graph" / "synth" / "portrait" / "attention.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.open():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out.append(AttentionRow(**d))
        except Exception:  # noqa: BLE001
            continue
    return out


def diff_against_latest_snapshot(portrait_dir: Path) -> str:
    """After a consolidator run, compare the new portrait vs the most
    recent snapshot. Returns a unified-ish diff as a string."""
    snap_root = portrait_dir / "_snapshots"
    if not snap_root.is_dir():
        return "(no snapshots — run with --diff to enable)"
    snaps = sorted([p for p in snap_root.iterdir() if p.is_dir()])
    if not snaps:
        return "(no snapshots yet)"
    latest = snaps[-1]
    out_lines: list[str] = []
    for current in sorted(portrait_dir.iterdir()):
        if not current.is_file():
            continue
        prior = latest / current.name
        if not prior.is_file():
            out_lines.append(f"### {current.name}  (new file)")
            out_lines.append(current.read_text()[:600])
            out_lines.append("")
            continue
        if current.read_text() == prior.read_text():
            continue
        out_lines.append(f"### {current.name}  (changed)")
        # Best-effort line diff
        import difflib
        d = difflib.unified_diff(
            prior.read_text().splitlines(),
            current.read_text().splitlines(),
            fromfile=f"prior/{current.name}",
            tofile=f"current/{current.name}",
            n=2, lineterm="",
        )
        out_lines.extend(list(d)[:80])
        out_lines.append("")
    return "\n".join(out_lines) if out_lines else "(no changes vs latest snapshot)"


__all__ = ["run_consolidation", "last_run_at", "ConsolidationResult"]
