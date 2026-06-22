"""Characterization passes — turn graph rows into agent-readable memory.

Each function here takes a slice of the graph + supplemental evidence
(README files for repos, git log, transcripts for voice memos) and
calls the user's chosen frontier model with a tight prompt that
returns a one-paragraph characterization.

The output goes into `graph/synth/portrait/{kind}.jsonl`. The MCP
server reads from these files; so does `pmc chat`'s context block
when the portrait exists.

Order in the file mirrors the priority the user asked for —
activity first, relationships last:

  characterize_projects()   - repos: PMC, token-street, etc.
  characterize_attention()  - apps + web: where time goes (no LLM, pure compute)
  characterize_themes()     - active themes only (mentions_30d > 0)
  characterize_people()     - small batch, last priority

Each function is idempotent. Re-running overwrites with the latest
characterization. The consolidator's run.py decides which subset to
re-characterize on each pass (changed entities first, slow refresh
of stable ones).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pmc.agent.providers.base import Message, ProviderConfig, ProviderError
from pmc.agent.providers.registry import get_provider
from pmc.cli.local_config import LocalConfig
from pmc.storage.graph_store import GraphStore


log = logging.getLogger("pmc.consolidator.characterize")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class ProjectCharacterization:
    """One characterized project/repo. Written to portrait/projects.jsonl."""
    id: str
    name: str
    kind: str                    # "repo" | "wallet_project" | etc.
    status: str                  # "active" | "dormant" | "wrapped"
    one_liner: str               # short headline the agent surfaces first
    body: str                    # one-paragraph characterization
    language: Optional[str] = None
    path: Optional[str] = None
    last_activity: Optional[str] = None
    commits_30d: int = 0
    salience: float = 0.0
    characterized_at: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Projects (repos)
# ---------------------------------------------------------------------------


PROJECTS_PORTRAIT  = "portrait/work.jsonl"
ATTENTION_PORTRAIT = "portrait/attention.jsonl"
INTERESTS_PORTRAIT = "portrait/interests.jsonl"
TIME_MD            = "portrait/time.md"
SELF_MD            = "portrait/self.md"
WHOAMI_TXT         = "portrait/whoami.txt"


def characterize_projects(
    cfg: LocalConfig,
    store: GraphStore,
    *,
    user_id: str,
    limit: int = 20,
) -> list[ProjectCharacterization]:
    """Pull each repo, gather evidence, call the LLM, write portrait."""
    repos = list(store.iter_entities(user_id, "repo"))
    if not repos:
        log.info("characterize_projects: no repos found")
        return []
    repos.sort(key=_repo_salience, reverse=True)
    repos = repos[:limit]

    out: list[ProjectCharacterization] = []
    for repo in repos:
        try:
            characterized = _characterize_one_project(cfg, repo)
            out.append(characterized)
        except Exception as e:  # noqa: BLE001
            log.warning("characterize_project failed for %s: %s",
                        repo.get("name", "?"), e)

    _write_portrait(cfg, user_id, PROJECTS_PORTRAIT, out)
    return out


def _repo_salience(r: dict[str, Any]) -> float:
    """Active = high salience. Dormant = low. Pure compute."""
    c30 = r.get("commit_count_30d") or 0
    last = r.get("last_commit") or ""
    days_old = _days_since(last)
    if c30 > 0:
        return 1.0
    if days_old < 30:
        return 0.7
    if days_old < 90:
        return 0.4
    return 0.1


def _characterize_one_project(
    cfg: LocalConfig, repo: dict[str, Any]
) -> ProjectCharacterization:
    """Gather evidence + call LLM for one repo."""
    name = repo.get("name") or "(unknown)"
    path = repo.get("path") or ""
    language = repo.get("language")
    last_commit = repo.get("last_commit")
    c30 = int(repo.get("commit_count_30d") or 0)
    days_since = _days_since(last_commit)

    # Status heuristic — used both in the LLM prompt and the row.
    if c30 > 0:
        status = "active"
    elif days_since < 60:
        status = "stable"
    elif days_since < 180:
        status = "dormant"
    else:
        status = "archived"

    # Evidence — read README + recent git log if the path still exists.
    readme = _read_readme(path)
    recent_commits = _git_log_messages(path, n=10)

    # LLM call.
    prompt = _project_prompt(
        name=name, path=path, language=language, status=status,
        days_since=days_since, c30=c30, readme=readme,
        commits=recent_commits,
    )
    text = _ask_llm(cfg, system=_PROJECT_SYSTEM, user=prompt, max_tokens=300)

    # Parse out one_liner + body. The prompt asks for "HEAD: ...\n\n..." shape.
    one_liner, body = _split_head_body(text, fallback_one_liner=name)

    return ProjectCharacterization(
        id=repo.get("id") or name,
        name=name,
        kind="repo",
        status=status,
        one_liner=one_liner,
        body=body,
        language=language,
        path=path,
        last_activity=last_commit,
        commits_30d=c30,
        salience=_repo_salience(repo),
        characterized_at=_now_iso(),
        evidence={
            "readme_excerpt": readme[:240] if readme else "",
            "recent_commit_count": len(recent_commits),
        },
    )


_PROJECT_SYSTEM = """\
You are characterizing one of the user's projects for an agent that
needs to know what the user is currently doing. The agent will read
your output before talking to the user.

Constraints:
  - Voice: second person, plain prose. No marketing language.
  - Status tense: if active, present tense. If dormant/archived, past tense.
  - No invention. If you don't know what it is, say so plainly.
  - No filler ("This is a project that..." / "Overall, X is...").

Output EXACTLY this shape:
  HEAD: <one short line, ~60 chars, what this project IS to the user right now>

  <one paragraph, 2-4 sentences, what it does, current state, themes,
  and — if dormant — when they stopped and what they were doing then>
"""


def _project_prompt(
    *, name: str, path: str, language: Optional[str], status: str,
    days_since: int, c30: int, readme: str, commits: list[str],
) -> str:
    lines = [
        f"Project: {name}",
        f"Path: {path}",
        f"Language: {language or '(unknown)'}",
        f"Status (computed): {status}",
        f"Commits in last 30 days: {c30}",
        f"Days since last commit: {days_since}",
    ]
    if readme:
        lines.append(f"\nREADME (truncated):\n{readme[:1200]}")
    if commits:
        lines.append(f"\nRecent commit messages:")
        for m in commits[:10]:
            lines.append(f"  - {m}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers — evidence gathering
# ---------------------------------------------------------------------------


def _read_readme(repo_path: str) -> str:
    if not repo_path:
        return ""
    p = Path(repo_path).expanduser()
    if not p.is_dir():
        return ""
    for candidate in ("README.md", "readme.md", "README", "README.rst"):
        f = p / candidate
        if f.is_file():
            try:
                return f.read_text(errors="ignore")
            except OSError:
                continue
    return ""


def _git_log_messages(repo_path: str, *, n: int = 10) -> list[str]:
    if not repo_path:
        return []
    p = Path(repo_path).expanduser()
    if not (p / ".git").is_dir():
        return []
    try:
        res = subprocess.run(
            ["git", "-C", str(p), "log", f"-{n}", "--pretty=%s"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode != 0:
            return []
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except (subprocess.TimeoutExpired, OSError):
        return []


def _days_since(iso: Optional[str]) -> int:
    if not iso:
        return 9999
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return 9999
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt_min(m: int) -> str:
    if m < 60:
        return f"{m}m"
    return f"{m/60:.1f}h"


def _strip_gen_header(text: str) -> str:
    """Remove the leading `<!-- generated by pmc consolidate ... -->` line(s)
    we add at file write time, so they don't get re-included when prior
    output gets fed back to the model."""
    lines = text.splitlines()
    out: list[str] = []
    skipping = True
    for line in lines:
        if skipping and (
            line.strip().startswith("<!--") or line.strip() == ""
        ):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------


def _ask_llm(cfg: LocalConfig, *, system: str, user: str, max_tokens: int) -> str:
    """Synchronous wrapper around the user's provider."""
    provider = get_provider(cfg.provider)
    if provider is None:
        raise RuntimeError(f"unknown provider {cfg.provider!r}")
    pcfg = ProviderConfig(
        provider=cfg.provider,
        model=cfg.model,
        api_key=cfg.api_key,
    )
    async def _call() -> str:
        resp = await provider.chat(
            messages=[Message(role="user", content=user)],
            config=pcfg,
            system=system,
            max_tokens=max_tokens,
        )
        return resp.text
    return asyncio.run(_call())


def _split_head_body(text: str, *, fallback_one_liner: str) -> tuple[str, str]:
    """Parse 'HEAD: ...\\n\\n<body>' shape. Robust to slight model
    variations (markdown decoration, extra whitespace)."""
    text = text.strip()
    # Strip leading code fences if any
    if text.startswith("```"):
        text = text.split("```", 2)[-1].strip()
    lines = text.splitlines()
    head = ""
    body_start = 0
    for i, line in enumerate(lines):
        l = line.strip().lstrip("*# ").strip()
        if l.upper().startswith("HEAD:"):
            head = l[5:].strip().lstrip("*").strip()
            body_start = i + 1
            break
    if not head:
        # No HEAD: prefix; treat the first line as headline.
        head = lines[0].strip().lstrip("*# ").strip()[:80] if lines else fallback_one_liner
        body_start = 1
    body = "\n".join(lines[body_start:]).strip()
    return head or fallback_one_liner, body


# ---------------------------------------------------------------------------
# Portrait write
# ---------------------------------------------------------------------------


def _portrait_dir(cfg: LocalConfig, user_id: str) -> Path:
    root = cfg.effective_storage_root() / "users" / user_id / "graph" / "synth" / "portrait"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_portrait(cfg: LocalConfig, user_id: str, rel: str, items: list[Any]) -> None:
    path = _portrait_dir(cfg, user_id).parent / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
    os.replace(tmp, path)
    log.info("portrait: wrote %d rows to %s", len(items), rel)


# ---------------------------------------------------------------------------
# Attention — pure compute over app_usage. No LLM.
# ---------------------------------------------------------------------------


@dataclass
class AttentionRow:
    """One row of where time goes: app or web domain."""
    id: str
    kind: str               # "app" | "web"
    label: str
    category: str
    minutes_30d: int
    share_pct: float        # percent of total tracked time
    status: str             # "active" | "dormant"


def compute_attention(
    cfg: LocalConfig,
    store: GraphStore,
    *,
    user_id: str,
    limit: int = 20,
) -> list[AttentionRow]:
    """Sum minutes-per-30-days across apps. Pure compute, no LLM call.
    Uses the proper categorization + label resolution from
    pmc.consolidator.categorize so 'Cursor' isn't displayed as
    '230313mzl4w4u92'."""
    from pmc.consolidator.categorize import app_label, categorize_app

    rows: list[dict[str, Any]] = []
    for a in store.iter_entities(user_id, "app"):
        mins = int(a.get("minutes_30d") or 0)
        if mins <= 0:
            continue
        rows.append(a)
    total = sum(int(r.get("minutes_30d") or 0) for r in rows) or 1
    rows.sort(key=lambda r: r.get("minutes_30d") or 0, reverse=True)
    out: list[AttentionRow] = []
    for a in rows[:limit]:
        mins = int(a.get("minutes_30d") or 0)
        label = app_label(
            bundle_id=a.get("bundle_id"),
            display_name=a.get("display_name"),
        )
        category = categorize_app(
            bundle_id=a.get("bundle_id"),
            display_name=a.get("display_name"),
            fallback=a.get("category"),
        )
        out.append(AttentionRow(
            id=a.get("id") or a.get("bundle_id") or label,
            kind="app",
            label=label,
            category=category,
            minutes_30d=mins,
            share_pct=round(100 * mins / total, 1),
            status="active",
        ))
    _write_portrait(cfg, user_id, ATTENTION_PORTRAIT, out)
    return out


# ---------------------------------------------------------------------------
# Interests — active themes only. Light characterization (one call, batched).
# ---------------------------------------------------------------------------


@dataclass
class InterestRow:
    id: str
    label: str
    trajectory: Optional[str]
    mentions_30d: int
    salience: float


def find_active_interests(
    cfg: LocalConfig,
    store: GraphStore,
    *,
    user_id: str,
    limit: int = 15,
) -> list[InterestRow]:
    """Filter themes to those with current activity (mentions_30d > 0)
    and sort by mention volume. Pure compute. The LLM-driven theme
    characterization can come later — this gets us the list."""
    rows: list[dict[str, Any]] = []
    for t in store.iter_entities(user_id, "theme"):
        m = int(t.get("mentions_30d") or 0)
        if m <= 0:
            continue
        rows.append(t)
    rows.sort(key=lambda r: r.get("mentions_30d") or 0, reverse=True)
    out: list[InterestRow] = []
    for t in rows[:limit]:
        m = int(t.get("mentions_30d") or 0)
        out.append(InterestRow(
            id=t.get("id") or t.get("label") or "?",
            label=(t.get("label") or "").strip(),
            trajectory=t.get("trajectory"),
            mentions_30d=m,
            salience=min(1.0, m / 30.0),  # 30 mentions → 1.0
        ))
    _write_portrait(cfg, user_id, INTERESTS_PORTRAIT, out)
    return out


# ---------------------------------------------------------------------------
# time.md — the spine
# ---------------------------------------------------------------------------


def compose_time_md(
    cfg: LocalConfig,
    *,
    user_id: str,
    projects: list[ProjectCharacterization],
    attention: list[AttentionRow],
    interests: list[InterestRow],
    time_model: Optional[Any] = None,
    reading_topics: Optional[list[Any]] = None,
) -> str:
    """Compose the top-level 'how you spend your time' picture.
    One LLM call. Synthesizes the structured rows into prose."""
    from pmc.consolidator.categorize import is_private_subcat

    work_lines = []
    for p in projects:
        if p.status == "active":
            work_lines.append(
                f"  - {p.name} ({p.language or '?'}) — active, {p.commits_30d} commits/30d. "
                f"{p.one_liner}"
            )
        elif p.status in ("stable", "dormant"):
            work_lines.append(f"  - {p.name} — {p.status}. {p.one_liner}")

    att_lines = []
    for a in attention[:10]:
        att_lines.append(
            f"  - {a.label} ({a.category}) — {a.minutes_30d} min/30d "
            f"({a.share_pct}%)"
        )

    interest_lines = []
    for t in interests[:10]:
        interest_lines.append(f"  - {t.label} ({t.mentions_30d} mentions/30d)")

    # NEW: time model section — the bedrock evidence.
    time_lines: list[str] = []
    if time_model is not None:
        tm = time_model
        total_h = tm.total_minutes_30d / 60.0
        time_lines.append(
            f"Total tracked Mac time, last 30 days: {total_h:.1f}h "
            f"(~{total_h / 30:.1f}h/day)"
        )
        time_lines.append(f"Daily rhythm: {tm.rhythm_label}.  {tm.sleep_summary}.")
        time_lines.append("")
        time_lines.append("Category breakdown (typical day):")
        sorted_cats = sorted(tm.day.by_category.items(),
                             key=lambda t: t[1], reverse=True)
        for cat, m in sorted_cats:
            if m <= 0:
                continue
            pct = m / max(1, tm.day.total_minutes) * 100
            time_lines.append(f"  - {cat}: {_fmt_min(m)} ({pct:.1f}%)")
        time_lines.append("")
        # Peak hours per category
        time_lines.append("Peak hours by category (when each happens, 0-23):")
        for cat, _ in sorted_cats[:5]:
            cat_hours = [tm.day.hours[h].by_category.get(cat, 0) for h in range(24)]
            if not any(cat_hours):
                continue
            peak_h = max(range(24), key=lambda h: cat_hours[h])
            time_lines.append(f"  - {cat}: peaks at {peak_h:02d}:00")
        time_lines.append("")
        # Weekly rhythm
        from pmc.consolidator.time_model import DOW_LABELS
        wk = tm.week
        peak_dow = max(range(7), key=lambda i: wk.by_dow[i])
        quiet_dow = min(range(7), key=lambda i: wk.by_dow[i])
        time_lines.append(
            f"Weekly: weekday avg {_fmt_min(wk.weekday_avg_min)} / "
            f"weekend avg {_fmt_min(wk.weekend_avg_min)}. "
            f"Peak day: {DOW_LABELS[peak_dow]} ({_fmt_min(wk.by_dow[peak_dow])}). "
            f"Quiet: {DOW_LABELS[quiet_dow]} ({_fmt_min(wk.by_dow[quiet_dow])})."
        )
        time_lines.append("")
        # Commits
        if tm.commits:
            time_lines.append("Coding hours (from git log):")
            for c in tm.commits:
                if c.total_30d <= 0:
                    continue
                peak_h = max(range(24), key=lambda h: c.by_hour[h])
                spread = sum(1 for v in c.by_hour if v > 0)
                time_lines.append(
                    f"  - {c.repo_name}: {c.total_30d} commits, "
                    f"peak {peak_h:02d}:00, spread across {spread} hours"
                )
            time_lines.append("")
        # Reading subcategories (FILTERED for private)
        if tm.web.by_subcat:
            time_lines.append("Reading subcategories (browser visits, 30d):")
            for subcat, v in sorted(tm.web.by_subcat.items(),
                                    key=lambda t: t[1], reverse=True)[:8]:
                if is_private_subcat(subcat):
                    continue
                time_lines.append(f"  - {subcat}: {v} visits")
            time_lines.append("")

        # Hour-by-hour narrative — the dominant activity each hour
        if tm.narrative_by_hour:
            time_lines.append("Typical day, hour by hour (dominant activity):")
            last = None
            run_start = 0
            for h in range(24):
                lbl = tm.narrative_by_hour[h]
                if lbl != last:
                    if last is not None and last != "idle":
                        time_lines.append(
                            f"  {run_start:02d}:00–{h-1:02d}:59  {last}"
                        )
                    elif last == "idle" and h - run_start > 1:
                        time_lines.append(
                            f"  {run_start:02d}:00–{h-1:02d}:59  idle"
                        )
                    last = lbl
                    run_start = h
            if last is not None:
                time_lines.append(f"  {run_start:02d}:00–23:59  {last}")
            time_lines.append("")

        # iMessage timing — direct from chat.db, granular comms signal
        if tm.messages_by_hour and any(tm.messages_by_hour):
            msg_total = sum(tm.messages_by_hour)
            msg_peak = max(range(24), key=lambda h: tm.messages_by_hour[h])
            time_lines.append(
                f"Messaging (iMessage, last 30d): {msg_total} messages, "
                f"peak hour {msg_peak:02d}:00 "
                f"({tm.messages_by_hour[msg_peak]} that hour)"
            )
            time_lines.append("")

        # Photo / video file creation timing
        if tm.photos_by_hour and any(tm.photos_by_hour):
            ph_total = sum(tm.photos_by_hour)
            ph_peak = max(range(24), key=lambda h: tm.photos_by_hour[h])
            time_lines.append(
                f"Photos & video files made: {ph_total} in 30d, "
                f"peak {ph_peak:02d}:00."
            )
            time_lines.append("")

        # Year-scale aggregate — over the LAST 365 DAYS, what did they ship
        if tm.year_scale and tm.year_scale.commits_by_month:
            ys = tm.year_scale
            total_year = sum(ys.commits_by_repo_year.values())
            time_lines.append(
                f"LAST 365 DAYS: {total_year} commits across "
                f"{len(ys.commits_by_repo_year)} repos, active in "
                f"{ys.months_with_activity} months."
            )
            # Per-repo year totals
            for name, n in sorted(ys.commits_by_repo_year.items(),
                                  key=lambda t: t[1], reverse=True):
                time_lines.append(f"  - {name}: {n} commits/year")
            # Monthly pattern — show the active months
            if ys.commits_by_month:
                months_sorted = sorted(ys.commits_by_month.items())
                # show last 12 months
                time_lines.append("  monthly:")
                for month, n in months_sorted[-12:]:
                    time_lines.append(f"    {month}: {n}")
            time_lines.append("")

        # Per-day reconstruction — last 7 days from precise timestamps
        if tm.recent_days:
            time_lines.append("Last 7 days (from precise timestamps):")
            for d in tm.recent_days:
                bits = [d.date]
                if d.commits:
                    top_repo = max(
                        d.commit_repos, key=d.commit_repos.get
                    ) if d.commit_repos else ""
                    bits.append(f"{d.commits} commits ({top_repo})")
                if d.messages_sent_estimate:
                    bits.append(f"{d.messages_sent_estimate} msgs")
                if d.photo_files:
                    bits.append(f"{d.photo_files} photos")
                if d.first_event_hour is not None:
                    bits.append(
                        f"active {d.first_event_hour:02d}-{d.last_event_hour:02d}"
                    )
                time_lines.append("  " + " · ".join(bits))

    # READING TOPICS — what the user has been actually thinking about,
    # extracted from the content of pages they visited.
    reading_lines: list[str] = []
    if reading_topics:
        for t in reading_topics[:8]:
            label = getattr(t, "label", "") or ""
            summary = getattr(t, "summary", "") or ""
            v = getattr(t, "visits_total", 0)
            reading_lines.append(f"- {label} ({v} visits): {summary}")

    evidence_blocks = []
    if time_lines:
        evidence_blocks.append("TIME MODEL (bedrock — how the user actually spends time):\n"
                               + "\n".join(time_lines))
    evidence_blocks.append("ACTIVE WORK:\n" + ("\n".join(work_lines) or "  (none)"))
    evidence_blocks.append(
        "TOP APPS (last 30d minutes):\n" + ("\n".join(att_lines) or "  (none)")
    )
    if reading_lines:
        evidence_blocks.append(
            "WHAT THEY'VE BEEN READING (clustered from actual page content):\n"
            + "\n".join(reading_lines)
        )
    if interest_lines:
        evidence_blocks.append(
            "INTERESTS (active themes):\n" + "\n".join(interest_lines)
        )

    evidence = "\n\n".join(evidence_blocks)

    text = _ask_llm(
        cfg,
        system=_TIME_SYSTEM,
        user=evidence,
        max_tokens=600,
    )

    # Persist to disk.
    path = _portrait_dir(cfg, user_id).parent / TIME_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"<!-- generated by pmc consolidate at {_now_iso()} -->\n\n"
    path.write_text(header + text.strip() + "\n")
    log.info("portrait: wrote time.md (%d chars)", len(text))
    return text


_TIME_SYSTEM = """\
You are writing a portrait of how the user spends their time. The
agent that talks to the user will read this to understand who they
are and what's currently in their life.

Voice:
  - Second person, plain prose. No headers, no bullet lists in the
    output (the structured data already has those — your job is to
    integrate it into 2-4 short paragraphs).
  - Specific. Cite numbers when they're meaningful. Cite project
    names. Don't say "various projects" — name them.
  - Present tense for active things. Past tense for dormant/archived.
  - Never moralize. Never tell them to do anything. Just describe.
  - No "Overall" / "In summary" / "Key takeaways" framing. Just the
    portrait, starting in motion.

What to cover:
  - What they're building right now (the active work)
  - Where their hours go (apps, attention pattern)
  - What's been on their mind (interests that show up across sources)
  - One observation that ties them together — what kind of week / month
    this looks like, in one sentence.

Length: 2-4 paragraphs. Stop when you've said what's true.
"""


# ---------------------------------------------------------------------------
# self.md — the slower synthesis
# ---------------------------------------------------------------------------


def compose_self_md(
    cfg: LocalConfig,
    *,
    user_id: str,
    time_md: str,
    prior_self: Optional[str] = None,
) -> str:
    """The slow self-portrait. Reads time.md + any prior self.md and
    REVISES — does not start fresh. The first run has no prior; later
    runs are diffs against what was true last week."""

    # Strip our own generated-by header from time_md and prior_self before
    # passing to the model — otherwise the header gets re-included in
    # subsequent regenerations.
    time_md_clean = _strip_gen_header(time_md)
    user_msg = "Current portrait of how the user spends time:\n\n" + time_md_clean
    if prior_self:
        user_msg += (
            "\n\n---\n\nPrevious self.md (what was true on the last run):\n\n"
            + _strip_gen_header(prior_self).strip()
            + "\n\n---\n\nRevise. Keep what's still true. Update what has "
            + "changed. Note what's new. Do not invent."
        )
    text = _ask_llm(
        cfg,
        system=_SELF_SYSTEM,
        user=user_msg,
        max_tokens=600,
    )
    path = _portrait_dir(cfg, user_id).parent / SELF_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"<!-- generated by pmc consolidate at {_now_iso()} -->\n\n"
    path.write_text(header + text.strip() + "\n")
    log.info("portrait: wrote self.md (%d chars)", len(text))
    return text


_SELF_SYSTEM = """\
You are writing a self portrait of the user — who they are right now,
based on what they actually do. This file is what other agents read
when they want to know the user.

Voice:
  - Second person, plain prose.
  - Specific to THIS person. Not generic. If you can't say something
    specific, don't say anything.
  - Present tense for what's true now. Past tense for what they used
    to do but stopped.
  - Three paragraphs maximum. Tight. Every sentence earns its place.

Cover:
  - Who they are at the level of work and creation (what they make)
  - Where they spend time and attention
  - One thing about how they live that's true from the data —
    a pattern, a rhythm, an orientation. Cite the evidence inline.

Never:
  - Moralize or recommend
  - Use "you should" or "you might want to"
  - Invent details not present in the input
  - Use marketing language ("passionate", "innovative", "thought leader")
"""


# ---------------------------------------------------------------------------
# whoami.txt — the bootstrap packet
# ---------------------------------------------------------------------------


def compose_whoami(
    cfg: LocalConfig,
    *,
    user_id: str,
    self_md: str,
    projects: list[ProjectCharacterization],
    attention: list[AttentionRow],
) -> str:
    """The 500-token packet the MCP server hands every connecting agent.
    Densest possible 'who is this person, what are they doing right now.'
    Composed from self.md + the top active rows. No LLM call — pure
    compute over already-characterized rows."""
    parts: list[str] = []
    parts.append(self_md.strip())
    parts.append("")
    parts.append("Active work right now:")
    for p in projects:
        if p.status == "active":
            parts.append(f"  - {p.name}: {p.one_liner}")
    if attention:
        parts.append("")
        parts.append("Where time goes (top apps, last 30 days):")
        for a in attention[:5]:
            parts.append(f"  - {a.label}: {a.minutes_30d} min ({a.share_pct}%)")
    parts.append("")
    parts.append(f"as_of: {_now_iso()}")

    text = "\n".join(parts).strip() + "\n"
    path = _portrait_dir(cfg, user_id).parent / WHOAMI_TXT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    log.info("portrait: wrote whoami.txt (%d chars)", len(text))
    return text


PICTURE_MD = "portrait/picture.md"


def compose_picture_md(
    cfg: LocalConfig,
    *,
    user_id: str,
    self_md: str,
    time_md: str,
    reading_md: str,
    recent_md: str,
    agenda_md: str = "",
    prior_picture: Optional[str] = None,
    other_signals: str = "",
) -> str:
    """The canonical top-level output: 'here is how the user seems to
    spend their time' + drill-down by dimension + what's in motion +
    what could be learned. The file every connecting agent reads first.

    If `prior_picture` is given, the LLM is asked to DEEPEN the existing
    picture rather than write fresh — preserves accumulated insight and
    keeps each pass enriching the graph rather than wiping it.
    """
    sections = [
        f"WHO THEY ARE:\n{self_md.strip()}" if self_md.strip() else "",
        f"HOW THEY SPEND TIME:\n{time_md.strip()}" if time_md.strip() else "",
        f"WHAT THEY READ:\n{reading_md.strip()}" if reading_md.strip() else "",
        f"WHAT THEY ACTUALLY DID, LAST 7 DAYS:\n{recent_md.strip()}" if recent_md.strip() else "",
        f"OTHER LIFE SIGNALS (calendar / places / wallet / etc.):\n{other_signals.strip()}" if other_signals.strip() else "",
        f"AGENDA (in motion / learnable):\n{agenda_md.strip()}" if agenda_md.strip() else "",
    ]
    if prior_picture and prior_picture.strip():
        sections.append(
            "PRIOR PICTURE (your previous output — deepen it; keep what's "
            "still true, sharpen what was thin, add depth where the new "
            "evidence supports it; do not invent):\n"
            + prior_picture.strip()
        )
    evidence = "\n\n---\n\n".join(filter(None, sections))
    system = _PICTURE_SYSTEM_DEEPEN if prior_picture else _PICTURE_SYSTEM
    text = _ask_llm(cfg, system=system, user=evidence, max_tokens=2500)
    path = _portrait_dir(cfg, user_id).parent / PICTURE_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")
    log.info("portrait: wrote picture.md (%d chars)", len(text))
    return text


_PICTURE_SYSTEM_DEEPEN = """\
You are revising the user's canonical portrait. A prior version exists
(included as PRIOR PICTURE in the evidence). New evidence is in the
other sections.

Your job is to DEEPEN, not restart:
  - Keep claims from the prior picture that are still supported.
  - Sharpen claims that were vague — add specific numbers, dates,
    names, quotes from the new evidence.
  - Add NEW observations only where the evidence has gotten stronger
    or new patterns have emerged.
  - Demote or rewrite claims that look weaker given the new evidence.
  - Note explicitly anything that has changed since the last version.

Output the same shape as the prior picture (the headings + sections
below). Same voice rules. Do not invent. Do not lose what was good.

Voice:
  - Second person, plain prose. No marketing language. No "passionate".
  - Specific to THIS person. Cite numbers, names, places that appear in
    the evidence. Never invent.
  - Present tense for what's true now. Past tense for what they used
    to do but stopped.
"""


_PICTURE_SYSTEM = """\
You are writing the canonical portrait of how the user spends their
time. This is the FIRST thing every connecting agent reads.

Voice:
  - Second person, plain prose. No marketing language. No "passionate".
  - Specific to THIS person. Cite numbers, names, places that appear in
    the evidence. Never invent.
  - Present tense for what's true now. Past tense for what they used
    to do but stopped.

Shape (exactly this structure, no others):

  # How they seem to spend their time

  <2-3 short paragraphs at the HIGHEST LEVEL — what kind of life this
   person is currently living. Tie together work, attention, social,
   creation. This is the elevator pitch of who they are right now.>

  ## Their work

  <1-2 paragraphs naming what they're building or doing as work,
   characterizing the projects with status. Drill in: specific
   technologies, features, recent moves.>

  ## Their reading and learning

  <1-2 paragraphs. What topics they're actively exploring, derived
   from page-content topic clusters. Specific — not "tech" but
   "configuring MCP servers for personal data" or whatever the data
   actually shows.>

  ## How their attention is distributed

  <1 paragraph. Where the hours go, by category. Sleep window if
   detectable. Rhythm (morning person, night owl, etc.). Day-of-week
   pattern.>

  ## What's in motion right now

  <Numbered list of 4-8 SPECIFIC items the user has on their plate.
   Each item is one short paragraph: what it is, what evidence, what
   would close the loop. Drawn from the AGENDA actionables.>

  ## Patterns a personal agent could learn to handle

  <Numbered list of 3-6 items. Each is a repeating workflow the user
   does that an agent could be trained to do on their behalf. Drawn
   from the AGENDA learnables.>

Rules:
  - Never use markdown bold/italic — terminal renders the asterisks
    literally. Use prose for emphasis.
  - Filter out anything sensitive (medical, sexual, financial in
    detail). If something is marked "private" anywhere in the evidence,
    don't surface it.
  - This must work for ANY user. Don't assume they code, parent,
    travel, study, etc. Let the data say what's true.
  - Stop when you've said what's true. No "in summary" closing.
"""


__all__ = [
    "characterize_projects", "ProjectCharacterization",
    "compute_attention", "AttentionRow",
    "find_active_interests", "InterestRow",
    "compose_time_md",
    "compose_self_md",
    "compose_whoami",
    "compose_picture_md", "PICTURE_MD",
]
