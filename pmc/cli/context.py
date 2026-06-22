"""Build the agent-facing context block for `pmc chat`.

Design principle: big capacity, small attention. The model holds
everything we know; it should speak like a person — pulling only
currently-in-motion entries when characterizing who the user IS right
now. Historical data stays in the prompt (the agent can find it if
asked) but it's quarantined into a second band so it can't be
mistaken for current behavior.

Two-band layout per kind:

   ## currently active
   [active]  Sam · last_seen=2026-06-22
   [active]  Beza · last_seen=2026-06-20
   [new]     Joseph · first appearance in last week

   ## historical (not currently active)
   [stable]  Marcus · last_seen=2026-04-15
   [dormant] Lisa · last_seen=2024-08-09

Plus a strict instruction at the top of the block: any current-
behavior characterization MUST cite an [active] or [new] entry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmc.cli.temporal import ACTIVE_WINDOW_DAYS, status_for
from pmc.storage.graph_store import GraphStore, summarize_for_agent
from pmc.synthesis import (
    load_causal,
    load_drift,
    load_patterns,
    load_threads,
    load_transcripts,
)


def build_context(
    storage_root: Path | str,
    user_id: str,
    *,
    max_threads: int = 8,
    max_patterns: int = 6,
    max_drift: int = 6,
    max_causal: int = 4,
    max_transcripts: int = 5,
) -> str:
    """Return a compact text block describing the user's graph + synthesis,
    with temporal status tags + two-band split."""
    store = GraphStore(storage_root)
    sections: list[str] = []

    sections.append(_header(storage_root, user_id))

    summary = summarize_for_agent(store, user_id)
    sections.append(_counts_section(summary["counts"]))
    sections.append(_people_section(summary["sample"].get("people", [])))
    sections.append(_places_section(summary["sample"].get("places", [])))
    sections.append(_themes_section(summary["sample"].get("themes", [])))
    sections.append(_projects_section(summary["sample"].get("projects", [])))
    sections.append(_apps_section(summary["sample"].get("apps", [])))
    sections.append(_open_loops_section(summary["sample"].get("open_loops", [])))

    sections.append(_threads_section(load_threads(storage_root, user_id), max_threads))
    sections.append(_patterns_section(load_patterns(storage_root, user_id), max_patterns))
    sections.append(_drift_section(load_drift(storage_root, user_id), max_drift))
    sections.append(_causal_section(load_causal(storage_root, user_id), max_causal))
    sections.append(_transcripts_section(load_transcripts(storage_root, user_id), max_transcripts))

    return "\n\n".join(s for s in sections if s).strip() + "\n"


# ---------------------------------------------------------------------------
# Header — the temporal instruction the model reads first
# ---------------------------------------------------------------------------


def _header(storage_root: Path | str, user_id: str) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        "# PERSONAL CONTEXT\n"
        f"as_of: {now}\n"
        f"user_id: {user_id}\n"
        f"storage_root: {storage_root}\n"
        "\n"
        "READ THIS FIRST. Every entity is tagged with a temporal status:\n"
        f"  [active]  touched in the last {ACTIVE_WINDOW_DAYS} days — currently in motion\n"
        "  [new]     first appeared in the last 7 days\n"
        "  [stable]  touched in the last 90 days but not active right now\n"
        "  [dormant] older than 90 days; biographical, not a current trait\n"
        "  [unknown] no recency information\n"
        "\n"
        "When the user asks what they are doing / who they are / what they care "
        "about RIGHT NOW, you may ONLY cite [active] or [new] entries.\n"
        "[stable] entries may be referenced if directly relevant to the topic.\n"
        "[dormant] entries are biography. NEVER surface them as a current trait. "
        "(For example: 'you record voice memos' is wrong if voice_memos is [dormant] — "
        "the correct phrasing is 'you used to record voice memos but have stopped.')\n"
        "\n"
        "Anything outside this context block: you do not know. Say so plainly."
    )


# ---------------------------------------------------------------------------
# Section builders — each row gets a [status] tag
# ---------------------------------------------------------------------------


def _counts_section(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    rows = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    return f"## graph counts\n{rows}" if rows else ""


def _tagged(entity: dict[str, Any], kind: str, body: str) -> str:
    return f"[{status_for(entity, kind=kind)}] {body}"


def _people_section(people: list[dict[str, Any]]) -> str:
    if not people:
        return ""
    lines = ["## people"]
    for p in people:
        name = p.get("display_name") or ""
        aliases = ", ".join(p.get("aliases") or [])
        last = p.get("last_seen") or ""
        bits = [b for b in [name, aliases, f"last_seen={last}" if last else ""] if b]
        lines.append(_tagged(p, "person", " · ".join(bits)))
    return "\n".join(_render_lines(lines))


def _places_section(places: list[dict[str, Any]]) -> str:
    if not places:
        return ""
    lines = ["## places"]
    for p in places:
        label = p.get("label") or ""
        kind = p.get("kind") or ""
        visits = p.get("visits") or ""
        last = p.get("last_seen") or ""
        bits = [b for b in [
            label,
            f"({kind})" if kind else "",
            f"visits={visits}" if visits else "",
            f"last={last}" if last else "",
        ] if b]
        lines.append(_tagged(p, "place", " · ".join(bits)))
    return "\n".join(_render_lines(lines))


def _themes_section(themes: list[dict[str, Any]]) -> str:
    if not themes:
        return ""
    lines = ["## themes"]
    for t in themes:
        label = t.get("label") or ""
        traj = t.get("trajectory") or ""
        m30 = t.get("mentions_30d") or 0
        bits = [b for b in [
            label,
            f"trajectory={traj}" if traj else "",
            f"mentions_30d={m30}" if m30 else "",
        ] if b]
        # Themes with mentions_30d == 0 are inherently dormant regardless
        # of their last_seen. Use that as a stronger signal.
        if not m30:
            t = {**t, "last_seen": ""}  # force dormant fallback
        lines.append(_tagged(t, "theme", " · ".join(bits)))
    return "\n".join(_render_lines(lines))


def _projects_section(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return ""
    lines = ["## repos"]
    for r in projects:
        name = r.get("name") or ""
        lang = r.get("language") or ""
        last = r.get("last_commit") or ""
        c30 = r.get("commit_count_30d") or 0
        bits = [b for b in [
            name,
            f"({lang})" if lang else "",
            f"last={last}" if last else "",
            f"commits_30d={c30}" if c30 else "",
        ] if b]
        lines.append(_tagged(r, "repo", " · ".join(bits)))
    return "\n".join(_render_lines(lines))


def _apps_section(apps: list[dict[str, Any]]) -> str:
    if not apps:
        return ""
    # Apps already report `minutes_30d` — if zero, dormant. Otherwise active.
    lines = ["## apps (where attention goes)"]
    for a in apps:
        name = a.get("display_name") or a.get("bundle_id") or ""
        cat = a.get("category") or ""
        mins = a.get("minutes_30d") or 0
        if not mins:
            continue  # already-zero apps add nothing
        bits = [name, f"({cat})" if cat else "", f"min_30d={mins}"]
        # Stamp as active by default — non-zero minutes in last 30d is the
        # definition of active for this kind.
        lines.append(f"[active] " + " · ".join(b for b in bits if b))
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _open_loops_section(loops: list[dict[str, Any]]) -> str:
    if not loops:
        return ""
    lines = ["## open loops"]
    for l in loops:
        title = l.get("title") or l.get("label") or l.get("summary") or l.get("excerpt") or ""
        title = title.strip().replace("\n", " ")
        if len(title) > 160:
            title = title[:160] + "…"
        first = l.get("first_seen") or l.get("opened_at") or ""
        if not title:
            continue
        bits = [title]
        if first:
            bits.append(f"({first})")
        lines.append(_tagged(l, "open_loop", " ".join(bits)))
    if len(lines) == 1:
        return ""
    return "\n".join(_render_lines(lines))


def _threads_section(threads: list[Any], limit: int) -> str:
    if not threads:
        return ""
    lines = ["## live threads (agent-named, what's in motion)"]
    for t in threads[:limit]:
        head = getattr(t, "headline", "")
        body = getattr(t, "body", "")
        urg = getattr(t, "urgency", "")
        kind = getattr(t, "kind", "")
        # Threads with urgency "now" or "this_week" → active.
        # "soon" → stable. "someday" → dormant.
        tag = {
            "now": "active", "this_week": "active",
            "soon": "stable", "someday": "dormant",
        }.get(urg, "stable")
        prefix = f"[{tag}] "
        bits = [head, f"({kind}/{urg})" if kind or urg else ""]
        lines.append(prefix + " · ".join(b for b in bits if b))
        if body:
            lines.append(f"  {body}")
    return "\n".join(lines)


def _patterns_section(patterns: list[Any], limit: int) -> str:
    if not patterns:
        return ""
    lines = ["## patterns (steady state — what your life keeps doing)"]
    for p in patterns[:limit]:
        head = getattr(p, "headline", "")
        detail = getattr(p, "detail", "")
        cat = getattr(p, "category", "")
        # Patterns describe the steady state — they're stable by definition
        # unless explicitly tagged dormant by the consolidator (future).
        lines.append(f"[stable] [{cat}] {head}")
        if detail:
            lines.append(f"  {detail}")
    return "\n".join(lines)


def _drift_section(drifts: list[Any], limit: int) -> str:
    if not drifts:
        return ""
    lines = ["## drift (what's different about now vs the recent past)"]
    for d in drifts[:limit]:
        head = getattr(d, "headline", "")
        detail = getattr(d, "detail", "")
        cat = getattr(d, "category", "")
        # Drift entries are inherently about "now" — they describe
        # current state vs past. Always active.
        lines.append(f"[active] [{cat}] {head}")
        if detail:
            lines.append(f"  {detail}")
    return "\n".join(lines)


def _causal_section(items: list[Any], limit: int) -> str:
    if not items:
        return ""
    lines = ["## correlations (when X, Y tends to)"]
    for c in items[:limit]:
        when = getattr(c, "when", "")
        then = getattr(c, "then", "")
        conf = getattr(c, "confidence", "")
        lines.append(f"[stable] when {when} → {then} ({conf})")
    return "\n".join(lines)


def _transcripts_section(items: list[Any], limit: int) -> str:
    if not items:
        return ""
    items = items[:limit]
    lines = ["## recent voice memos (transcribed)"]
    for t in items:
        fid = getattr(t, "file_id", "")
        text = getattr(t, "text_excerpt", "") or ""
        text = text.strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "…"
        # Transcripts are tagged by whether the underlying memo is
        # recent. Most of yours are dormant. The transcripts module
        # doesn't carry a date on its rows so we default to stable —
        # the active vs dormant distinction for voice memos as a
        # behavior is handled at the apps / files level.
        lines.append(f"[stable] {fid}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Two-band rendering
# ---------------------------------------------------------------------------


def _render_lines(lines: list[str]) -> list[str]:
    """Given a `## title` followed by tagged rows, partition into
    active+new first, then stable, then dormant. Empty lines between
    bands. The first line (`## title`) is preserved unchanged."""
    if len(lines) <= 1:
        return lines
    title = lines[0]
    rest = lines[1:]
    active: list[str] = []
    stable: list[str] = []
    dormant: list[str] = []
    other: list[str] = []
    for line in rest:
        if line.startswith("[active]") or line.startswith("[new]"):
            active.append(line)
        elif line.startswith("[stable]"):
            stable.append(line)
        elif line.startswith("[dormant]"):
            dormant.append(line)
        else:
            other.append(line)

    out = [title]
    if active:
        out.extend(active)
    if stable:
        if active:
            out.append("")
        out.extend(stable)
    if dormant:
        if active or stable:
            out.append("")
        out.extend(dormant)
    if other:
        if active or stable or dormant:
            out.append("")
        out.extend(other)
    return out


__all__ = ["build_context"]
