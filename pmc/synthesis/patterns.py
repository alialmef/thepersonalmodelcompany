"""Pattern detection over the structured graph.

Threads name "what's in motion right now." Patterns name "what your
life keeps doing." Together they give the first-boot agent a real
picture of the user.

This pass runs entirely in Python over the graph store — no agent
calls, no LLM cost. Each pattern is a structured fact derived from
counting + clustering across the multi-source data. The agent
consumes patterns as context for future threads/right-now passes.

Output: <storage_root>/users/<uid>/graph/synth/patterns.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pmc.storage.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class Pattern:
    id: str
    category: str             # "travel" | "events" | "attention" | "places" | "curation" | "creation" | "filing"
    headline: str             # short second-person statement
    detail: str               # 1-2 sentences of context
    metric: dict[str, Any] = field(default_factory=dict)  # numbers backing the pattern
    examples: list[str] = field(default_factory=list)     # 1-5 representative items


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _synth_dir(storage_root: Path | str, user_id: str) -> Path:
    p = Path(storage_root) / "users" / user_id / "graph" / "synth"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _patterns_path(storage_root: Path | str, user_id: str) -> Path:
    return _synth_dir(storage_root, user_id) / "patterns.jsonl"


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------


def _travel_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """Wallet boarding passes → flight count, top routes, top airlines.

    The Wallet extractor stores each boarding pass as a Project with
    sources=["wallet:boardingPass"]. Headlines look like "Delta · JFK → PHX".
    """
    routes: Counter[tuple[str, str]] = Counter()
    airlines: Counter[str] = Counter()
    years: Counter[str] = Counter()
    samples: list[str] = []
    flights = 0

    for p in graph_store.iter_entities(user_id, "project"):
        sources = p.get("sources") or []
        if not any("wallet:boardingPass" in s for s in sources):
            continue
        flights += 1
        name = p.get("name") or ""
        # name = "<airline> · <depart> → <dest>"
        if " · " in name:
            airline, rest = name.split(" · ", 1)
            airlines[airline.strip()] += 1
            if " → " in rest:
                a, b = rest.split(" → ", 1)
                routes[(a.strip(), b.strip())] += 1
        if p.get("last_activity"):
            try:
                years[str(p["last_activity"])[:4]] += 1
            except Exception:
                pass
        if len(samples) < 5 and name:
            samples.append(name)

    if flights < 3:
        return None

    top_routes = routes.most_common(3)
    top_airlines = airlines.most_common(3)
    headline = f"You've taken {flights} flights on record."
    detail_parts: list[str] = []
    if top_airlines:
        air = ", ".join(f"{a} ({n})" for a, n in top_airlines)
        detail_parts.append(f"Most-flown: {air}.")
    if top_routes:
        r = ", ".join(f"{a}→{b} (×{n})" for (a, b), n in top_routes)
        detail_parts.append(f"Top routes: {r}.")
    return Pattern(
        id=f"travel_flights_{flights}",
        category="travel",
        headline=headline,
        detail=" ".join(detail_parts),
        metric={
            "flights": flights,
            "top_routes": [{"from": a, "to": b, "count": n} for (a, b), n in top_routes],
            "top_airlines": [{"airline": a, "count": n} for a, n in top_airlines],
            "by_year": dict(years),
        },
        examples=samples,
    )


def _events_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """Wallet event tickets → venue/category clustering."""
    venues: Counter[str] = Counter()
    samples: list[str] = []
    events = 0

    for p in graph_store.iter_entities(user_id, "project"):
        sources = p.get("sources") or []
        if not any("wallet:eventTicket" in s for s in sources):
            continue
        events += 1
        name = p.get("name") or ""
        if " · " in name:
            org, _ = name.split(" · ", 1)
            venues[org.strip()] += 1
        if len(samples) < 5 and name:
            samples.append(name)

    if events < 3:
        return None

    top_venues = venues.most_common(3)
    headline = f"You've been to {events} ticketed events on record."
    detail = ""
    if top_venues:
        v = ", ".join(f"{a} (×{n})" for a, n in top_venues)
        detail = f"Most-frequent organizers: {v}."
    return Pattern(
        id=f"events_total_{events}",
        category="events",
        headline=headline,
        detail=detail,
        metric={
            "events": events,
            "top_venues": [{"venue": v, "count": n} for v, n in top_venues],
        },
        examples=samples,
    )


def _attention_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """Screen Time → top apps by 30d minutes + a quick category split."""
    rows: list[tuple[str, str, int, str]] = []  # (name, bundle, min30d, category)
    for a in graph_store.iter_entities(user_id, "app"):
        rows.append((
            a.get("display_name") or a.get("bundle_id") or "?",
            a.get("bundle_id") or "?",
            int(a.get("minutes_30d") or 0),
            a.get("category") or "other",
        ))
    rows.sort(key=lambda r: -r[2])
    rows = [r for r in rows if r[2] > 0]
    if not rows:
        return None

    total = sum(r[2] for r in rows)
    top5 = rows[:5]
    by_cat: Counter[str] = Counter()
    for _, _, m, c in rows:
        by_cat[c] += m

    headline = f"You spent {total} minutes (~{total // 60}h) on tracked apps in the last 30 days."
    top_str = ", ".join(f"{n} ({m}m)" for n, _, m, _ in top5)
    cat_str = ", ".join(f"{c} {m}m" for c, m in by_cat.most_common(3))
    detail = f"Top apps: {top_str}. By category: {cat_str}."
    return Pattern(
        id=f"attention_30d_{total}",
        category="attention",
        headline=headline,
        detail=detail,
        metric={
            "total_minutes_30d": total,
            "top_apps": [
                {"name": n, "bundle_id": b, "minutes_30d": m, "category": c}
                for n, b, m, c in top5
            ],
            "by_category_minutes_30d": dict(by_cat),
        },
        examples=[n for n, _, _, _ in top5],
    )


def _places_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """Themes from photo concepts (categories 1/2/3 = streets/localities/regions)
    → most-photographed places. These are 'where your camera takes you',
    which is a strong proxy for where you spend meaningful time."""
    place_themes: list[tuple[str, int]] = []
    for t in graph_store.iter_entities(user_id, "theme"):
        kws = t.get("keywords") or []
        if not any(str(k).startswith("place_") for k in kws):
            continue
        mentions = int(t.get("mentions_180d") or 0)
        label = (t.get("label") or "").strip()
        if not label:
            continue
        place_themes.append((label, mentions))

    if len(place_themes) < 5:
        return None
    place_themes.sort(key=lambda r: -r[1])
    top10 = place_themes[:10]
    headline = f"You take a lot of photos in {len(place_themes)} distinct places."
    detail = "Top photographed locations (from Apple's photo geotagging): " + ", ".join(
        f"{lbl} (×{n})" for lbl, n in top10[:5]
    )
    return Pattern(
        id=f"places_photographed_{len(place_themes)}",
        category="places",
        headline=headline,
        detail=detail,
        metric={"distinct_locations": len(place_themes), "top": [
            {"label": lbl, "photo_mentions": n} for lbl, n in top10
        ]},
        examples=[lbl for lbl, _ in top10[:5]],
    )


def _curation_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """Bookmarks + Reading List → what you've explicitly saved.
    Detects domain concentration."""
    domains: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    examples: list[str] = []
    total = 0
    for t in graph_store.iter_entities(user_id, "taste"):
        total += 1
        creator = (t.get("creator") or "").strip()
        if creator:
            domains[creator] += 1
        by_kind[t.get("kind") or "?"] += 1
        if len(examples) < 5 and (t.get("name") or "").strip():
            examples.append(t["name"])

    if total < 5:
        return None
    top_domains = domains.most_common(5)
    headline = f"You've saved {total} bookmarks + reading-list items across the years."
    detail = f"Kind split: {dict(by_kind)}. " + (
        f"Most-saved domains: {', '.join(f'{d} (×{n})' for d, n in top_domains)}."
        if top_domains else ""
    )
    return Pattern(
        id=f"curation_total_{total}",
        category="curation",
        headline=headline,
        detail=detail,
        metric={
            "total_items": total,
            "by_kind": dict(by_kind),
            "top_domains": [{"domain": d, "count": n} for d, n in top_domains],
        },
        examples=examples,
    )


def _creation_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """Voice memo file count + duration distribution → 'how often you talk
    to yourself'. Pulled from the transcript manifest if available, falls
    back to file count from FileSignal entries."""
    voice_count = 0
    transcript_count = 0
    char_total = 0
    storage_root = Path(graph_store.storage_root)
    manifest = storage_root / "users" / user_id / "graph" / "synth" / "transcripts" / "_manifest.jsonl"
    if manifest.is_file():
        for line in manifest.open():
            try:
                v = json.loads(line)
                transcript_count += 1
                char_total += int(v.get("text_chars") or 0)
            except Exception:
                continue
    # Also count voice_memo FileSignal entries
    for f in graph_store.iter_entities(user_id, "file"):
        if f.get("kind") == "voice_memo":
            voice_count += 1

    if voice_count < 5:
        return None
    headline = f"You've made {voice_count} voice memos."
    if transcript_count > 0:
        headline += f" {transcript_count} transcribed."
    detail = ""
    if char_total > 0:
        avg = char_total // max(transcript_count, 1)
        detail = f"Total {char_total:,} characters of transcribed thought (~{avg} chars per memo)."
    return Pattern(
        id=f"creation_voice_memos_{voice_count}",
        category="creation",
        headline=headline,
        detail=detail,
        metric={
            "voice_memos": voice_count,
            "transcripts": transcript_count,
            "transcript_chars_total": char_total,
        },
        examples=[],
    )


def _filing_pattern(graph_store: GraphStore, user_id: str) -> Pattern | None:
    """iCloud Drive walker → semantic kind counts. Surfaces 'what's in
    the user's filing cabinet': how many passports, contracts, rent
    statements, etc."""
    kinds: Counter[str] = Counter()
    examples_per_kind: dict[str, list[str]] = defaultdict(list)
    icloud_count = 0
    for f in graph_store.iter_entities(user_id, "file"):
        path = f.get("path") or ""
        if "Mobile Documents" not in path:
            continue
        icloud_count += 1
        k = f.get("kind") or "other"
        kinds[k] += 1
        if k in ("passport", "license", "receipt", "rent", "contract", "interview", "resume", "statement", "invoice"):
            if len(examples_per_kind[k]) < 2:
                name = f.get("name") or ""
                if name:
                    examples_per_kind[k].append(name)

    flagged = sum(n for k, n in kinds.items() if k not in ("other", "code", "document", "image"))
    if icloud_count < 10:
        return None

    headline = f"Your iCloud Drive has {icloud_count} files."
    detail_parts: list[str] = []
    high_signal = [
        (k, kinds[k]) for k in ("passport", "rent", "contract", "resume", "receipt", "license", "invoice")
        if kinds.get(k, 0) > 0
    ]
    if high_signal:
        detail_parts.append(
            "Flagged document kinds: "
            + ", ".join(f"{k} (×{n})" for k, n in high_signal)
            + "."
        )
    examples: list[str] = []
    for k, names in examples_per_kind.items():
        for n in names[:1]:
            examples.append(f"{k}: {n}")
            if len(examples) >= 5:
                break
        if len(examples) >= 5:
            break

    return Pattern(
        id=f"filing_icloud_{icloud_count}",
        category="filing",
        headline=headline,
        detail=" ".join(detail_parts),
        metric={
            "icloud_files": icloud_count,
            "by_kind": dict(kinds),
            "high_signal_count": flagged,
        },
        examples=examples,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_DETECTORS = [
    _travel_pattern,
    _events_pattern,
    _attention_pattern,
    _places_pattern,
    _curation_pattern,
    _creation_pattern,
    _filing_pattern,
]


def build_patterns(
    *,
    graph_store: GraphStore,
    storage_root: Path | str,
    user_id: str,
) -> list[Pattern]:
    """Run all pattern detectors, persist, return."""
    patterns: list[Pattern] = []
    for detector in _DETECTORS:
        try:
            p = detector(graph_store, user_id)
        except Exception:
            continue
        if p is not None:
            patterns.append(p)
    _write_patterns(storage_root, user_id, patterns)
    return patterns


def load_patterns(storage_root: Path | str, user_id: str) -> list[Pattern]:
    p = _patterns_path(storage_root, user_id)
    if not p.is_file():
        return []
    out: list[Pattern] = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(Pattern(**d))
        except Exception:
            continue
    return out


def _write_patterns(storage_root: Path | str, user_id: str, patterns: list[Pattern]) -> None:
    import os
    p = _patterns_path(storage_root, user_id)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        for pat in patterns:
            f.write(json.dumps(asdict(pat), default=str) + "\n")
    os.replace(tmp, p)
