"""Python reader for the personal knowledge graph written by the Rust
extractors.

This is the load-bearing missing piece that closes the Tauri-local /
backend disconnect. The Rust scheduler in `desktop/src/graph/store.rs`
writes typed entity JSONL files to:

    <storage_root>/users/<user_id>/graph/<entity_kind>.jsonl

where <entity_kind> is one of:
    people.jsonl, places.jsonl, events.jsonl, episodes.jsonl,
    projects.jsonl, themes.jsonl, open_loops.jsonl, taste.jsonl,
    files.jsonl, repos.jsonl, web.jsonl, app_usage.jsonl,
    shell.jsonl, notifications.jsonl, edges.jsonl

When the backend's storage_root matches `~/.pmc-dev/storage` (the
Rust default), they share the same files — Rust writes, Python reads.
For prod deploys where the backend lives on Railway, the Mac app
will push entities over HTTP (next commit); the on-disk schema stays
identical so both paths use this same reader.

This module is *read-only* from the Python side. Writes still happen
via the Rust scheduler. The future agent-driven synthesis layer
(entity resolution + theme detection) will write to a SEPARATE
`graph/synth/` directory so we never clobber what the extractors
produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


# Mirror of `EntityKind::filename()` in desktop/src/graph/schema.rs.
# If you add a kind on the Rust side, add it here too.
ENTITY_FILENAMES: dict[str, str] = {
    "person":              "people.jsonl",
    "place":               "places.jsonl",
    "event":               "events.jsonl",
    "episode":             "episodes.jsonl",
    "project":             "projects.jsonl",
    "theme":               "themes.jsonl",
    "open_loop":           "open_loops.jsonl",
    "taste":               "taste.jsonl",
    "file":                "files.jsonl",
    "repo":                "repos.jsonl",
    "web":                 "web.jsonl",
    "app":                 "app_usage.jsonl",
    "shell":               "shell.jsonl",
    "notification":        "notifications.jsonl",
    "edge":                "edges.jsonl",
}


# Subset that's "node-like" — used for graph snapshot rendering. Edges
# are treated separately because they reference node ids.
NODE_KINDS: tuple[str, ...] = (
    "person", "place", "event", "episode", "project", "theme",
    "open_loop", "taste", "file", "repo", "web", "app", "shell",
    "notification",
)


class GraphStore:
    """Per-user typed-entity store. Wraps the JSONL layout the Rust
    extractors write to."""

    def __init__(self, storage_root: Path | str) -> None:
        self.storage_root = Path(storage_root)

    # ---- paths ---------------------------------------------------------

    def _graph_dir(self, user_id: str) -> Path:
        return self.storage_root / "users" / user_id / "graph"

    def _file(self, user_id: str, kind: str) -> Path:
        fname = ENTITY_FILENAMES[kind]
        return self._graph_dir(user_id) / fname

    def exists(self, user_id: str) -> bool:
        return self._graph_dir(user_id).is_dir()

    # ---- reads ---------------------------------------------------------

    def iter_entities(self, user_id: str, kind: str) -> Iterator[dict[str, Any]]:
        """Yield every entity of `kind` for `user_id`. Each line is one
        JSON object. Lines that fail to parse are skipped silently —
        they're typically partial writes during an in-progress scheduler
        run, and the next poll picks them up cleanly."""
        if kind not in ENTITY_FILENAMES:
            return
        p = self._file(user_id, kind)
        if not p.is_file():
            return
        try:
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return

    def load_entities(self, user_id: str, kind: str) -> list[dict[str, Any]]:
        return list(self.iter_entities(user_id, kind))

    def count(self, user_id: str, kind: str) -> int:
        return sum(1 for _ in self.iter_entities(user_id, kind))

    def counts(self, user_id: str) -> dict[str, int]:
        """Per-kind counts. Cheap — scans line-by-line, doesn't parse."""
        out: dict[str, int] = {}
        for kind, fname in ENTITY_FILENAMES.items():
            p = self._file(user_id, kind)
            if not p.is_file():
                out[kind] = 0
                continue
            try:
                n = 0
                with p.open("rb") as f:
                    for _ in f:
                        n += 1
                out[kind] = n
            except OSError:
                out[kind] = 0
        return out

    def quality_counts(self, user_id: str) -> dict[str, int]:
        """Like `counts()` but filtered to the quality tier the agent
        actually surfaces. Today: Persons must pass `_is_quality_person`
        (drops the empty + phone-fragment-only records). Other kinds
        pass through unchanged; their per-extractor filters are already
        stricter."""
        out: dict[str, int] = {}
        for kind in ENTITY_FILENAMES:
            if kind == "person":
                n = sum(1 for v in self.iter_entities(user_id, kind) if _is_quality_person(v))
                out[kind] = n
            else:
                out[kind] = self.count(user_id, kind)
        return out

    def total_node_count(self, user_id: str) -> int:
        return sum(self.count(user_id, k) for k in NODE_KINDS)

    # ---- snapshot for the memory-web visualization --------------------

    def snapshot(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return `{nodes, edges}` in the same shape as the Tauri-side
        `graph_snapshot` command. Used by the browser frontend (which
        can't invoke Tauri commands) to render the memory web.

        Nodes: stable id + kind + optional weight hint. No labels, no
        content — same privacy posture as the Rust side.
        """
        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for kind in NODE_KINDS:
            for v in self.iter_entities(user_id, kind):
                eid = v.get("id")
                if not isinstance(eid, str) or not eid:
                    continue
                weight = _weight_hint(v)
                nodes.append({"id": eid, "kind": kind, "weight": weight})
                node_ids.add(eid)

        edges: list[dict[str, Any]] = []
        for e in self.iter_entities(user_id, "edge"):
            eid = e.get("id") or ""
            src = e.get("from_id") or ""
            tgt = e.get("to_id") or ""
            ek = e.get("kind") or ""
            if not (eid and src and tgt):
                continue
            if src not in node_ids or tgt not in node_ids:
                continue
            edges.append({"id": eid, "source": src, "target": tgt, "kind": ek})

        return {"nodes": nodes, "edges": edges}

    # ---- per-kind sample (for synthesis / debugging) ------------------

    def sample(
        self, user_id: str, kind: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for v in self.iter_entities(user_id, kind):
            out.append(v)
            if len(out) >= limit:
                break
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _weight_hint(v: dict[str, Any]) -> float | None:
    """Best-effort node-weight extraction. The renderer uses this to
    scale node radius — bigger circles for entities you have more
    signal about. Falls back to None if no obvious field exists."""
    for k in (
        "visit_count", "visits_180d", "play_count",
        "minutes_180d", "count_180d", "mentions_180d",
        "commit_count_30d",
    ):
        x = v.get(k)
        if isinstance(x, (int, float)):
            import math
            return min(1.0, math.log1p(float(x)) / 12.0)
    return None


def _is_quality_person(v: dict[str, Any]) -> bool:
    """A Person worth showing the agent has at least one signal-bearing
    name field. Drops the empty `display_name: '' aliases: ['']` records
    that some extractors emit when their row has a blank handle, and
    the phone-fragment-only records (`aliases: ['48267']`) that the
    agent shouldn't surface as 'people you know'."""
    import re as _re
    PHONEFRAG = _re.compile(r"^[\d\s\(\)\-\+]+$")
    name = (v.get("display_name") or "").strip()
    if name:
        return True
    aliases = [a.strip() for a in (v.get("aliases") or []) if isinstance(a, str) and a.strip()]
    emails = [e for e in (v.get("emails") or []) if isinstance(e, str) and e.strip()]
    # Email-only person — keep (they have an addressable identity)
    if emails:
        return True
    # Phone-fragment-only aliases — drop (not surface-worthy)
    if aliases and all(PHONEFRAG.match(a) for a in aliases):
        return False
    # Has a meaningful alias (not empty, not phone-fragment-only)
    return bool(aliases)


def summarize_for_agent(
    store: GraphStore, user_id: str, *, per_kind_limit: int = 8,
) -> dict[str, Any]:
    """Build a compact, agent-readable summary of the user's graph for
    the synthesis pass. Returns counts + a small sample per kind with
    just the *most identifying* fields. The agent uses this to generate
    /confirm claims and /right-now observations.

    The shape is deliberately lossy. The full graph would blow context
    budgets for any of the providers we support. This gets the agent
    enough to reason about "who", "where", "what", "when" without
    drowning it.

    People are filtered through `_is_quality_person` before sampling —
    the raw graph has thousands of empty / phone-fragment-only Persons
    that would otherwise eat the agent's context budget with noise.
    """
    counts = store.counts(user_id)
    sample: dict[str, list[dict[str, Any]]] = {}

    # People — only quality-passing rows. Walk the file lazily and stop
    # once we have enough so we don't load all 6k records when 8 suffice.
    people_sample: list[dict[str, Any]] = []
    for v in store.iter_entities(user_id, "person"):
        if not _is_quality_person(v):
            continue
        people_sample.append({
            "id": v.get("id"),
            "display_name": v.get("display_name"),
            "aliases": (v.get("aliases") or [])[:3],
            "last_seen": v.get("last_seen"),
        })
        if len(people_sample) >= per_kind_limit:
            break
    sample["people"] = people_sample

    # Places — label, kind, last visit
    sample["places"] = [
        {
            "id": v.get("id"),
            "label": v.get("label"),
            "kind": v.get("kind"),
            "visits": v.get("visit_count"),
            "last_seen": v.get("last_seen"),
        }
        for v in store.sample(user_id, "place", per_kind_limit)
    ]

    # Themes — label, trajectory, recency
    sample["themes"] = [
        {
            "id": v.get("id"),
            "label": v.get("label"),
            "trajectory": v.get("trajectory"),
            "mentions_30d": v.get("mentions_30d"),
            "last_seen": v.get("last_seen"),
        }
        for v in store.sample(user_id, "theme", per_kind_limit)
    ]

    # Projects — name + active recency
    sample["projects"] = [
        {
            "id": v.get("id"),
            "name": v.get("name"),
            "language": v.get("language"),
            "last_commit": v.get("last_commit"),
            "commit_count_30d": v.get("commit_count_30d"),
        }
        for v in store.sample(user_id, "repo", per_kind_limit)
    ]

    # Open loops — what's unresolved
    sample["open_loops"] = [
        {
            "id": v.get("id"),
            "title": v.get("title") or v.get("label"),
            "first_seen": v.get("first_seen"),
        }
        for v in store.sample(user_id, "open_loop", per_kind_limit)
    ]

    # App usage — what time goes to
    sample["apps"] = [
        {
            "bundle_id": v.get("bundle_id"),
            "display_name": v.get("display_name"),
            "category": v.get("category"),
            "minutes_30d": v.get("minutes_30d"),
        }
        for v in store.sample(user_id, "app", per_kind_limit)
    ]

    return {"counts": counts, "sample": sample}
