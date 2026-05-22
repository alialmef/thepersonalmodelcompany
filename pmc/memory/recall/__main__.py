"""CLI for the recall layer.

Usage:
    python -m pmc.memory.recall migrate    --user UID [--storage PATH]
    python -m pmc.memory.recall consolidate --user UID [--storage PATH] [--limit N]
    python -m pmc.memory.recall working    --user UID [--storage PATH]
    python -m pmc.memory.recall narrative  --user UID [--storage PATH]
    python -m pmc.memory.recall retrieve   --user UID [--storage PATH] --query "..." [--k 10]
    python -m pmc.memory.recall stats      --user UID [--storage PATH]

Default storage path: $PMC_DEV_ROOT/storage or ~/.pmc-dev/storage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pmc.memory.recall import RecallStore
from pmc.memory.recall.consolidate import Consolidator
from pmc.memory.recall.migrate import build_episodes, raw_text_for_episode, storage_paths
from pmc.memory.recall.narrative import build_narrative
from pmc.memory.recall.preview import preview_consolidate
from pmc.memory.recall.retrieve import retrieve, RetrievalScope
from pmc.memory.recall.working import build_working_memory


DEFAULT_USER = "11c7ace3-f395-4353-8acb-d6f7a2ec6113"


def _storage_root() -> Path:
    return Path(os.environ.get("PMC_DEV_ROOT", str(Path.home() / ".pmc-dev"))) / "storage"


def cmd_migrate(args) -> int:
    counts = build_episodes(args.storage, args.user)
    print("Migrated episodes per source:")
    for k, v in counts.items():
        print(f"  {k:<10} {v}")
    return 0


def cmd_preview(args) -> int:
    paths = storage_paths(args.storage, args.user)
    store = RecallStore(paths["recall"])
    started = time.time()
    def lookup(ep):
        return raw_text_for_episode(args.storage, args.user, ep)
    n = preview_consolidate(store, lookup, limit=args.limit)
    dur = time.time() - started
    print(f"Preview-consolidated {n} episodes in {dur:.1f}s (heuristic + local embeddings)")
    print("\n--- store stats ---")
    print(json.dumps(store.stats(), indent=2))
    print("\nNext step: set ANTHROPIC_API_KEY and run consolidate to upgrade")
    print("preview summaries to Claude-produced ones with state-change detection.")
    store.close()
    return 0


def cmd_consolidate(args) -> int:
    paths = storage_paths(args.storage, args.user)
    store = RecallStore(paths["recall"])
    c = Consolidator(
        model=args.model,
        max_episodes_per_run=args.limit,
        max_concurrency=args.concurrency,
    )
    started = time.time()
    def lookup(ep):
        return raw_text_for_episode(args.storage, args.user, ep)
    last_print = [time.time()]
    def progress(done, total):
        if time.time() - last_print[0] >= 5.0 or done == total:
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] {rate:.1f}/s  ETA {eta/60:.1f}m", flush=True)
            last_print[0] = time.time()
    results = c.consolidate_pending(store, lookup, progress=progress)
    dur = time.time() - started
    print(f"Consolidated {len(results)} episodes in {dur:.1f}s ({(dur/len(results) if results else 0):.2f}s/ep)")
    if args.sample > 0:
        print("\n--- sample summaries ---")
        for r in results[: args.sample]:
            print(f"\n[{r.episode_id}] tone={r.emotional_tone} imp={r.importance:.2f}")
            print(f"  {r.summary}")
            if r.topics:
                print(f"  topics: {', '.join(r.topics)}")
            if r.entity_states:
                print(f"  state changes: {len(r.entity_states)}")
    print("\n--- store stats ---")
    print(json.dumps(store.stats(), indent=2))
    store.close()
    return 0


def cmd_working(args) -> int:
    paths = storage_paths(args.storage, args.user)
    store = RecallStore(paths["recall"])
    snap = build_working_memory(store, model=args.model)
    print(f"Working memory snapshot ({snap.produced_by}):")
    print(f"  Anticipation:")
    for item in snap.anticipation:
        print(f"    - {item}")
    print(f"  Top open loops:")
    for ol in snap.top_open_loops[:5]:
        print(f"    - {ol}")
    print(f"  Hot people:")
    for p in snap.hot_people[:5]:
        print(f"    - {p}")
    store.close()
    return 0


def cmd_narrative(args) -> int:
    paths = storage_paths(args.storage, args.user)
    store = RecallStore(paths["recall"])
    snap = build_narrative(store, model=args.model)
    print(f"Narrative for {snap.snapshot_month} ({snap.produced_by}):")
    print(f"\n--- eras ---")
    for era in snap.eras:
        print(f"  {era.get('label')} ({era.get('start')}-{era.get('end')})")
        print(f"    {era.get('summary')}")
    if snap.current_era:
        print(f"\n--- current era ---")
        print(f"  {snap.current_era.get('label')}")
        print(f"  {snap.current_era.get('summary')}")
    print(f"\n--- identity arcs ---")
    for arc in snap.identity_arcs:
        print(f"  {arc.get('label')}: {arc.get('summary')}")
    print(f"\n--- trajectory ---")
    for note in snap.trajectory_notes:
        print(f"  - {note}")
    store.close()
    return 0


def cmd_retrieve(args) -> int:
    paths = storage_paths(args.storage, args.user)
    store = RecallStore(paths["recall"])
    frags = retrieve(store, args.query, k=args.k)
    print(f"Top {len(frags)} fragments for: {args.query!r}\n")
    for f in frags:
        print(f"  [{f.score:.3f}] {f.time_start.date()} | {f.source}")
        print(f"          {f.summary}")
        print(f"          (vec={f.vector_score:.2f} bm25={f.bm25_score:.2f} ent={f.entity_score:.2f} rcy={f.recency_boost:.2f})")
        print()
    store.close()
    return 0


def cmd_stats(args) -> int:
    paths = storage_paths(args.storage, args.user)
    store = RecallStore(paths["recall"])
    print(json.dumps(store.stats(), indent=2))
    store.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m pmc.memory.recall")
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--storage", default=str(_storage_root()))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("migrate")

    pv = sub.add_parser("preview", help="Cheap heuristic summaries + embeddings, no LLM. Validates the pipeline.")
    pv.add_argument("--limit", type=int, default=5000)

    cs = sub.add_parser("consolidate")
    cs.add_argument("--model", default="claude-sonnet-4-6")
    cs.add_argument("--limit", type=int, default=200)
    cs.add_argument("--sample", type=int, default=5)
    cs.add_argument("--concurrency", type=int, default=12)

    wk = sub.add_parser("working")
    wk.add_argument("--model", default="claude-sonnet-4-6")

    nr = sub.add_parser("narrative")
    nr.add_argument("--model", default="claude-sonnet-4-6")

    rt = sub.add_parser("retrieve")
    rt.add_argument("--query", required=True)
    rt.add_argument("--k", type=int, default=10)

    sub.add_parser("stats")

    args = p.parse_args()
    handlers = {
        "migrate": cmd_migrate,
        "preview": cmd_preview,
        "consolidate": cmd_consolidate,
        "working": cmd_working,
        "narrative": cmd_narrative,
        "retrieve": cmd_retrieve,
        "stats": cmd_stats,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
