"""Agent-driven synthesis layer.

The Rust extractors produce typed entities (people, places, themes,
events, open_loops, ...). This module is the second pass: the user's
chosen frontier model reads the structured graph and produces the
*threads* layer — named, categorized, evidence-cited items the user
can actually act on at first boot.

The structuring moat lives here. Extractors are commodity; pre-
existing data is what every user has. What makes PMC different is
that we take the pre-existing data and turn it into a navigable
picture of who this person is and what's in motion in their life.

Output: `<storage_root>/users/<uid>/graph/synth/threads.jsonl` — each
line a Thread the agent thinks is alive enough to surface. The
`/right-now` endpoint reads from here.
"""

from pmc.synthesis.threads import (
    Thread,
    ThreadEvidence,
    build_threads,
    load_threads,
)

__all__ = ["Thread", "ThreadEvidence", "build_threads", "load_threads"]
