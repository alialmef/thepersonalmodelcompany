"""The Consolidator — daily memory-consolidation pass over the graph.

Distinct from extraction (pmc-ingest, Rust) and chat (pmc chat). The
consolidator's job is to read the graph + new entities and produce
the *theory of you*: characterized people, projects, places, plus a
slowly-evolving self portrait.

Architecture:
  scorer.py        - salience decay + reinforcement (pure compute)
  characterize.py  - LLM pass: one entity at a time, "who is this to you"
  self_portrait.py - the slow `self.md` composer
  run.py           - the orchestrator: one full pass
  versioned.py     - snapshot every pass so the user can look back

Cadence: daily, triggered by `pmc watch` (task #57). Manual invocation
via `pmc consolidate` for forced runs.
"""

from pmc.consolidator.run import run_consolidation

__all__ = ["run_consolidation"]
