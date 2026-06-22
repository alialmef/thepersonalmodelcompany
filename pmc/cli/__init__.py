"""Standalone terminal CLI for the personal agent.

`pmc chat` (and friends) live here. The old training-pipeline CLI in
pmc/orchestrator/cli.py keeps working; this module is what users get
when they install the binary to talk to their agent.

The design intent is "easy to talk to in the terminal" — a single
command after install that opens a REPL where the agent already knows
who you are because it has read your local graph.
"""
