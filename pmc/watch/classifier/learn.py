"""Learning hooks — the gate's personal-to-this-user weights.

The rules layer applies the same defaults to everyone. This module
overlays per-user adjustments on top of those defaults: paths to
always promote, paths to always drop, salience boosts for entities
the user has marked important.

Storage: a JSONL file at `~/.pmc/gate-feedback.jsonl`. Each row is
one feedback event:

  { "ts": ..., "kind": "remember"|"forget"|"boost"|"demote",
    "target_type": "path"|"person"|"theme"|"file",
    "target": "<value>",
    "weight": <float -1..+1>,
    "note": "<optional user note>" }

The classifier reads this on startup (and on cache invalidation) and
folds it into rule evaluation: a `forget` for a path forces DROP; a
`remember` for a path forces PROMOTE; a `boost` on a person/theme
makes events touching that entity promote even if rules say defer.

User-facing surface: `pmc remember <thing>` and `pmc forget <thing>`
CLI commands (added separately in pmc/cli/learn.py).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pmc.cli.local_config import CONFIG_DIR
from pmc.watch.event import Decision


log = logging.getLogger("pmc.watch.learn")


FEEDBACK_FILE = CONFIG_DIR / "gate-feedback.jsonl"


@dataclass
class Feedback:
    ts: float
    kind: str            # "remember" | "forget" | "boost" | "demote"
    target_type: str     # "path" | "person" | "theme" | "file"
    target: str
    weight: float = 1.0  # +1 to forcibly promote, -1 to forcibly drop
    note: str = ""

    def to_json(self) -> dict:
        return {
            "ts": self.ts,
            "kind": self.kind,
            "target_type": self.target_type,
            "target": self.target,
            "weight": self.weight,
            "note": self.note,
        }


def record(fb: Feedback) -> None:
    """Append one feedback row to the JSONL log."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a") as f:
        f.write(json.dumps(fb.to_json()) + "\n")


def load_all() -> list[Feedback]:
    if not FEEDBACK_FILE.is_file():
        return []
    out: list[Feedback] = []
    for line in FEEDBACK_FILE.open():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out.append(Feedback(
                ts=float(row.get("ts", 0)),
                kind=row.get("kind", ""),
                target_type=row.get("target_type", ""),
                target=row.get("target", ""),
                weight=float(row.get("weight", 0)),
                note=row.get("note", ""),
            ))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------
# Overlay logic
# ---------------------------------------------------------------------------


class FeedbackOverlay:
    """In-memory index of user feedback. Pass to the classifier so it
    can override default rule decisions per-user."""

    def __init__(self) -> None:
        self._path_remember: list[str] = []
        self._path_forget: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        items = load_all()
        rem: list[str] = []
        forg: list[str] = []
        for fb in items:
            if fb.target_type != "path":
                continue
            if fb.kind == "remember" or fb.weight > 0:
                rem.append(fb.target)
            elif fb.kind == "forget" or fb.weight < 0:
                forg.append(fb.target)
        # Latest wins — dedupe but prefer the most-recent direction.
        self._path_remember = list(dict.fromkeys(rem))
        self._path_forget = list(dict.fromkeys(forg))

    def override_for(self, path: str) -> Optional[Decision]:
        """Return PROMOTE / DROP if the user has explicitly taught us
        about this path; None to leave the rule decision alone.

        Forget wins ties — if both lists touch the path, we drop.
        """
        # Resolve symlinks so /tmp/X and /private/tmp/X match the same
        # stored pattern. Cheap stat — only fires for matches, not the
        # 90% of events the junk filter already dropped.
        try:
            canonical = str(Path(path).resolve())
        except Exception:  # noqa: BLE001
            canonical = path
        candidates = {path, canonical}
        for pat in self._path_forget:
            for cand in candidates:
                if cand == pat or cand.startswith(pat.rstrip("/") + "/"):
                    return Decision.DROP
        for pat in self._path_remember:
            for cand in candidates:
                if cand == pat or cand.startswith(pat.rstrip("/") + "/"):
                    return Decision.PROMOTE
        return None


__all__ = ["Feedback", "FeedbackOverlay", "record", "load_all", "FEEDBACK_FILE"]
