"""Curate-stage supervisor.

Runs Claude over a sample of curated training examples and flags anything
that shouldn't go into the model. Three categories of issue:

  * **noise**       — pure-noise rows (one-word replies, "k", "lol",
                       quoted memes). These get *dropped* automatically.
  * **leaked**      — items that look like secrets (API keys, passwords,
                       full credit card numbers, full home addresses if
                       clearly private). *Flagged* for user review.
  * **third_party** — items where the user is mostly quoting or pasting
                       someone else's content, not their own writing.
                       *Flagged* for user review.

We do NOT auto-delete leaked or third_party items — only the user can
make that call. Noise is auto-dropped because dropping it is what
upstream curate's quality filter is *trying* to do anyway; we're just
catching what slipped through.

Cost: one Claude call per batch of ~50 examples. ~$0.005 per batch.
For a 5,000-example dataset that's 100 batches ≈ $0.50.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from pmc.schema.conversation import Completion
from pmc.train.formatter import completion_to_messages


log = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 50  # examples per Claude call
SAMPLE_SIZE = 500  # cap total reviewed examples to keep cost bounded


SUPERVISOR_SYSTEM_PROMPT = """You are reviewing a batch of training examples
for a personal AI model that will learn to write in the user's voice.

For each example, decide if it should be:
  * "keep"        — represents the user's writing fairly
  * "noise"       — too thin to teach voice from (one-word reply, "k",
                    "lol", emoji only, single emoji + word, link-only)
  * "leaked"      — contains a secret (API key, password, full credit
                    card, full home address, social security number,
                    private medical detail)
  * "third_party" — the user is mostly quoting or pasting someone
                    else's content, not their own writing

Return JSON only, no markdown fence:

{
  "verdicts": [
    {"index": 0, "decision": "keep|noise|leaked|third_party",
     "reason": "one short phrase if not 'keep', else null"},
    ...
  ]
}

Be permissive about "keep" — only flag clear cases. Marginal examples
are fine to keep; the curate stage already filtered most short text.
"""


@dataclass
class SupervisorVerdict:
    index: int
    decision: str           # 'keep' | 'noise' | 'leaked' | 'third_party'
    reason: Optional[str]   # one-line explanation if not 'keep'


@dataclass
class SupervisorReport:
    kept_indices: list[int] = field(default_factory=list)
    noise_indices: list[int] = field(default_factory=list)
    leaked: list[SupervisorVerdict] = field(default_factory=list)
    third_party: list[SupervisorVerdict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_flags(self) -> bool:
        return bool(self.leaked or self.third_party)

    def summary(self) -> dict:
        return {
            "kept":         len(self.kept_indices),
            "noise":        len(self.noise_indices),
            "leaked":       len(self.leaked),
            "third_party":  len(self.third_party),
            "flagged_total": len(self.leaked) + len(self.third_party),
        }


def supervise(
    completions: list[Completion],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    sample_size: int = SAMPLE_SIZE,
    batch_size: int = BATCH_SIZE,
    max_concurrency: int = 8,
) -> SupervisorReport:
    """Run Claude over a sample of completions; return verdicts.

    The returned report contains:
      * `kept_indices` — completions that survived review (note: this
        is a list of original indices; the caller filters its own list)
      * `noise_indices` — completions to drop
      * `leaked`, `third_party` — completions to surface for user review

    If the API key is missing or Claude is unreachable, returns an
    "all kept" report with an error noted. Supervision is a soft layer:
    failing it should never block training.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return SupervisorReport(
            kept_indices=list(range(len(completions))),
            errors=["ANTHROPIC_API_KEY not set; skipped supervision"],
        )

    try:
        import anthropic
    except ImportError as e:
        return SupervisorReport(
            kept_indices=list(range(len(completions))),
            errors=[f"anthropic import failed: {e}"],
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Subsample (stratified is overkill; first-N is fine since curate
    # already shuffles by quality score upstream).
    sample = list(enumerate(completions))[: min(sample_size, len(completions))]
    if not sample:
        return SupervisorReport()

    # Chunk into batches.
    batches: list[list[tuple[int, Completion]]] = []
    for i in range(0, len(sample), batch_size):
        batches.append(sample[i : i + batch_size])

    all_verdicts: list[SupervisorVerdict] = []
    errors: list[str] = []

    def review_batch(batch: list[tuple[int, Completion]]) -> list[SupervisorVerdict]:
        prompt = _format_batch_prompt(batch)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": SUPERVISOR_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[: -3]
            data = json.loads(text)
            verdicts = []
            # The batch positions map back to the original indices in
            # `completions` via the (original_index, completion) tuple
            # we passed in.
            for v in data.get("verdicts", []):
                local_idx = v.get("index")
                if not isinstance(local_idx, int) or local_idx >= len(batch):
                    continue
                orig_idx = batch[local_idx][0]
                verdicts.append(SupervisorVerdict(
                    index=orig_idx,
                    decision=v.get("decision", "keep"),
                    reason=v.get("reason"),
                ))
            return verdicts
        except Exception as e:
            log.warning("supervisor batch failed: %s", e)
            errors.append(str(e))
            # On failure, default to "keep" — never lose data to a
            # transient API hiccup.
            return [SupervisorVerdict(index=i, decision="keep", reason=None)
                    for i, _ in batch]

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futures = [ex.submit(review_batch, b) for b in batches]
        for fut in as_completed(futures):
            all_verdicts.extend(fut.result())

    report = SupervisorReport(errors=errors)
    seen: set[int] = set()
    for v in all_verdicts:
        if v.index in seen:
            continue
        seen.add(v.index)
        if v.decision == "noise":
            report.noise_indices.append(v.index)
        elif v.decision == "leaked":
            report.leaked.append(v)
        elif v.decision == "third_party":
            report.third_party.append(v)
        else:
            report.kept_indices.append(v.index)

    # Any completion we didn't review (beyond the sample) is implicitly kept.
    for i in range(len(completions)):
        if i not in seen:
            report.kept_indices.append(i)

    return report


def _format_batch_prompt(batch: list[tuple[int, Completion]]) -> str:
    """Render a batch of completions in a numbered list Claude can review."""
    lines = ["Review these training examples and return one verdict per example."]
    lines.append("")
    for local_idx, (_orig_idx, completion) in enumerate(batch):
        messages = completion_to_messages(completion) or []
        # Compact: show only assistant turn (the training target) plus
        # last user turn for context.
        last_user = ""
        last_assistant = ""
        for m in messages:
            if m.get("role") == "user":
                last_user = m.get("content", "")
            elif m.get("role") == "assistant":
                last_assistant = m.get("content", "")
        lines.append(f"[{local_idx}]")
        if last_user:
            lines.append(f"  context: {last_user[:300]}")
        lines.append(f"  user wrote: {last_assistant[:600]}")
        lines.append("")
    lines.append("Return the JSON verdicts.")
    return "\n".join(lines)


def apply_report(completions: list[Completion], report: SupervisorReport) -> list[Completion]:
    """Return a new list with `noise_indices` dropped. Leaked /
    third_party stay in the dataset for the user to decide via /eval."""
    drop = set(report.noise_indices)
    return [c for i, c in enumerate(completions) if i not in drop]
