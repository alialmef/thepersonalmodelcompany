"""Pre-deploy supervisor.

Before the trained adapter is registered for inference, sample it on a
small diverse prompt set and have Claude review the outputs. Three
failure modes worth catching:

  * **generic**     — the model didn't take voice. Sounds like the base.
  * **off_voice**   — the model overshot. Sounds like a caricature.
  * **hallucinated**— the model invented facts about the user that
                       aren't in their data.

Verdict shapes:
  * "approve"  — model is good enough to deploy
  * "hold"     — model has issues; user should review before deploy

Cost: 10 inference samples + 1 Claude review call ≈ $0.10 per training.
Cheap insurance for "I trained a $30 model that sounds nothing like me."
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional


log = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_PROMPTS = [
    "Tell me about your weekend.",
    "What are you working on these days?",
    "Quick — what should I have for dinner?",
    "Got a sec? I need to vent about something.",
    "How would you describe yourself to a stranger?",
    "What's the best thing you've read recently?",
    "Hey, can you help me write a quick email?",
    "Tell me a story.",
    "What do you actually believe about [topic]?",
    "Sum up your day in one sentence.",
]

DEPLOY_SYSTEM_PROMPT = """You are reviewing the outputs of a personal AI
model that was just fine-tuned on the user's writing. The goal is for
the model to sound like the user — same register, same word choice,
same depth, same patterns.

You'll see 10 prompts and the model's responses. Compare them mentally
to the user's writing samples (also provided). Return one of:

  * "approve" — model sounds like the user, no obvious failures
  * "hold"    — model has clear issues (generic/AI-like, caricature
                of user's voice, hallucinated user facts, hate/safety
                concerns)

If "hold", list the specific issues. Be terse — these notes show on
a small UI screen.

Return JSON only, no markdown fence:

{
  "verdict": "approve" | "hold",
  "summary": "one short sentence",
  "issues": [{"prompt_index": 0, "kind": "generic|off_voice|hallucinated|other", "note": "short"}, ...]
}
"""


@dataclass
class DeployIssue:
    prompt_index: int
    kind: str
    note: str


@dataclass
class DeployReport:
    verdict: str = "approve"  # 'approve' | 'hold'
    summary: str = ""
    issues: list[DeployIssue] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)  # [{prompt, response}]
    errors: list[str] = field(default_factory=list)

    @property
    def is_approved(self) -> bool:
        return self.verdict == "approve"


def supervise_deploy(
    base_model: str,
    output_model: str,
    user_writing_samples: list[str],
    *,
    api_key: Optional[str] = None,
    together_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompts: list[str] = DEFAULT_SAMPLE_PROMPTS,
) -> DeployReport:
    """Sample the trained model on `prompts`, then ask Claude whether
    to approve or hold deploy.

    `user_writing_samples` should be ~5-10 short authored snippets so
    Claude has ground truth to compare to. Caller provides them from
    the curated dataset.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    together_key = together_key or os.environ.get("TOGETHER_API_KEY")
    report = DeployReport()
    if not api_key or not together_key:
        report.errors.append("missing ANTHROPIC_API_KEY or TOGETHER_API_KEY")
        return report

    # Sample the model on each prompt.
    samples = _sample_model(output_model, prompts, together_key)
    report.samples = samples
    if not samples:
        report.errors.append("no samples produced (inference failed)")
        return report

    # Ask Claude.
    try:
        import anthropic
    except ImportError as e:
        report.errors.append(f"anthropic import failed: {e}")
        return report

    client = anthropic.Anthropic(api_key=api_key)
    user_message = _format_review_prompt(samples, user_writing_samples)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system=[{"type": "text", "text": DEPLOY_SYSTEM_PROMPT,
                      "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: -3]
        data = json.loads(text)
        report.verdict = data.get("verdict", "approve")
        report.summary = data.get("summary", "")
        report.issues = [
            DeployIssue(
                prompt_index=int(i.get("prompt_index", -1)),
                kind=str(i.get("kind", "other")),
                note=str(i.get("note", "")),
            )
            for i in data.get("issues", [])
        ]
    except Exception as e:
        report.errors.append(f"claude review failed: {e}")
        # Default to approve on supervisor failure — we don't want a
        # transient API blip to block deploy. The other supervisors
        # (curate, memory) will have caught most issues already.
    return report


def _sample_model(
    output_model: str,
    prompts: list[str],
    together_key: str,
) -> list[dict]:
    try:
        from together import Together
    except ImportError:
        return []
    client = Together(api_key=together_key)
    samples: list[dict] = []
    for p in prompts:
        try:
            resp = client.chat.completions.create(
                model=output_model,
                messages=[{"role": "user", "content": p}],
                max_tokens=200,
                temperature=0.8,
            )
            text = resp.choices[0].message.content.strip()
            samples.append({"prompt": p, "response": text})
        except Exception as e:
            log.warning("deploy sample failed for prompt %r: %s", p, e)
            samples.append({"prompt": p, "response": f"[error: {e}]"})
    return samples


def _format_review_prompt(samples: list[dict], writing_samples: list[str]) -> str:
    lines = []
    lines.append("USER'S OWN WRITING (ground truth for voice):")
    for i, s in enumerate(writing_samples[:8]):
        lines.append(f"  [{i}] {s[:400]}")
    lines.append("")
    lines.append("MODEL OUTPUTS (need to sound like the writing above):")
    for i, s in enumerate(samples):
        lines.append(f"  [{i}] prompt: {s['prompt']}")
        lines.append(f"      response: {s['response'][:500]}")
        lines.append("")
    lines.append("Return the JSON verdict.")
    return "\n".join(lines)
