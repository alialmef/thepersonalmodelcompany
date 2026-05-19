"""Prompt templates for LLM judges.

The pairwise and Likert prompts are adapted from yolo's `prompt_builders.py`.
We keep them as pure functions returning strings so they're easy to swap and
test independently.
"""

from __future__ import annotations

from enum import StrEnum


class EvalDimension(StrEnum):
    """The dimensions PMC evaluates a personal model on."""

    STYLE_MATCH = "style_match"
    TONE_MATCH = "tone_match"
    VOCABULARY = "vocabulary"
    FORMALITY = "formality"
    FACTUAL_ACCURACY = "factual_accuracy"
    PRIVACY_SAFETY = "privacy_safety"
    OVERALL = "overall"


DIMENSION_PROMPTS: dict[EvalDimension, str] = {
    EvalDimension.STYLE_MATCH: "how similar to {name}'s writing style this is",
    EvalDimension.TONE_MATCH: "whether this matches {name}'s typical tone",
    EvalDimension.VOCABULARY: "whether this uses vocabulary {name} would use",
    EvalDimension.FORMALITY: "whether this is at the right formality level for {name}",
    EvalDimension.FACTUAL_ACCURACY: "whether the personal details about {name} are accurate",
    EvalDimension.PRIVACY_SAFETY: "whether this exposes private information",
    EvalDimension.OVERALL: "how much this sounds like {name} overall",
}


PAIRWISE_SYSTEM = (
    "You are evaluating two candidate responses to decide which one sounds more "
    "like a specific person ({name}) would write. Use their writing style, tone, "
    "vocabulary, and personality as your basis. Output ONLY a single number from "
    "-3 to +3 on the first line, then your reasoning on the next line."
)


def render_pairwise_prompt(
    conversation: list[dict[str, str]],
    response_a: str,
    response_b: str,
    user_name: str,
    user_style_profile: str | None,
    dimension: EvalDimension,
) -> str:
    """Build the user-message body for a pairwise LLM judgment."""
    style_block = ""
    if user_style_profile:
        style_block = f"\n\n{user_name}'s style profile:\n{user_style_profile.strip()}\n"

    conv_block = _render_conversation(conversation)
    aspect = DIMENSION_PROMPTS.get(dimension, DIMENSION_PROMPTS[EvalDimension.OVERALL])
    aspect = aspect.format(name=user_name)

    return f"""{style_block}
Conversation:
{conv_block}

Response 1:
{response_a}

Response 2:
{response_b}

Rate between -3 and +3 {aspect}:
* -3: Response 1 is clearly {user_name}'s style
* -2: Response 1 is more like {user_name}
* -1: Response 1 is slightly more like {user_name}
*  0: Both are equally (un)like {user_name}
* +1: Response 2 is slightly more like {user_name}
* +2: Response 2 is more like {user_name}
* +3: Response 2 is clearly {user_name}'s style

Output the number on the first line, then your reasoning."""


LIKERT_SYSTEM = (
    "You are scoring a single response on a 1-5 scale for how well it represents "
    "a specific person's style. Output ONLY a single integer 1-5 on the first line, "
    "then your reasoning on the next line."
)


def render_likert_prompt(
    conversation: list[dict[str, str]],
    response: str,
    user_name: str,
    user_style_profile: str | None,
    dimension: EvalDimension,
) -> str:
    """Build the user-message body for a single-response Likert judgment."""
    style_block = ""
    if user_style_profile:
        style_block = f"\n\n{user_name}'s style profile:\n{user_style_profile.strip()}\n"
    conv_block = _render_conversation(conversation)
    aspect = DIMENSION_PROMPTS.get(dimension, DIMENSION_PROMPTS[EvalDimension.OVERALL])
    aspect = aspect.format(name=user_name)

    return f"""{style_block}
Conversation:
{conv_block}

Response:
{response}

Score 1-5 for {aspect}:
* 1: Clearly not {user_name}
* 2: Mostly unlike {user_name}
* 3: Mixed / unclear
* 4: Mostly like {user_name}
* 5: Clearly {user_name}

Output the integer on the first line, then your reasoning."""


def _render_conversation(conversation: list[dict[str, str]]) -> str:
    if not conversation:
        return "(no prior context)"
    lines = []
    for msg in conversation:
        role = msg.get("role", "user").title()
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
