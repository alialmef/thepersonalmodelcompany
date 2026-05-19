"""Convert PMC Completions into the message format SFT trainers expect.

PMC semantics: the user's writing lives in `Role.ASSISTANT`. The context (what
others said to the user) lives in `Role.USER`. The chat-format mapping is the
identity — we just rename our enum values to the strings TRL/HF expect.

We deliberately do NOT inject a system prompt at training time. The whole point
of the adapter is that it IS the user; teaching it via a system-prompt scaffold
would just make it dependent on that scaffold at inference time. System prompts
are an inference-time concern.
"""

from __future__ import annotations

from collections.abc import Iterable

from pmc.schema.annotations import PreferenceAnnotation
from pmc.schema.conversation import Completion, CompletionCandidate, Role


def completion_to_messages(completion: Completion) -> list[dict[str, str]] | None:
    """Convert a Completion into a chat-format messages list for SFT.

    Returns None if the completion is unusable (no candidates, empty candidate).
    Uses the first candidate as the training target.
    """
    if not completion.candidates:
        return None
    candidate = completion.candidates[0]
    if not candidate.messages or not any(m.content.strip() for m in candidate.messages):
        return None

    messages: list[dict[str, str]] = []
    for msg in completion.conversation.messages:
        if msg.content.strip():
            messages.append({"role": msg.role.value, "content": msg.content})
    for msg in candidate.messages:
        if msg.content.strip():
            messages.append({"role": msg.role.value, "content": msg.content})

    if not messages:
        return None
    if not any(m["role"] == Role.ASSISTANT.value for m in messages):
        return None
    return messages


def completions_to_messages(
    completions: Iterable[Completion],
) -> list[list[dict[str, str]]]:
    out: list[list[dict[str, str]]] = []
    for c in completions:
        messages = completion_to_messages(c)
        if messages is not None:
            out.append(messages)
    return out


def completion_to_dpo_pair(
    completion: Completion,
) -> dict[str, list[dict[str, str]] | str] | None:
    """For future DPO use. Extract chosen/rejected from a Completion.

    Prefers explicit PreferenceAnnotations on candidates; falls back to first
    two candidates if no annotations are present.
    """
    if len(completion.candidates) < 2:
        return None

    chosen, rejected = _pick_chosen_rejected(completion.candidates)
    if chosen is None or rejected is None:
        return None

    prompt = [
        {"role": m.role.value, "content": m.content}
        for m in completion.conversation.messages
        if m.content.strip()
    ]
    chosen_msgs = [
        {"role": m.role.value, "content": m.content}
        for m in chosen.messages
        if m.content.strip()
    ]
    rejected_msgs = [
        {"role": m.role.value, "content": m.content}
        for m in rejected.messages
        if m.content.strip()
    ]
    if not chosen_msgs or not rejected_msgs:
        return None
    return {"prompt": prompt, "chosen": chosen_msgs, "rejected": rejected_msgs}


def _pick_chosen_rejected(
    candidates: list[CompletionCandidate],
) -> tuple[CompletionCandidate | None, CompletionCandidate | None]:
    chosen: CompletionCandidate | None = None
    rejected: CompletionCandidate | None = None
    for cand in candidates:
        for ann in cand.annotations:
            if isinstance(ann, PreferenceAnnotation):
                if ann.chosen and chosen is None:
                    chosen = cand
                elif not ann.chosen and rejected is None:
                    rejected = cand
    if chosen is None:
        chosen = candidates[0]
    if rejected is None or rejected is chosen:
        rejected = next((c for c in candidates if c is not chosen), None)
    return chosen, rejected


def build_inference_system_prompt(
    user_name: str | None,
    style_description: str | None,
) -> str | None:
    """Optional system prompt for *inference* time (not training).

    The adapter learns the style; at inference, this primes the model further.
    """
    parts: list[str] = []
    if user_name:
        parts.append(f"You are responding as {user_name}.")
    if style_description:
        parts.append(f"Style: {style_description}")
    if not parts:
        return None
    return " ".join(parts)
