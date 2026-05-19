"""Synthesize prompts for standalone writing (notes, documents).

Raw documents and notes have no context — just the user's writing. To turn them
into SFT examples we need a "prompt" the writing could be a response to.

Two generators:
- HeuristicSyntheticPrompter: simple templates, no LLM needed.
- LLMSyntheticPrompter: asks a judge model to generate a plausible prompt.
"""

from __future__ import annotations

from typing import Protocol

from pmc.curate.llm import LLMClient
from pmc.schema.conversation import (
    Completion,
    Conversation,
    Message,
    Role,
)


class SyntheticPrompter(Protocol):
    def prompt_for(self, completion: Completion) -> str | None: ...


class HeuristicSyntheticPrompter:
    """Templates that work without any LLM call."""

    TEMPLATES = {
        "default": "Continue writing in your own voice.",
        "notes": "Share your thoughts on the following topic.",
        "document": "Draft this in your own voice.",
        "email": "Write a response.",
        "imessage": "Reply to this.",
    }

    def prompt_for(self, completion: Completion) -> str | None:
        if completion.conversation.messages:
            return None
        source = completion.conversation.source_type
        key = source.value if source else "default"
        return self.TEMPLATES.get(key, self.TEMPLATES["default"])


class LLMSyntheticPrompter:
    """Generate a prompt the writing could plausibly be a response to."""

    SYSTEM = (
        "Given a piece of writing, generate a single short prompt or message "
        "that this writing could plausibly be a response to. The prompt should "
        "feel natural — like something a person would actually say to invite "
        "this response. Reply with ONLY the prompt, no preamble, no quotes."
    )

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def prompt_for(self, completion: Completion) -> str | None:
        if completion.conversation.messages:
            return None
        if not completion.candidates or not completion.candidates[0].messages:
            return None
        writing = completion.candidates[0].messages[0].content[:1500]
        try:
            return self.client.complete(
                system=self.SYSTEM,
                prompt=f"Writing:\n\n{writing}",
                max_tokens=120,
                temperature=0.7,
            ).strip()
        except Exception:
            return None


def attach_synthetic_prompt(
    completion: Completion,
    prompter: SyntheticPrompter,
) -> Completion:
    """If the completion has no context, generate a synthetic USER prompt for it.

    Returns a new Completion with the prompt prepended; the original if no
    prompter result was produced or the completion already had context.
    """
    if completion.conversation.messages:
        return completion
    prompt_text = prompter.prompt_for(completion)
    if not prompt_text:
        return completion
    new_context = Conversation(
        messages=[Message(role=Role.USER, content=prompt_text)],
        source_type=completion.conversation.source_type,
    )
    return Completion(
        id=completion.id,
        conversation=new_context,
        candidates=completion.candidates,
        annotations=completion.annotations,
        user_id=completion.user_id,
    )
