"""Identity layer — the system prompt that frames the model as the user's AI.

The model is a distinct entity, made by the user, trained on the user's writing.
It is NOT the user. It can speak in the user's voice as a capability when asked,
and it knows facts about the user from what it learned, but it always refers to
the user in the second person ("you", never "I").

The identity prompt does three things at inference time:

1. Pins **who the model is**: "you are <user>'s personal AI model"
2. Pins **who it's talking to**: "the user is <user> — refer to them as 'you'"
3. Sets the **recall discipline**: "use the retrieved snippets when they apply,
   say you don't remember when nothing relevant has been retrieved — do not
   fabricate facts about <user>"

Without this, even with a perfectly trained LoRA, the model has no grounded
sense of the relationship and will sometimes lapse into "I am you" first-person
confusion. The prompt is cheap insurance.

The composer optionally accepts concrete style facts (from `pmc.curate.style_profile`)
so the prompt can name observed style traits — that's what powers the
"model speaks first" hero moment in the chat UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityProfile:
    """Minimum identity facts the system prompt needs.

    Stored alongside each user's adapter bundle so the same identity is
    served at inference time as was used at training time. Travels with the
    new adapter on every retrain.
    """

    user_id: str
    display_name: str             # the name the model addresses the user with
    style_summary: str | None = None  # short prose summary from style_profile
    style_facts: tuple[str, ...] = ()  # 2–5 concrete style observations
    tone: str | None = None       # e.g. "lowercase, sparse, dry"


# Base identity skeleton. Every line of this is a tax on the model's
# context window, so keep it tight. Composes with retrieved memory snippets
# inserted by the serve layer just below this prompt.
_IDENTITY_TEMPLATE = """You are {name}'s personal AI model. \
{name} made you from their own writing — texts, notes, mail, documents — \
so you've learned how they write and you know things about them.

You are not {name}. You are their model. Always refer to them as "you" \
(never "I" or "me"). When {name} asks you to draft something, you can \
write in their voice; that is a capability you have, not your identity.

When relevant snippets from {name}'s past writing appear below as context, \
use them. When the context does not contain the answer to a factual question \
about {name}, say plainly that you do not remember — do not fabricate.

{style_block}"""


def build_identity_prompt(profile: IdentityProfile) -> str:
    """Compose the per-user system prompt."""
    style_block = _format_style_block(profile)
    return _IDENTITY_TEMPLATE.format(
        name=profile.display_name,
        style_block=style_block,
    ).strip()


def _format_style_block(profile: IdentityProfile) -> str:
    parts: list[str] = []
    if profile.tone:
        parts.append(f"{profile.display_name}'s voice tends to be: {profile.tone}.")
    if profile.style_facts:
        bullets = "\n".join(f"- {f}" for f in profile.style_facts)
        parts.append(
            f"Style traits you observed in {profile.display_name}'s writing:\n{bullets}"
        )
    if profile.style_summary:
        parts.append(profile.style_summary)
    return "\n\n".join(parts).strip()


def build_first_contact_message(profile: IdentityProfile) -> str:
    """Generate the seed prompt for the very first message the model "sends".

    Used by the chat UI's "model speaks first" moment. The model itself
    generates the line at inference time using this as its instruction.
    It speaks as ITSELF (the model), addressing the user in the second person,
    and names 2–3 observed style facts so the user immediately sees themselves
    reflected.

    This returns the *prompt that produces* the opening message — not the
    message itself.
    """
    facts = ", ".join(profile.style_facts[:3]) if profile.style_facts else "their voice"
    return (
        f"Write a single short opening message to {profile.display_name}, "
        f"introducing yourself as their model. Mention 2 or 3 things you've "
        f"observed about how they write ({facts}), and end with a question "
        f"that invites them to start a conversation. Address them as 'you', "
        f"never as 'I'. Use lowercase if that matches their style. "
        f"One short paragraph, no preamble."
    )
