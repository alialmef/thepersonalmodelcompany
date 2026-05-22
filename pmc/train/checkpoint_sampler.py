"""Checkpoint sampling for the /training "watch your voice arrive" UI.

Together's fine-tuning API doesn't reliably expose intermediate
checkpoint inference mid-run, so we ship a pragmatic version that
gives the user a real "before → after" moment without faking
intermediate states:

  * **Baseline sample** — at the start of the run, ask the base model
    (no adapter) to answer the fixed prompt. This is what generic
    Kimi/Llama produces — generic, polite, not the user.
  * **Final sample** — when the job completes, ask the fine-tuned
    model the same prompt. This is the user's voice.

Both samples are emitted as `checkpoint_sample` audit events. The
/training screen renders them stacked as they arrive: the user
literally watches the generic model become *them* in two beats.

If Together adds checkpoint-inference support in the future, this
module is where the intermediate sample logic goes — just append
calls at 33% / 66% trained tokens.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional


log = logging.getLogger(__name__)


# Fixed prompt used at every checkpoint. Picked because it surfaces
# voice (register, sentence length, what-you-talk-about) but doesn't
# require any specific personal knowledge — so even the base model
# can produce a meaningful baseline.
DEFAULT_PROMPT = "Tell me about your weekend."

DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8


@dataclass
class CheckpointSample:
    stage: str           # 'baseline' | 'final'
    response: str
    base_model: str
    adapter_model: Optional[str]  # None for baseline


def sample_base(
    base_model: str,
    *,
    api_key: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
) -> CheckpointSample:
    """Sample the base model with no adapter — the 'before' shot."""
    response = _together_chat(
        api_key=api_key,
        model=base_model,
        prompt=prompt,
    )
    return CheckpointSample(
        stage="baseline",
        response=response,
        base_model=base_model,
        adapter_model=None,
    )


def sample_final(
    base_model: str,
    output_model: str,
    *,
    api_key: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
) -> CheckpointSample:
    """Sample the fine-tuned model — the 'after' shot."""
    response = _together_chat(
        api_key=api_key,
        model=output_model,
        prompt=prompt,
    )
    return CheckpointSample(
        stage="final",
        response=response,
        base_model=base_model,
        adapter_model=output_model,
    )


def _together_chat(
    *,
    api_key: Optional[str],
    model: str,
    prompt: str,
) -> str:
    """One-shot chat call against Together. Returns the assistant's text."""
    api_key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY required for checkpoint sampling")
    try:
        from together import Together
    except ImportError as e:
        raise RuntimeError("together package required") from e

    client = Together(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )
    # Together follows OpenAI's response shape.
    try:
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Unexpected Together response shape: %s", e)
        return ""
