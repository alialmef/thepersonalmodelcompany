"""Tests for the curate pipeline and its components."""

from __future__ import annotations

from datetime import datetime

import pytest

from pmc.curate import (
    CurateConfig,
    CuratePipeline,
    Deduplicator,
    HeuristicQualityScorer,
    HeuristicSyntheticPrompter,
    LLMQualityScorer,
    LLMSyntheticPrompter,
    MockLLMClient,
    attach_synthetic_prompt,
    clean,
    content_hash,
    detect_pii,
    extract_style_profile,
    jaccard,
    redact_text,
    shingles,
    split_conversation,
)
from pmc.schema.annotations import PIIType, QualityAnnotation
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
    SourceType,
)


# ---------- clean ----------


def test_clean_strips_device_footer():
    text = "Sounds great, see you then.\n\nSent from my iPhone"
    assert clean(text) == "Sounds great, see you then."


def test_clean_strips_quoted_reply():
    text = (
        "Yes, that works.\n\n"
        "On Mon, Jan 15, 2025 at 3:00 PM, Alice <alice@x.com> wrote:\n"
        "> Can you make it Thursday?\n"
        "> Let me know."
    )
    assert clean(text) == "Yes, that works."


def test_clean_strips_signature():
    text = "Sure, sounds good.\n\n--\nJohn Smith\nVP of Things"
    assert clean(text) == "Sure, sounds good."


def test_clean_strips_forwarded():
    text = (
        "Have a look at this.\n\n"
        "---------- Forwarded message ----------\n"
        "From: bob@x.com\nLong forwarded content here..."
    )
    assert clean(text) == "Have a look at this."


def test_clean_preserves_normal_text():
    text = "Just a regular email body. Nothing to strip."
    assert clean(text) == text


def test_clean_handles_empty():
    assert clean("") == ""
    assert clean("   \n  ") == ""


# ---------- splitter ----------


def _make_conv(*pairs: tuple[Role, str], source_type: SourceType | None = None) -> Conversation:
    return Conversation(
        messages=[Message(role=r, content=c) for r, c in pairs],
        source_type=source_type,
    )


def test_split_conversation_multi_turn():
    conv = _make_conv(
        (Role.USER, "How was the meeting?"),
        (Role.ASSISTANT, "Went well, we agreed on next steps and timeline."),
        (Role.USER, "Can you summarize?"),
        (Role.ASSISTANT, "Yes — Q1 ship date confirmed, Sarah owns backend rewrite."),
        source_type=SourceType.EMAIL,
    )
    completions = split_conversation(conv)
    assert len(completions) == 2

    first = completions[0]
    assert len(first.conversation.messages) == 1
    assert first.conversation.messages[0].role == Role.USER
    assert "Went well" in first.candidates[0].messages[0].content

    second = completions[1]
    assert len(second.conversation.messages) == 3
    assert "Q1 ship date" in second.candidates[0].messages[0].content


def test_split_conversation_drops_too_short():
    conv = _make_conv(
        (Role.USER, "?"),
        (Role.ASSISTANT, "ok"),
    )
    assert split_conversation(conv, min_response_chars=10) == []


def test_split_conversation_standalone_writing():
    conv = _make_conv((Role.ASSISTANT, "A standalone journal entry. " * 5))
    completions = split_conversation(conv, include_empty_context=True)
    assert len(completions) == 1
    assert completions[0].conversation.messages == []


def test_split_conversation_skip_empty_context():
    conv = _make_conv((Role.ASSISTANT, "Standalone writing piece long enough."))
    assert split_conversation(conv, include_empty_context=False) == []


def test_split_conversation_cleans_boilerplate():
    conv = _make_conv(
        (Role.USER, "What do you think?"),
        (Role.ASSISTANT, "I agree with the plan.\n\nSent from my iPhone"),
    )
    [completion] = split_conversation(conv, clean=True)
    assert completion.candidates[0].messages[0].content == "I agree with the plan."


# ---------- PII ----------


def test_detect_email_address():
    annotations = detect_pii("Contact me at alice@example.com please")
    assert len(annotations) == 1
    assert annotations[0].pii_type == PIIType.EMAIL_ADDRESS


def test_detect_phone_number():
    annotations = detect_pii("Call (555) 123-4567 or +1 555-987-6543")
    assert len(annotations) == 2
    assert all(a.pii_type == PIIType.PHONE_NUMBER for a in annotations)


def test_detect_ssn_and_credit_card():
    text = "SSN 123-45-6789 and card 4111 1111 1111 1111"
    annotations = detect_pii(text)
    types = {a.pii_type for a in annotations}
    assert PIIType.SSN in types
    assert PIIType.CREDIT_CARD in types
    assert all(a.sensitivity == 1.0 for a in annotations if a.pii_type in {PIIType.SSN, PIIType.CREDIT_CARD})


def test_detect_no_false_positives_on_dates():
    annotations = detect_pii("Meeting on 1/15/2025 at 3pm")
    types = {a.pii_type for a in annotations}
    assert PIIType.PHONE_NUMBER not in types
    assert PIIType.SSN not in types


def test_redact_text_replaces_severe_only():
    text = "SSN 123-45-6789 email alice@x.com"
    annotations = detect_pii(text)
    redacted, _ = redact_text(
        text, annotations, only_types={PIIType.SSN}
    )
    assert "123-45-6789" not in redacted
    assert "alice@x.com" in redacted
    assert "[REDACTED:ssn]" in redacted


def test_redact_text_preserves_indices_with_reverse_order():
    text = "First 123-45-6789 and second 987-65-4321"
    annotations = detect_pii(text)
    redacted, new_annotations = redact_text(
        text, annotations, only_types={PIIType.SSN}
    )
    assert "123-45-6789" not in redacted
    assert "987-65-4321" not in redacted
    redacted_count = sum(1 for a in new_annotations if a.redacted)
    assert redacted_count == 2


# ---------- dedup ----------


def test_content_hash_normalizes_whitespace():
    assert content_hash("hello world") == content_hash("  HELLO   WORLD  ")


def test_jaccard_basic():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0
    assert jaccard({"a", "b", "c"}, {"a", "b", "d"}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0


def test_deduplicator_exact_duplicate():
    d = Deduplicator()
    text = "The quick brown fox jumps over the lazy dog."
    is_dup1, _ = d.check(text)
    is_dup2, sim2 = d.check(text)
    assert is_dup1 is False
    assert is_dup2 is True
    assert sim2 == 1.0


def test_deduplicator_near_duplicate():
    d = Deduplicator(threshold=0.7)
    d.check("Hey Alice, can you confirm Thursday's meeting works for you?")
    is_dup, sim = d.check("Hi Alice, can you confirm Thursday's meeting works for you?")
    assert is_dup is True
    assert sim > 0.7


def test_deduplicator_distinct():
    d = Deduplicator()
    d.check("Sounds great, let's go ahead.")
    is_dup, _ = d.check("I disagree with this entire plan and want to start over.")
    assert is_dup is False


def test_shingles_short_text():
    assert shingles("hi", n=5) == {"hi"}
    assert "hello" in shingles("hello world", n=5)


# ---------- quality scoring ----------


def _completion_with_response(response: str, context: str = "What do you think?") -> Completion:
    return Completion(
        conversation=Conversation(
            messages=[Message(role=Role.USER, content=context)] if context else []
        ),
        candidates=[
            CompletionCandidate(messages=[Message(role=Role.ASSISTANT, content=response)])
        ],
    )


def test_heuristic_quality_scores_short_low():
    scorer = HeuristicQualityScorer()
    bad = scorer.score(_completion_with_response("ok"))
    good = scorer.score(_completion_with_response(
        "I think we should ship in Q1. The backend rewrite is on track and "
        "Sarah has good momentum. Let's not block on the dashboard refresh."
    ))
    assert good.overall > bad.overall
    assert good.overall > 0.4


def test_heuristic_quality_flags_boilerplate():
    scorer = HeuristicQualityScorer()
    boilerplate = scorer.score(_completion_with_response(
        "Out of office until Monday. I will respond when I return."
    ))
    assert boilerplate.boilerplate_score >= 0.5


def test_heuristic_quality_penalizes_all_caps():
    scorer = HeuristicQualityScorer()
    caps = scorer.score(_completion_with_response("WHY WOULD YOU EVER DO THAT??? STOP IT NOW"))
    normal = scorer.score(_completion_with_response("Why would you do that? Please stop."))
    assert caps.coherence < normal.coherence


def test_heuristic_quality_factors_duplicate_risk():
    scorer = HeuristicQualityScorer()
    base = scorer.score(_completion_with_response(
        "Going well, talked to Tom and he's on board with the new direction."
    ), similarity=0.0)
    high_dup = scorer.score(_completion_with_response(
        "Going well, talked to Tom and he's on board with the new direction."
    ), similarity=0.95)
    assert high_dup.overall < base.overall


def test_llm_quality_scorer_with_mock():
    client = MockLLMClient(default="0.8, 0.7")
    scorer = LLMQualityScorer(client)
    style, coherence = scorer.score(_completion_with_response("A real message"))
    assert style == 0.8
    assert coherence == 0.7
    assert len(client.calls) == 1


def test_llm_quality_scorer_handles_garbage_response():
    client = MockLLMClient(default="huh?")
    scorer = LLMQualityScorer(client)
    style, coherence = scorer.score(_completion_with_response("A real message"))
    assert style == 0.5 and coherence == 0.5


# ---------- synthesize ----------


def test_heuristic_prompter_for_standalone():
    prompter = HeuristicSyntheticPrompter()
    comp = _completion_with_response("My standalone thought", context="")
    comp = Completion(
        conversation=Conversation(messages=[], source_type=SourceType.NOTES),
        candidates=comp.candidates,
    )
    assert prompter.prompt_for(comp) == HeuristicSyntheticPrompter.TEMPLATES["notes"]


def test_heuristic_prompter_returns_none_with_context():
    prompter = HeuristicSyntheticPrompter()
    comp = _completion_with_response("Reply")
    assert prompter.prompt_for(comp) is None


def test_attach_synthetic_prompt_adds_context():
    comp = Completion(
        conversation=Conversation(messages=[], source_type=SourceType.NOTES),
        candidates=[CompletionCandidate(
            messages=[Message(role=Role.ASSISTANT, content="A note I wrote.")]
        )],
    )
    new_comp = attach_synthetic_prompt(comp, HeuristicSyntheticPrompter())
    assert len(new_comp.conversation.messages) == 1
    assert new_comp.conversation.messages[0].role == Role.USER


def test_llm_prompter_generates_prompt():
    client = MockLLMClient(default="What's on your mind?")
    prompter = LLMSyntheticPrompter(client)
    comp = Completion(
        conversation=Conversation(messages=[], source_type=SourceType.NOTES),
        candidates=[CompletionCandidate(
            messages=[Message(role=Role.ASSISTANT, content="Standalone writing piece")]
        )],
    )
    assert prompter.prompt_for(comp) == "What's on your mind?"


# ---------- style profile ----------


def test_extract_style_profile_basic():
    completions = [
        _completion_with_response(
            "Yeah, totally agree. Let's just ship it. We can iterate after."
        ),
        _completion_with_response(
            "Sounds good. I'm down for that approach. Let me know when."
        ),
        _completion_with_response(
            "Yep, makes sense. I'll loop in the team and get back to you."
        ),
    ]
    profile = extract_style_profile(completions)
    assert profile.formality < 0.6
    assert profile.sentence_length_avg is not None
    assert "casual" in profile.tone_tags or profile.formality < 0.5


def test_extract_style_profile_formal():
    completions = [
        _completion_with_response(
            "Furthermore, the proposal warrants careful consideration. "
            "Regarding the timeline, we should proceed with deliberate pace."
        ),
        _completion_with_response(
            "Sincerely appreciate your patience. Pursuant to our discussion, "
            "I am attaching the requested documentation."
        ),
    ]
    profile = extract_style_profile(completions)
    assert profile.formality > 0.55


def test_extract_style_profile_empty():
    profile = extract_style_profile([])
    assert profile.formality == 0.5
    assert profile.common_phrases == []


def test_extract_style_profile_with_llm_description():
    client = MockLLMClient(default="Writes with a warm, direct voice.")
    profile = extract_style_profile(
        [_completion_with_response("A long enough message for the LLM to assess.")],
        llm_client=client,
    )
    assert "warm" in profile.description.lower()


# ---------- end-to-end pipeline ----------


def test_pipeline_end_to_end():
    user = "user@example.com"
    convs = [
        # Threaded conversation with quality user responses
        Conversation(
            messages=[
                Message(role=Role.USER, content="What did you think of the demo?"),
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        "Honestly, pretty solid. The latency improvements are real and "
                        "the new UI feels much cleaner. I'd push on the onboarding flow "
                        "next — that's still where we lose people."
                    ),
                ),
            ],
            source_type=SourceType.EMAIL,
        ),
        # An auto-reply that should be filtered
        Conversation(
            messages=[
                Message(role=Role.USER, content="Hi, quick question"),
                Message(
                    role=Role.ASSISTANT,
                    content="Out of office until Monday. I will respond when I return.",
                ),
            ],
            source_type=SourceType.EMAIL,
        ),
        # A standalone note
        Conversation(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        "Random thought on focus: deep work blocks really do beat "
                        "context-switching. Going to try 90-minute blocks next week."
                    ),
                ),
            ],
            source_type=SourceType.NOTES,
        ),
    ]

    pipeline = CuratePipeline(CurateConfig(min_quality_score=0.25))
    result = pipeline.curate(convs)

    assert result.stats.input_conversations == 3
    assert result.stats.split_completions >= 3
    assert result.stats.dropped_low_quality >= 1  # the auto-reply
    assert result.stats.output_completions >= 2

    # The standalone note got a synthetic prompt
    standalone = [c for c in result.completions if c.conversation.source_type == SourceType.NOTES][0]
    assert len(standalone.conversation.messages) == 1
    assert standalone.conversation.messages[0].role == Role.USER

    # Every kept completion has a quality annotation
    for c in result.completions:
        assert any(isinstance(a, QualityAnnotation) for a in c.annotations)

    # The style profile was extracted
    assert result.style_profile.sentence_length_avg is not None


def test_pipeline_dedup_filters_duplicates():
    base_msg = (
        "Thanks for sending this over. I had a chance to review it last night "
        "and the structure looks solid. Let me circle back tomorrow with notes."
    )
    convs = [
        Conversation(
            messages=[
                Message(role=Role.USER, content=f"Question {i}"),
                Message(role=Role.ASSISTANT, content=base_msg),
            ],
            source_type=SourceType.EMAIL,
        )
        for i in range(3)
    ]
    pipeline = CuratePipeline(CurateConfig(min_quality_score=0.0))
    result = pipeline.curate(convs)
    assert result.stats.dropped_duplicate >= 2
    assert result.stats.output_completions == 1


def test_pipeline_redacts_severe_pii():
    convs = [
        Conversation(
            messages=[
                Message(role=Role.USER, content="What's your SSN?"),
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        "I would never share that. But hypothetically my SSN is "
                        "123-45-6789 — please don't use it for anything."
                    ),
                ),
            ],
            source_type=SourceType.EMAIL,
        ),
    ]
    result = CuratePipeline(CurateConfig(min_quality_score=0.0)).curate(convs)
    assert result.stats.redacted_severe >= 1
    for c in result.completions:
        text = " ".join(m.content for m in c.candidates[0].messages)
        assert "123-45-6789" not in text


def test_pipeline_stats_quality_buckets():
    convs = [
        Conversation(
            messages=[
                Message(role=Role.USER, content="What do you think?"),
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        "I think it's a strong direction. The reasoning around "
                        "user pull holds up and the timing aligns with how the "
                        "market is moving. I'd commit."
                    ),
                ),
            ],
            source_type=SourceType.EMAIL,
        ),
    ]
    result = CuratePipeline(CurateConfig(min_quality_score=0.0)).curate(convs)
    total_bucketed = sum(result.stats.quality_by_bucket.values())
    assert total_bucketed == result.stats.output_completions
