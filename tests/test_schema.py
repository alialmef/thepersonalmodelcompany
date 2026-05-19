"""Tests for the core PMC schema."""

from datetime import datetime

from pmc.schema import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    PreferenceAnnotation,
    PreferencePair,
    QualityAnnotation,
    Role,
    SFTExample,
    SourceAnnotation,
    SourceType,
    StyleProfile,
    TrainingConfig,
    User,
)


def test_message_creation():
    msg = Message(role=Role.USER, content="Hello, how are you?")
    assert msg.role == Role.USER
    assert msg.content == "Hello, how are you?"
    assert msg.annotations == []


def test_conversation_from_messages():
    messages = [
        Message(role=Role.USER, content="Hey"),
        Message(role=Role.ASSISTANT, content="Hi there!"),
    ]
    conv = Conversation(messages=messages, source_type=SourceType.EMAIL)
    assert len(conv.messages) == 2
    assert conv.source_type == SourceType.EMAIL
    assert conv.id is not None


def test_completion_sft():
    context = Conversation(
        messages=[Message(role=Role.USER, content="What do you think about this?")]
    )
    candidate = CompletionCandidate(
        messages=[Message(role=Role.ASSISTANT, content="I think it's great.")]
    )
    completion = Completion(
        conversation=context,
        candidates=[candidate],
        user_id="user-123",
    )
    assert len(completion.candidates) == 1
    assert completion.user_id == "user-123"


def test_completion_preference_pair():
    context = Conversation(
        messages=[Message(role=Role.USER, content="Summarize this meeting")]
    )
    chosen = CompletionCandidate(
        messages=[Message(role=Role.ASSISTANT, content="Short and direct summary.")],
        annotations=[PreferenceAnnotation(chosen=True, dimension="style")],
    )
    rejected = CompletionCandidate(
        messages=[Message(role=Role.ASSISTANT, content="A very long and verbose summary...")],
        annotations=[PreferenceAnnotation(chosen=False, dimension="style")],
    )
    completion = Completion(conversation=context, candidates=[chosen, rejected])
    assert len(completion.candidates) == 2

    chosen_ann = completion.candidates[0].annotations[0]
    assert isinstance(chosen_ann, PreferenceAnnotation)
    assert chosen_ann.chosen is True


def test_sft_example():
    example = SFTExample(
        messages=[
            Message(role=Role.USER, content="Draft an email to the team"),
            Message(role=Role.ASSISTANT, content="Hey team, quick update..."),
        ],
        source=SourceAnnotation(
            source_type="email",
            source_id="gmail-abc123",
            timestamp=datetime(2025, 1, 15),
        ),
        train_on_last_n=1,
    )
    assert len(example.messages) == 2
    assert example.source is not None
    assert example.source.source_type == "email"


def test_preference_pair():
    pair = PreferencePair(
        conversation=[Message(role=Role.USER, content="How should I respond?")],
        chosen="Casual and warm response",
        rejected="Overly formal corporate response",
        dimension="style",
    )
    assert pair.chosen != pair.rejected
    assert pair.dimension == "style"


def test_training_config_defaults():
    config = TrainingConfig(user_id="user-123")
    # Default switched to Llama 3.1 8B — natively supported by Together AI
    # serverless multi-LoRA, our V0 serving target.
    assert config.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.adapter.rank == 32
    assert config.adapter.use_qlora is True
    assert config.learning_rate == 2e-4


def test_user_with_style_profile():
    user = User(
        email="test@example.com",
        name="Test User",
        style_profile=StyleProfile(
            formality=0.3,
            verbosity=0.6,
            tone_tags=["warm", "direct", "casual"],
            common_phrases=["honestly", "to be fair", "the thing is"],
        ),
    )
    assert user.style_profile is not None
    assert "warm" in user.style_profile.tone_tags
    assert user.total_training_examples == 0


def test_quality_annotation():
    msg = Message(
        role=Role.ASSISTANT,
        content="A well-written response",
        annotations=[
            QualityAnnotation(
                style_signal=0.9,
                coherence=0.85,
                sufficient_context=0.8,
                duplicate_risk=0.1,
                boilerplate_score=0.05,
                overall=0.85,
            )
        ],
    )
    quality = msg.annotations[0]
    assert isinstance(quality, QualityAnnotation)
    assert quality.overall == 0.85


def test_completion_serialization_roundtrip():
    completion = Completion(
        conversation=Conversation(
            messages=[Message(role=Role.USER, content="Test")]
        ),
        candidates=[
            CompletionCandidate(
                messages=[Message(role=Role.ASSISTANT, content="Response")]
            )
        ],
        user_id="user-456",
    )
    json_str = completion.model_dump_json()
    restored = Completion.model_validate_json(json_str)
    assert restored.user_id == completion.user_id
    assert restored.candidates[0].messages[0].content == "Response"
