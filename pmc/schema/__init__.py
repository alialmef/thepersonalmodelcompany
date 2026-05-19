"""Core data models for the personal model pipeline."""

from pmc.schema.annotations import (
    Annotation,
    PIIAnnotation,
    PreferenceAnnotation,
    QualityAnnotation,
    SourceAnnotation,
    StyleAnnotation,
)
from pmc.schema.base_models import (
    ALIASES,
    MODEL_REGISTRY,
    AgenticGrade,
    BaseModelSpec,
    ModelTier,
    default_spec,
    get_spec,
    list_specs,
    register,
    spec_for_tier,
    try_get_spec,
)
from pmc.schema.conversation import (
    Completion,
    CompletionCandidate,
    Conversation,
    Message,
    Role,
    SourceType,
)
from pmc.schema.training import (
    AdapterConfig,
    PreferencePair,
    SFTExample,
    TrainingConfig,
)
from pmc.schema.user import DataManifest, StyleProfile, User

__all__ = [
    "ALIASES",
    "AdapterConfig",
    "AgenticGrade",
    "Annotation",
    "BaseModelSpec",
    "Completion",
    "CompletionCandidate",
    "Conversation",
    "DataManifest",
    "MODEL_REGISTRY",
    "Message",
    "ModelTier",
    "PIIAnnotation",
    "PreferenceAnnotation",
    "PreferencePair",
    "QualityAnnotation",
    "Role",
    "SFTExample",
    "SourceAnnotation",
    "SourceType",
    "StyleAnnotation",
    "StyleProfile",
    "TrainingConfig",
    "User",
    "default_spec",
    "get_spec",
    "list_specs",
    "register",
    "spec_for_tier",
    "try_get_spec",
]
