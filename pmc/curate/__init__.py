"""Agent-powered data curation: raw personal data → clean training examples."""

from pmc.curate.clean import clean
from pmc.curate.dedup import Deduplicator, content_hash, jaccard, shingles
from pmc.curate.llm import LLMClient, MockLLMClient
from pmc.curate.pii import SEVERE_PII, detect_pii, redact_text
from pmc.curate.pipeline import (
    CurateConfig,
    CuratePipeline,
    CurateResult,
    CurateStats,
)
from pmc.curate.quality import HeuristicQualityScorer, LLMQualityScorer
from pmc.curate.splitter import split_conversation, split_many
from pmc.curate.style_profile import extract_style_profile
from pmc.curate.synthesize import (
    HeuristicSyntheticPrompter,
    LLMSyntheticPrompter,
    SyntheticPrompter,
    attach_synthetic_prompt,
)

__all__ = [
    "CurateConfig",
    "CuratePipeline",
    "CurateResult",
    "CurateStats",
    "Deduplicator",
    "HeuristicQualityScorer",
    "HeuristicSyntheticPrompter",
    "LLMClient",
    "LLMQualityScorer",
    "LLMSyntheticPrompter",
    "MockLLMClient",
    "SEVERE_PII",
    "SyntheticPrompter",
    "attach_synthetic_prompt",
    "clean",
    "content_hash",
    "detect_pii",
    "extract_style_profile",
    "jaccard",
    "redact_text",
    "shingles",
    "split_conversation",
    "split_many",
]
