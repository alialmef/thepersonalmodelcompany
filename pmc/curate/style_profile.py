"""Extract a StyleProfile from a corpus of curated Completions.

The profile captures aggregate writing characteristics — average sentence
length, formality, common phrases, vocabulary markers. It's both a useful
artifact for the user and an input to evaluation prompts ("does this sound
like the user, whose style is X?").
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from pmc.curate.llm import LLMClient
from pmc.schema.conversation import Completion
from pmc.schema.user import StyleProfile

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "must", "shall", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "from", "into", "out",
    "about", "so", "no", "not", "all", "any", "some", "just", "now", "also",
}

CONTRACTIONS = {
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i'll",
    "you'll", "we'll", "they'll", "i've", "you've", "we've", "they've",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "let's", "that's", "what's", "here's", "there's",
}

FORMAL_MARKERS = {
    "furthermore", "moreover", "nevertheless", "consequently", "therefore",
    "henceforth", "thereby", "whereby", "regards", "sincerely", "kindly",
    "pursuant", "aforementioned", "regarding", "respectfully",
}

CASUAL_MARKERS = {
    "yeah", "yep", "nope", "kinda", "gonna", "wanna", "gotta", "ya", "ok",
    "okay", "haha", "lol", "lmao", "btw", "tbh", "imo", "fyi", "ish", "yo",
}


def _candidate_texts(completions: Iterable[Completion]) -> list[str]:
    texts: list[str] = []
    for c in completions:
        for cand in c.candidates:
            for m in cand.messages:
                if m.content.strip():
                    texts.append(m.content)
    return texts


def extract_style_profile(
    completions: Iterable[Completion],
    *,
    top_phrases: int = 10,
    top_vocab: int = 15,
    llm_client: LLMClient | None = None,
) -> StyleProfile:
    texts = _candidate_texts(completions)
    if not texts:
        return StyleProfile()

    full_text = "\n\n".join(texts)
    words = re.findall(r"[A-Za-z']+", full_text.lower())
    sentences = [s for s in re.split(r"[.!?]+\s+", full_text) if s.strip()]

    avg_sent_len = (
        sum(len(s.split()) for s in sentences) / len(sentences)
        if sentences
        else None
    )

    formality = _formality_score(words)
    verbosity = _verbosity_score(sentences)
    tone_tags = _tone_tags(words, full_text.lower())

    bigrams = Counter(zip(words, words[1:]))
    common_bigrams = [
        f"{a} {b}"
        for (a, b), _count in bigrams.most_common(top_phrases * 4)
        if a not in STOPWORDS and b not in STOPWORDS
    ][:top_phrases]

    vocab_counts = Counter(w for w in words if w not in STOPWORDS and len(w) > 4)
    total = sum(vocab_counts.values()) or 1
    vocab_markers = [
        w
        for w, _c in vocab_counts.most_common(top_vocab * 2)
        if vocab_counts[w] / total < 0.05
    ][:top_vocab]

    description = (
        _generate_description(llm_client, full_text)
        if llm_client is not None
        else _heuristic_description(formality, verbosity, tone_tags, avg_sent_len)
    )

    return StyleProfile(
        formality=round(formality, 3),
        verbosity=round(verbosity, 3),
        tone_tags=tone_tags,
        vocabulary_markers=vocab_markers,
        sentence_length_avg=round(avg_sent_len, 2) if avg_sent_len else None,
        common_phrases=common_bigrams,
        description=description,
    )


def _formality_score(words: list[str]) -> float:
    if not words:
        return 0.5
    formal_hits = sum(1 for w in words if w in FORMAL_MARKERS)
    casual_hits = sum(1 for w in words if w in CASUAL_MARKERS)
    contraction_hits = sum(1 for w in words if w in CONTRACTIONS)
    n = len(words)
    formality = 0.5 + (formal_hits - casual_hits - contraction_hits * 0.5) / max(n, 1) * 50
    return max(0.0, min(1.0, formality))


def _verbosity_score(sentences: list[str]) -> float:
    if not sentences:
        return 0.5
    avg = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg <= 6:
        return 0.1
    if avg >= 30:
        return 0.95
    return min(1.0, (avg - 6) / 24)


def _tone_tags(words: list[str], text_lower: str) -> list[str]:
    tags: list[str] = []
    n = max(1, len(words))
    if sum(1 for w in words if w in CASUAL_MARKERS) / n > 0.005:
        tags.append("casual")
    if sum(1 for w in words if w in FORMAL_MARKERS) / n > 0.003:
        tags.append("formal")
    if text_lower.count("!") / max(1, len(text_lower)) > 0.005:
        tags.append("enthusiastic")
    if text_lower.count("?") / max(1, len(text_lower)) > 0.005:
        tags.append("inquisitive")
    if any(w in text_lower for w in ["haha", "lol", "lmao", "😂", "🤣"]):
        tags.append("humorous")
    if any(w in text_lower for w in ["thanks", "appreciate", "grateful"]):
        tags.append("warm")
    return tags


def _heuristic_description(
    formality: float,
    verbosity: float,
    tone_tags: list[str],
    avg_sent_len: float | None,
) -> str:
    formality_label = (
        "formal" if formality > 0.65 else "casual" if formality < 0.4 else "balanced"
    )
    verbosity_label = (
        "verbose" if verbosity > 0.6 else "terse" if verbosity < 0.35 else "moderate"
    )
    parts = [f"{formality_label} register", f"{verbosity_label} sentences"]
    if tone_tags:
        parts.append(f"tone: {', '.join(tone_tags)}")
    if avg_sent_len:
        parts.append(f"~{avg_sent_len:.0f} words/sentence avg")
    return "; ".join(parts)


def _generate_description(client: LLMClient, full_text: str) -> str:
    sample = full_text[:3000]
    system = (
        "Describe this person's writing style in 2-3 sentences. "
        "Focus on voice, tone, structure, and characteristic patterns. "
        "Write a description that another writer could use to imitate them."
    )
    try:
        return client.complete(
            system=system,
            prompt=sample,
            max_tokens=200,
            temperature=0.3,
        ).strip()
    except Exception:
        return ""
