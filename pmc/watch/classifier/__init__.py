"""Two-tier classifier: rules first, LLM for ambiguity."""

from pmc.watch.classifier.rules import classify

__all__ = ["classify"]
