"""Build HuggingFace Datasets from PMC Completions.

`datasets` is a heavy dependency — it's imported lazily inside each builder so
the rest of the train package (formatter, config, bundle) stays importable
without it. Install with `pip install pmc[train]`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pmc.schema.conversation import Completion
from pmc.train.formatter import (
    completion_to_dpo_pair,
    completion_to_messages,
)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


def _require_datasets() -> Any:
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "The `datasets` library is required. Install with `pip install pmc[train]`."
        ) from e
    return datasets


def build_sft_dataset(completions: Iterable[Completion]) -> Dataset:
    """Build an SFT dataset with a 'messages' column ready for TRL SFTTrainer."""
    datasets = _require_datasets()
    rows: list[dict[str, list[dict[str, str]]]] = []
    for c in completions:
        messages = completion_to_messages(c)
        if messages is not None:
            rows.append({"messages": messages})
    if not rows:
        raise ValueError("No usable SFT examples after formatting.")
    return datasets.Dataset.from_list(rows)


def build_dpo_dataset(completions: Iterable[Completion]) -> Dataset:
    """Build a DPO dataset with 'prompt' / 'chosen' / 'rejected' columns."""
    datasets = _require_datasets()
    rows: list[dict[str, list[dict[str, str]] | str]] = []
    for c in completions:
        pair = completion_to_dpo_pair(c)
        if pair is not None:
            rows.append(pair)
    if not rows:
        raise ValueError("No usable DPO pairs — completions need 2+ candidates.")
    return datasets.Dataset.from_list(rows)


def build_reward_dataset(completions: Iterable[Completion]) -> Dataset:
    """Reward model dataset — same shape as DPO (TRL RewardTrainer accepts it)."""
    return build_dpo_dataset(completions)


def split_train_eval(
    dataset: Dataset,
    eval_fraction: float = 0.1,
    seed: int = 42,
    min_eval_size: int = 1,
) -> DatasetDict:
    """Train/eval split. Returns the original as 'train' with empty eval if
    the dataset is too small to spare any examples."""
    _require_datasets()
    if eval_fraction <= 0 or len(dataset) < max(2, min_eval_size + 1):
        return _empty_eval_split(dataset)
    return dataset.train_test_split(test_size=eval_fraction, seed=seed)


def _empty_eval_split(dataset: Dataset) -> DatasetDict:
    datasets = _require_datasets()
    return datasets.DatasetDict(
        {
            "train": dataset,
            "test": dataset.select(range(0)),
        }
    )
