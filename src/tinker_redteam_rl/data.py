from __future__ import annotations

from typing import Iterable

import datasets
from datasets import Dataset

from .config import DataConfig


_DEFAULT_PROMPT_FIELDS = ["prompt", "question", "input", "text"]


def load_dataset(cfg: DataConfig) -> Dataset:
    dataset = datasets.load_dataset(
        cfg.dataset_name,
        split=cfg.dataset_split,
        data_dir=cfg.dataset_data_dir,
    )
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=cfg.seed)

    if cfg.only_unsafe and "is_safe" in dataset.features:
        dataset = dataset.filter(lambda row: not bool(row["is_safe"]))

    if cfg.data_size > 0:
        dataset = dataset.select(range(min(cfg.data_size, len(dataset))))
    return dataset


def get_prompt_from_row(row: dict, cfg: DataConfig) -> str:
    if cfg.prompt_field in row:
        return str(row[cfg.prompt_field])
    for field in _DEFAULT_PROMPT_FIELDS:
        if field in row:
            return str(row[field])
    raise KeyError(
        f"Could not find prompt field in row. Tried {cfg.prompt_field} and {_DEFAULT_PROMPT_FIELDS}."
    )


def iter_prompts(batch: Dataset, cfg: DataConfig) -> Iterable[str]:
    for row in batch:
        yield get_prompt_from_row(row, cfg)
