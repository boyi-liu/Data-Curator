"""Shared helpers for the per-dataset modules (``dataset/load_<name>.py``).

Each dataset is a single module that both *prepares* and *loads* its data:

  * it reformats a raw source into ``{"instruction", "input", "output"}`` records,
  * caches them to ``{data_dir}/<name>.jsonl`` (built once, on first ``load``),
  * exposes ``load(cfg, tokenizer)`` for the registry and a ``__main__`` CLI to
    (re)build the JSONL explicitly.

These helpers hold the parts every dataset shares.
"""

import json
import os
import random

from datasets import load_dataset

from dataset.formatting import make_tokenize_fn


def subsample(records, percentage, seed):
    """Randomly keep ``percentage`` of ``records`` (reproducible via ``seed``)."""
    if percentage < 1.0:
        rng = random.Random(seed)
        k = int(round(len(records) * percentage))
        if k < len(records):
            return rng.sample(records, k)
    return list(records)


def write_jsonl(records, path):
    """Write ``records`` as JSON lines to ``path``, creating parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def tokenize_split(cfg, tokenizer, path):
    """Load a unified JSONL, tokenize, and split off a validation set.

    Returns ``{"train": Dataset, "validation": Dataset | None}``.
    """
    raw = load_dataset("json", data_files=path, split="train")
    tokenized = raw.map(
        make_tokenize_fn(cfg, tokenizer),
        remove_columns=raw.column_names,
        desc=f"Tokenizing {cfg.dataset.name}",
    )
    val_split = cfg.dataset.validation_split or 0.0
    if val_split and val_split > 0:
        split = tokenized.train_test_split(test_size=val_split, seed=cfg.seed)
        return {"train": split["train"], "validation": split["test"]}
    return {"train": tokenized, "validation": None}
