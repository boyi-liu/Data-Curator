"""Build the Alpaca instruction-tuning dataset.

Loads the Stanford Alpaca 52k set (Taori et al., 2023) and writes it in the
framework's ``{"instruction", "input", "output"}`` schema (see
``dataset/formatting.py``) as a single ``alpaca.jsonl``.

Alpaca already ships in instruction/input/output form, so reformatting is just
validation and stripping. By default the data is pulled from the Hub
(``tatsu-lab/alpaca``); pass ``--data_files`` to load a local copy instead.

Usage
-----
    # from the Hub
    python -m dataset.generate_alpaca --output ./data/alpaca.jsonl

    # from a local json/jsonl you already have
    python -m dataset.generate_alpaca --data_files ./raw/alpaca.json \
        --output ./data/alpaca.jsonl

    # a reproducible 10% subsample
    python -m dataset.generate_alpaca --sample_percentage 0.1 --seed 3
"""

import argparse
import json
import os
import random

DEFAULT_DATASET = "tatsu-lab/alpaca"


def load_raw(args):
    """Return a ``datasets.Dataset`` from the Hub or local ``--data_files``."""
    from datasets import load_dataset

    if args.data_files:
        ext = os.path.splitext(args.data_files)[1].lstrip(".").lower()
        fmt = {"jsonl": "json", "": "json"}.get(ext, ext)
        return load_dataset(fmt, data_files=args.data_files, split="train")
    return load_dataset(args.dataset_name, split="train")


def to_record(example):
    """Map one Alpaca row to ``{instruction, input, output}`` or ``None``."""
    instruction = (example.get("instruction") or "").strip()
    output = (example.get("output") or "").strip()
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": (example.get("input") or "").strip(), "output": output}


def subsample(records, percentage, rng):
    """Randomly keep ``percentage`` of ``records``."""
    if percentage < 1.0:
        k = int(round(len(records) * percentage))
        if k < len(records):
            return rng.sample(records, k)
    return list(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET, help="HF Hub dataset id.")
    parser.add_argument("--data_files", default=None, help="Local file to load instead of the Hub.")
    parser.add_argument("--output", default="./data/alpaca.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--sample_percentage",
        type=float,
        default=1.0,
        help="Fraction of the dataset to keep (default: all).",
    )
    parser.add_argument("--seed", type=int, default=3, help="Subsample seed.")
    args = parser.parse_args()

    raw = load_raw(args)
    records = [r for r in (to_record(ex) for ex in raw) if r is not None]
    kept = subsample(records, args.sample_percentage, random.Random(args.seed))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(kept)} examples (of {len(records)} valid) to {args.output}")


if __name__ == "__main__":
    main()
