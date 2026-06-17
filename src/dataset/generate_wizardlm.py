"""Build the WizardLM Evol-Instruct dataset.

Loads the WizardLM Evol-Instruct data (Xu et al., 2023) and writes it in the
framework's ``{"instruction", "input", "output"}`` schema (see
``dataset/formatting.py``) as a single ``wizardlm.jsonl``.

Two on-Hub layouts are handled automatically:

  * ``instruction`` / ``output`` columns (e.g. ``WizardLM_evol_instruct_70k``)
  * ShareGPT-style ``conversations`` -- a list of ``{"from": "human"/"gpt",
    "value": ...}`` turns (e.g. ``WizardLM_evol_instruct_V2_196k``). The first
    human turn becomes ``instruction`` and its ``gpt`` reply becomes ``output``;
    Evol-Instruct rows are single-turn.

By default the data is pulled from the Hub (``--dataset_name``); pass
``--data_files`` to load a local copy instead.

Usage
-----
    # 70k version from the Hub
    python -m dataset.generate_wizardlm --output ./data/wizardlm.jsonl

    # the larger V2 196k version
    python -m dataset.generate_wizardlm \
        --dataset_name WizardLMTeam/WizardLM_evol_instruct_V2_196k

    # from a local file you already have
    python -m dataset.generate_wizardlm --data_files ./raw/wizardlm.json
"""

import argparse
import json
import os
import random

DEFAULT_DATASET = "WizardLMTeam/WizardLM_evol_instruct_70k"


def load_raw(args):
    """Return a ``datasets.Dataset`` from the Hub or local ``--data_files``."""
    from datasets import load_dataset

    if args.data_files:
        ext = os.path.splitext(args.data_files)[1].lstrip(".").lower()
        fmt = {"jsonl": "json", "": "json"}.get(ext, ext)
        return load_dataset(fmt, data_files=args.data_files, split="train")
    return load_dataset(args.dataset_name, split="train")


def _from_conversations(conversations):
    """Pull the first human->gpt pair out of a ShareGPT ``conversations`` list."""
    instruction = output = None
    for turn in conversations:
        role, value = turn.get("from"), (turn.get("value") or "").strip()
        if not value:
            continue
        if instruction is None and role == "human":
            instruction = value
        elif instruction is not None and role == "gpt":
            output = value
            break
    return instruction, output


def to_record(example):
    """Map one WizardLM row to ``{instruction, input, output}`` or ``None``."""
    if example.get("conversations"):
        instruction, output = _from_conversations(example["conversations"])
    else:
        instruction = (example.get("instruction") or "").strip()
        output = (example.get("output") or "").strip()
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": "", "output": output}


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
    parser.add_argument("--output", default="./data/wizardlm.jsonl", help="Output JSONL path.")
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
