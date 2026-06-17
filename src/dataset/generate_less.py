"""Build the LESS four-dataset instruction-tuning mixture.

Reproduces the training pool used by

  * LESS: Selecting Influential Data for Targeted Instruction Tuning
    (Xia et al., 2024)
  * Rethinking Data Curation in LLM Training: Online Reweighting Offers
    Better Generalization than Offline Methods

namely a mix of **FLAN V2**, **CoT**, **Dolly** and **OpenAssistant1**.

Both papers build on the open-instruct / Tulu pre-processing, where each source
is reformatted into a uniform ``messages`` record::

    {"dataset": "flan_v2",
     "id": "flan_v2_42",
     "messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

This script consumes those four ``{name}_data.jsonl`` files, flattens each into
the framework's ``{"instruction", "input", "output"}`` schema (see
``dataset/formatting.py``), subsamples every source by ``--sample_percentage``
with a fixed ``--seed`` (LESS uses 5% / seed 3 for the warmup model), shuffles
the union, and writes a single ``less.jsonl``.

Getting the processed source files
----------------------------------
The four ``*_data.jsonl`` files are produced by open-instruct's
``reformat_datasets.py`` (run via ``scripts/data/prepare_train_data.sh``), and
are exactly what the LESS repo ships under ``data/train/processed``::

    data/train/processed/
      flan_v2/flan_v2_data.jsonl
      cot/cot_data.jsonl
      dolly/dolly_data.jsonl
      oasst1/oasst1_data.jsonl

Point ``--processed_dir`` at a directory holding the four ``{name}_data.jsonl``
files (flat or in per-dataset subfolders -- both layouts are searched).

Usage
-----
    # full mixture
    python -m dataset.generate_less --processed_dir ./raw --output ./data/less.jsonl

    # LESS warmup pool: 5% of each source, seed 3
    python -m dataset.generate_less --processed_dir ./raw \
        --sample_percentage 0.05 --seed 3 --output ./data/less.jsonl

Then train on it via the existing loader, e.g. point ``dataset.load_a`` at the
written file or add a ``load_less`` that reads ``less.jsonl``.
"""

import argparse
import json
import os
import random

DATASETS = ["flan_v2", "cot", "dolly", "oasst1"]


# --------------------------------------------------------------------------- #
# messages -> {instruction, input, output}
# --------------------------------------------------------------------------- #
def messages_to_record(messages):
    """Flatten a Tulu ``messages`` conversation into a single SFT record.

    The response is the final assistant turn; any preceding turns (system /
    earlier user+assistant exchanges) are folded into ``instruction`` with role
    markers so multi-turn examples (common in OpenAssistant1) are preserved as
    context. ``input`` is left empty -- the conversation lives in ``instruction``.
    Returns ``None`` if there is no assistant turn to predict.
    """
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("content", "").strip():
            last_assistant = i
            break
    if last_assistant is None:
        return None

    output = messages[last_assistant]["content"].strip()

    history = messages[:last_assistant]
    if len(history) == 1 and history[0].get("role") == "user":
        instruction = history[0]["content"].strip()
    else:
        # System prompt + multi-turn history -> a single tagged transcript.
        role_tag = {"system": "System", "user": "User", "assistant": "Assistant"}
        parts = []
        for m in history:
            content = m.get("content", "").strip()
            if not content:
                continue
            parts.append(f"{role_tag.get(m.get('role'), m.get('role', ''))}: {content}")
        instruction = "\n\n".join(parts)

    if not instruction:
        return None
    return {"instruction": instruction, "input": "", "output": output}


# --------------------------------------------------------------------------- #
# source readers
# --------------------------------------------------------------------------- #
def _find_processed_file(processed_dir, name):
    """Locate ``{name}_data.jsonl`` in a flat or per-dataset-subfolder layout."""
    candidates = [
        os.path.join(processed_dir, f"{name}_data.jsonl"),
        os.path.join(processed_dir, name, f"{name}_data.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def read_processed(processed_dir, name):
    """Yield unified records from a Tulu-format ``{name}_data.jsonl`` file."""
    path = _find_processed_file(processed_dir, name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{name}_data.jsonl' under {processed_dir!r}. Expected "
            f"{processed_dir}/{name}_data.jsonl or {processed_dir}/{name}/{name}_data.jsonl. "
            "See the module docstring for how to produce the processed files."
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = messages_to_record(obj.get("messages", []))
            if record is not None:
                yield record


# --------------------------------------------------------------------------- #
# mixing
# --------------------------------------------------------------------------- #
def subsample(records, percentage, rng):
    """Randomly keep ``percentage`` of ``records``."""
    if percentage < 1.0:
        k = int(round(len(records) * percentage))
        if k < len(records):
            return rng.sample(records, k)
    return list(records)


def build_mixture(args):
    rng = random.Random(args.seed)
    combined = []
    stats = {}
    for name in args.datasets:
        records = list(read_processed(args.processed_dir, name))
        kept = subsample(records, args.sample_percentage, rng)
        for r in kept:
            r["dataset"] = name
        stats[name] = (len(records), len(kept))
        combined.extend(kept)

    rng.shuffle(combined)
    return combined, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--processed_dir",
        default="./raw",
        help="Directory with the four Tulu-format {name}_data.jsonl files.",
    )
    parser.add_argument("--output", default="./data/less.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Subset of sources to mix (default: all four).",
    )
    parser.add_argument(
        "--sample_percentage",
        type=float,
        default=1.0,
        help="Fraction of each source to keep (LESS warmup uses 0.05).",
    )
    parser.add_argument("--seed", type=int, default=3, help="Subsample/shuffle seed (LESS uses 3).")
    args = parser.parse_args()

    combined, stats = build_mixture(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(combined)} examples to {args.output}")
    print(f"{'dataset':<12}{'available':>12}{'kept':>10}")
    for name in args.datasets:
        avail, kept = stats[name]
        print(f"{name:<12}{avail:>12}{kept:>10}")


if __name__ == "__main__":
    main()
