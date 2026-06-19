"""WizardLM Evol-Instruct dataset -- prepare and load in one module.

As a library (the registry calls this): ``load(cfg, tokenizer)`` builds
``{data_dir}/wizardlm.jsonl`` on first use -- the WizardLM Evol-Instruct data
(Xu et al., 2023) reformatted to ``{"instruction", "input", "output"}`` -- then
tokenizes it. Select with ``cfg.dataset.name = wizardlm``.

As a script: ``python -m dataset.load_wizardlm [--dataset_name ...]
[--data_files ...] [--sample_percentage ...]`` (re)builds just the JSONL.

Two on-Hub layouts are handled automatically:
  * ``instruction`` / ``output`` columns (e.g. ``WizardLM_evol_instruct_70k``),
  * ShareGPT ``conversations`` (e.g. ``WizardLM_evol_instruct_V2_196k``) -- the
    first human turn becomes ``instruction`` and its ``gpt`` reply ``output``.

The source defaults to the Hub (``WizardLM_evol_instruct_70k``). Override via
``dataset.source`` (Hub id) or ``dataset.data_files`` (a local file).
"""

import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataset._common import subsample, tokenize_split, write_jsonl

DEFAULT_SOURCE = "WizardLMTeam/WizardLM_evol_instruct_70k"
FILENAME = "wizardlm.jsonl"


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


def build_records(source=DEFAULT_SOURCE, data_files=None, sample_percentage=1.0, seed=3):
    """Download/read the source and return unified, subsampled records."""
    from datasets import load_dataset

    if data_files:
        ext = os.path.splitext(data_files)[1].lstrip(".").lower()
        fmt = {"jsonl": "json", "": "json"}.get(ext, ext)
        raw = load_dataset(fmt, data_files=data_files, split="train")
    else:
        raw = load_dataset(source, split="train")
    records = [r for r in (to_record(ex) for ex in raw) if r is not None]
    return subsample(records, sample_percentage, seed)


def prepare(cfg):
    """Build ``{data_dir}/wizardlm.jsonl`` if missing; return its path."""
    path = os.path.join(cfg.dataset.data_dir, FILENAME)
    if not os.path.exists(path):
        records = build_records(
            source=cfg.dataset.source or DEFAULT_SOURCE,
            data_files=cfg.dataset.data_files,
            sample_percentage=cfg.dataset.sample_percentage or 1.0,
            seed=cfg.seed,
        )
        write_jsonl(records, path)
        print(f"[wizardlm] wrote {len(records)} examples to {path}")
    return path


def load(cfg, tokenizer):
    return tokenize_split(cfg, tokenizer, prepare(cfg))


def _cli():
    p = argparse.ArgumentParser(description="Build the WizardLM JSONL.")
    p.add_argument("--dataset_name", default=DEFAULT_SOURCE, help="HF Hub dataset id.")
    p.add_argument("--data_files", default=None, help="Local file to load instead of the Hub.")
    p.add_argument("--output", default=f"./data/{FILENAME}", help="Output JSONL path.")
    p.add_argument("--sample_percentage", type=float, default=1.0, help="Fraction to keep.")
    p.add_argument("--seed", type=int, default=3, help="Subsample seed.")
    a = p.parse_args()
    records = build_records(a.dataset_name, a.data_files, a.sample_percentage, a.seed)
    write_jsonl(records, a.output)
    print(f"Wrote {len(records)} examples to {a.output}")


if __name__ == "__main__":
    _cli()
