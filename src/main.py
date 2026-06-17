"""Entry point for the data-selection pipeline.

    python main.py --dataset a --method less --budget 0.05
    python main.py --method random --no-train             # baseline, selection only
    python main.py --dataset a --method less --benchmark gsm8k --eval-limit 200

Flow:
    parse args -> load model+tokenizer -> load dataset -> select subset
               -> (optionally) fine-tune -> (optionally) evaluate a benchmark
"""

import argparse
import json
import os

from dataset import get_dataset
from alg import get_selector
from utils.model_utils import load_model_and_tokenizer
from utils.options import parse_args
from utils.train_utils import set_seed, train as run_training


def _save_json(cfg, name, payload):
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    # Run-level flags layered on top of the config-driven options.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--no-train", action="store_true",
                     help="Run selection only; skip the final fine-tuning.")
    pre.add_argument("--benchmark", default=None,
                     help="Evaluate this benchmark after training (e.g. gsm8k).")
    pre.add_argument("--eval-limit", type=int, default=100,
                     help="Number of benchmark examples to evaluate.")
    known, remaining = pre.parse_known_args()

    cfg = parse_args(remaining)
    set_seed(cfg.seed)

    print(f"[1/5] Loading model & tokenizer: {cfg.model.name}")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print(f"[2/5] Loading dataset: {cfg.dataset.name}")
    data = get_dataset(cfg, tokenizer)
    train_set, val_set = data["train"], data.get("validation")
    print(f"      train={len(train_set)} | "
          f"validation={len(val_set) if val_set is not None else 0}")

    print(f"[3/5] Selecting data with method: {cfg.selection.method}")
    selector = get_selector(cfg, model, tokenizer)
    indices = selector.select(train_set, val_set)
    selected = train_set.select(indices)
    out = _save_json(cfg, "selection.json", {
        "method": cfg.selection.method,
        "budget": cfg.selection.budget,
        "num_selected": len(indices),
        "selected_indices": indices,
    })
    print(f"      kept {len(indices)}/{len(train_set)} examples -> {out}")

    if known.no_train:
        print("[4/5] --no-train set; skipping fine-tuning and evaluation.")
        return

    print(f"[4/5] Fine-tuning on {len(selected)} selected examples")
    # Online methods (e.g. ADAPT) supply their own reweighting trainer; offline
    # selectors return None and fall back to the generic Trainer.
    trainer = selector.make_trainer(cfg, model, tokenizer, selected, val_set)
    run_training(cfg, model, tokenizer, selected, val_set, trainer=trainer)
    print(f"      artifacts in {cfg.output_dir}")

    if known.benchmark:
        print(f"[5/5] Evaluating on {known.benchmark}")
        from utils.eval_utils import evaluate
        result = evaluate(cfg, model, tokenizer, known.benchmark, limit=known.eval_limit)
        _save_json(cfg, "eval.json", result)
        print(f"      {known.benchmark}: accuracy={result['accuracy']:.4f} (n={result['n']})")
    else:
        print("[5/5] No --benchmark given; done.")


if __name__ == "__main__":
    main()
