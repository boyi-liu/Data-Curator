"""Minimal benchmark evaluation.

Just enough to close the loop end-to-end. Currently supports GSM8K (exact-match
on the final numeric answer); add new benchmarks by registering a scorer in
``BENCHMARKS``. For serious leaderboard numbers use a dedicated harness
(lm-evaluation-harness, OpenCompass, ...); this is a lightweight sanity check.
"""

import re

import torch

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text):
    matches = _NUM.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def _gsm8k_examples(limit):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    prompts = [f"Question: {q}\nAnswer:" for q in ds["question"]]
    golds = [_last_number(a) for a in ds["answer"]]
    return prompts, golds


BENCHMARKS = {"gsm8k": _gsm8k_examples}


@torch.no_grad()
def evaluate(cfg, model, tokenizer, benchmark="gsm8k", limit=100, max_new_tokens=256):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; choices: {list(BENCHMARKS)}")
    prompts, golds = BENCHMARKS[benchmark](limit)

    model.eval()
    correct = 0
    for prompt, gold in zip(prompts, golds):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=cfg.model.max_length).to(cfg.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if gold is not None and _last_number(gen) == gold:
            correct += 1

    acc = correct / max(1, len(golds))
    return {"benchmark": benchmark, "n": len(golds), "accuracy": acc}
