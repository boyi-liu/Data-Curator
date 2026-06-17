"""Shared instruction-tuning formatting and tokenization.

Used by the per-dataset loaders so they stay short and consistent. Records are
expected to look like ``{"instruction": ..., "input": ..., "output": ...}``.
Tokenized examples carry ``input_ids`` / ``attention_mask`` / ``labels`` (with
the prompt masked to ``-100``) and, optionally, the raw ``text`` of the prompt
-- which embedding/BM25-style selectors need to compute their scores.
"""

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{maybe_input_note}. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{input_block}### Response:\n"
)


def format_prompt(example):
    has_input = bool(example.get("input", "").strip())
    return PROMPT_TEMPLATE.format(
        maybe_input_note=", paired with an input" if has_input else "",
        instruction=example["instruction"].strip(),
        input_block=f"### Input:\n{example['input'].strip()}\n\n" if has_input else "",
    )


def make_tokenize_fn(cfg, tokenizer, keep_text=True):
    """Return a ``map`` function that tokenizes one record for causal LM tuning."""
    max_length = cfg.model.max_length

    def tokenize(example):
        prompt = format_prompt(example)
        response = example["output"].strip() + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt + response, add_special_tokens=True)["input_ids"]
        full_ids = full_ids[:max_length]

        labels = list(full_ids)
        # Mask the prompt portion so loss is computed only on the response.
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        out = {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }
        if keep_text:
            out["text"] = prompt
        return out

    return tokenize
