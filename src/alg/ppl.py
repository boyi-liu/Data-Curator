"""Perplexity baseline.

A naive, model-only selector: score every example by the perplexity the
(pretrained) model assigns to its *response* tokens and keep one tail of the
distribution. Perplexity is just the exponential of the standard
instruction-tuning loss (the prompt is masked to ``-100`` by the tokenizer, so
``model(**inputs).loss`` already averages cross-entropy over the response):

    ppl(A | Q) = exp( s(A | Q) )

Two opposite intuitions are both common in the literature, so the direction is
a knob (``--ppl-select``):

    * ``high`` (default): keep the highest-perplexity samples -- the answers the
      model still finds hard, i.e. where there is the most to learn.
    * ``low``: keep the lowest-perplexity samples -- the fluent, "clean" answers
      the model is already confident about, a quality/denoising filter.

No warmup, no gradients, no validation set: a single forward pass per example.
Enable with ``--method ppl``.
"""

import numpy as np
import torch

from alg.base import BaseSelector
from utils.selector_utils import model_inputs, tqdm


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if model is None or tokenizer is None:
            raise ValueError("PPL needs a model and tokenizer.")

        self.device = cfg.device
        sel = cfg.selection
        self.direction = sel.ppl_select or "high"
        if self.direction not in ("high", "low"):
            raise ValueError("ppl_select must be 'high' or 'low'.")

        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False

    # ---- BaseSelector API --------------------------------------------------

    def select(self, train_dataset, val_dataset=None):
        ppl = self._perplexities(train_dataset)

        valid = np.isfinite(ppl)
        print(f"[PPL] scored {int(valid.sum())}/{len(ppl)} samples; "
              f"keeping {self.direction}-perplexity examples")

        # topk_by_score keeps the highest score, so for the "low" direction we
        # rank by negative perplexity. Unscorable examples (no response tokens)
        # sink to the bottom either way.
        signed = ppl if self.direction == "high" else -ppl
        scores = np.where(valid, signed, -np.inf)
        return self.topk_by_score(scores.tolist())

    # ---- scoring -----------------------------------------------------------

    @torch.no_grad()
    def _perplexities(self, dataset):
        self.model.eval()
        scores = np.empty(len(dataset), dtype=np.float64)
        for i in tqdm(range(len(dataset)), desc="PPL scoring"):
            loss = self._response_loss(dataset[i])
            scores[i] = np.nan if loss is None else float(np.exp(loss))
        return scores

    def _response_loss(self, example):
        """Mean cross-entropy on the response tokens (prompt masked to -100)."""
        if all(l == -100 for l in example["labels"]):
            return None
        out = self.model(**model_inputs(example, self.device))
        return float(out.loss)


def add_args(parser):
    """Register PPL-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("PPL")
    g.add_argument("--ppl-select", choices=["high", "low"], default="high",
                   dest="selection.ppl_select",
                   help="Keep the highest-perplexity (hard, default) or "
                        "lowest-perplexity (clean) examples.")
