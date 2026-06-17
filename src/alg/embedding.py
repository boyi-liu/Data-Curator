"""Embedding-similarity baseline.

A naive selector that scores every training example by the cosine similarity of
its embedding to a *reference* embedding, then keeps one tail. Embeddings are
mean-pooled last-layer hidden states from the (frozen) model, so no extra
encoder is required:

    emb(x) = meanpool_t  h_L(x)_t
    score(d) = cos( emb(d), reference )

The reference is the centroid (mean embedding) of the validation set when one is
available -- selecting training data that looks like the target task. With no
validation set it falls back to the centroid of the training set itself, turning
the score into a representativeness measure. The direction is a knob
(``--embedding-select``):

    * ``near`` (default): keep the most similar examples (on-target /
      representative).
    * ``far``: keep the least similar examples (outliers / diversity).

The model weights stay frozen (a single forward pass per batch, no gradients).
Operates on the raw prompt ``text`` field (see dataset/formatting.py). Enable
with ``--method embedding``.
"""

import numpy as np
import torch
import torch.nn.functional as F

from alg.base import BaseSelector
from utils.selector_utils import mean_pool, tqdm


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if model is None or tokenizer is None:
            raise ValueError("Embedding needs a model and tokenizer.")

        self.device = cfg.device
        sel = cfg.selection
        self.direction = sel.embedding_select or "near"
        if self.direction not in ("near", "far"):
            raise ValueError("embedding_select must be 'near' or 'far'.")
        self.encode_batch = int(sel.encode_batch or 16)

        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False

    # ---- BaseSelector API --------------------------------------------------

    def select(self, train_dataset, val_dataset=None):
        if "text" not in train_dataset.column_names:
            raise ValueError(
                "Embedding needs a 'text' field on the dataset; use a loader "
                "built on dataset.formatting (it keeps the raw prompt text)."
            )

        emb = self._encode(train_dataset["text"], desc="encode pool")  # (N, d)

        if val_dataset is not None and len(val_dataset) > 0 \
                and "text" in val_dataset.column_names:
            ref = self._encode(val_dataset["text"], desc="encode val").mean(0)
            source = "validation centroid"
        else:
            ref = emb.mean(0)
            source = "train centroid"
        print(f"[Embedding] scoring {emb.shape[0]} docs vs {source}; "
              f"keeping {self.direction} examples")

        sim = F.cosine_similarity(emb, ref.unsqueeze(0), dim=1)  # (N,)
        scores = sim if self.direction == "near" else -sim
        return self.topk_by_score(scores.tolist())

    # ---- encoding ----------------------------------------------------------

    @torch.no_grad()
    def _encode(self, texts, desc="encode"):
        """Mean-pooled last-layer hidden states as CPU float tensors -> (N, d)."""
        self.model.eval()
        chunks = []
        for i in tqdm(range(0, len(texts), self.encode_batch), desc=f"Embedding {desc}"):
            batch = texts[i:i + self.encode_batch]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.model.max_length,
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            pooled = mean_pool(out.hidden_states[-1], enc["attention_mask"])
            chunks.append(pooled.float().cpu())
        return torch.cat(chunks)


def add_args(parser):
    """Register Embedding-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("Embedding")
    g.add_argument("--embedding-select", choices=["near", "far"], default="near",
                   dest="selection.embedding_select",
                   help="Keep examples most similar (near, default) or least "
                        "similar (far) to the reference centroid.")
    g.add_argument("--embedding-encode-batch", type=int, default=16,
                   dest="selection.encode_batch",
                   help="Batch size for embedding extraction.")
