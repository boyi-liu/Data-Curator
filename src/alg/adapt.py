"""ADAPT: Adaptive Data reweighting for Pretraining and FineTuning (ICLR 2026).

Reference: "Rethinking Data Curation in LLM Training: Online Reweighting Offers
Better Generalization than Offline Methods." (Under review, ICLR 2026.)

Unlike offline selection (LESS, IFD, ...) which freezes a *subset* before
training, ADAPT keeps the **full** dataset and reweights every example *during*
training. The weight ``w_t(i)`` scales the per-sample gradient -- i.e. it is a
per-sample learning-rate multiplier (paper Eq. 7):

    theta_{t+1} = theta_t - eta * sum_{i in B_t} w_t(i) * grad ell(f(x_i), y_i)

The weight comes from a similarity-based quality signal between a training
example and the validation/anchor set ``D_val``. Two scoring functions are
implemented:

  * ``embed`` (ADAPT, the headline method, Sec. 5.3). Uses the model's *own*
    last-layer hidden states. For input x with hidden states {h_1..h_L} we take
    a position-weighted mean pool that up-weights later tokens to counteract the
    causal-mask bias of decoder-only models (Eq. 9):

        w_i = i / sum_j j ,   phi(x) = sum_i w_i h_i ,   phi <- phi / ||phi||

    The score is the mean cosine to the anchor embeddings (Eq. 10):

        s_ADAPT(x) = (1/|D_val|) sum_{v in D_val} cos(phi(x), phi(v))

    Because phi(x) and the anchors are L2-normalized, this equals
    ``phi(x) . centroid`` where ``centroid = mean_v phi(v)`` -- so we cache a
    single anchor centroid. To stay aligned with the *evolving* model, the
    anchors are refreshed every ``R`` steps via forward passes under the current
    parameters (Sec. 5.3, "Online Validation Embedding Updates").

  * ``bm25`` (ADAPT-BM25, the model-agnostic variant, Sec. 5.2). A static sparse
    retrieval signal: s_BM25(x) = mean_v BM25(x, v). Precomputed once.

The score is mapped to an *absolute* weight with a temperature-scaled sigmoid
(Eq. 11). Crucially this is a *global* (per-sample) transform -- it does NOT
normalize over the batch (contrast with softmax weighting), so a sample's weight
depends only on its own similarity, not on its batch-mates:

    w_t(i) = sigmoid( s_ADAPT(x_i) / max(tau, eps) )

Weights are treated as stop-gradient scalar multipliers and clipped for
stability. ADAPT does not change the dataset size, so :meth:`select` returns
*all* indices; the reweighting lives in :class:`ADAPTTrainer`, wired in through
:meth:`BaseSelector.make_trainer`.
"""

import math

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, Trainer

from alg.base import BaseSelector
from utils.model_utils import maybe_wrap_lora
from utils.selector_utils import tqdm


# --------------------------------------------------------------------------- #
# BM25 (dependency-free) for the model-agnostic ADAPT-BM25 signal
# --------------------------------------------------------------------------- #
class _BM25:
    """Okapi BM25 over a small document collection (the anchor/validation set).

    ``score_query_mean`` returns the BM25 score of a query example averaged over
    *all* documents, i.e. s_BM25(x) = (1/|D_val|) sum_v BM25(x, v).
    """

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = max(1, len(docs_tokens))
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) or 1.0
        self.tf = []                       # per-doc {term: count}
        df = {}
        for toks in docs_tokens:
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        # Robertson-Sparck-Jones idf with the standard +1 to keep it non-negative.
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def score_query_mean(self, q_tokens):
        qset = set(q_tokens)
        total = 0.0
        for j in range(self.N):
            tfj, dl = self.tf[j], self.doc_len[j]
            s = 0.0
            for t in qset:
                f = tfj.get(t)
                if f is None:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * f * (self.k1 + 1) / denom
            total += s
        return total / self.N


# --------------------------------------------------------------------------- #
# Reweighter: turns a (batch, anchors) pair into per-sample weights
# --------------------------------------------------------------------------- #
class ADAPTReweighter:
    """Computes the per-sample learning-rate multipliers ``w_t(i)``.

    Owns the anchor centroid (for ``embed``) or the precomputed per-example BM25
    scores (for ``bm25``), the sigmoid gating, and the online anchor refresh.
    """

    def __init__(self, cfg, tokenizer, val_dataset):
        sel = cfg.selection
        self.device = cfg.device
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset

        self.signal = (sel.signal or "embed").lower()
        self.tau = float(sel.tau if sel.tau is not None else 1.0)
        self.eps = float(sel.eps if sel.eps is not None else 1e-8)
        self.refresh_interval = int(sel.refresh_interval or 50)
        self.encode_batch = int(sel.encode_batch or 8)
        self.weight_norm = (sel.weight_norm or "mean").lower()
        self.w_min = float(sel.w_min if sel.w_min is not None else 0.0)
        self.w_max = float(sel.w_max if sel.w_max is not None else 10.0)
        self.standardize = bool(sel.standardize_signal)

        self.needs_hidden = self.signal == "embed"
        self.anchor_centroid = None        # (d,) float32, embed signal only
        self._last_refresh = -1
        self.bm25_scores = None            # (N,) tensor, bm25 signal only

    # ---- embedding pooling (Eq. 9-10) -------------------------------------
    def _pool(self, hidden, attention_mask):
        """Position-weighted mean pool + L2 norm. Later tokens weighted higher."""
        mask = attention_mask.to(torch.float32)              # (B, T)
        pos = torch.cumsum(mask, dim=1) * mask               # 1..n_i on real tokens
        pos = pos / pos.sum(dim=1, keepdim=True).clamp(min=1.0)
        phi = (hidden.to(torch.float32) * pos.unsqueeze(-1)).sum(dim=1)
        return F.normalize(phi, dim=-1, eps=self.eps)        # (B, d)

    @torch.no_grad()
    def _compute_centroid(self, model):
        """Re-encode the anchor set under the *current* model and average phi(v)."""
        was_training = model.training
        model.eval()
        collate = DataCollatorForSeq2Seq(
            self.tokenizer, padding="longest", label_pad_token_id=-100
        )
        feats = []
        keys = ("input_ids", "attention_mask")
        for i in range(0, len(self.val_dataset), self.encode_batch):
            rows = [self.val_dataset[j]
                    for j in range(i, min(i + self.encode_batch, len(self.val_dataset)))]
            enc = collate([{k: r[k] for k in ("input_ids", "attention_mask", "labels")}
                           for r in rows])
            enc = {k: enc[k].to(self.device) for k in keys}
            out = model(**enc, output_hidden_states=True)
            feats.append(self._pool(out.hidden_states[-1], enc["attention_mask"]))
        if was_training:
            model.train()
        # Mean of unit vectors (not re-normalized): s_i = phi_i . centroid recovers
        # the mean cosine over the anchor set exactly.
        return torch.cat(feats, dim=0).mean(dim=0)

    def maybe_refresh(self, model, step):
        if self.signal != "embed":
            return
        due = self.anchor_centroid is None or (
            self.refresh_interval > 0
            and step % self.refresh_interval == 0
            and step != self._last_refresh
        )
        if due:
            self.anchor_centroid = self._compute_centroid(model)
            self._last_refresh = step

    # ---- bm25 precompute (Sec. 5.2) ---------------------------------------
    @staticmethod
    def _tok(text):
        return text.lower().split()

    def precompute_bm25(self, train_dataset):
        if "text" not in train_dataset.column_names:
            raise ValueError(
                "ADAPT-BM25 needs a raw 'text' field on the dataset; use a loader "
                "built on dataset.formatting (keep_text=True)."
            )
        val_text = (self.val_dataset["text"] if "text" in self.val_dataset.column_names
                    else [self.tokenizer.decode(r["input_ids"], skip_special_tokens=True)
                          for r in self.val_dataset])
        bm25 = _BM25([self._tok(t) for t in val_text])
        scores = [bm25.score_query_mean(self._tok(t))
                  for t in tqdm(train_dataset["text"], desc="ADAPT-BM25 scoring")]
        scores = torch.tensor(scores, dtype=torch.float32)
        if self.standardize:
            scores = (scores - scores.mean()) / (scores.std() + self.eps)
        self.bm25_scores = scores

    # ---- weights (Eq. 11) --------------------------------------------------
    def _gate(self, score):
        w = torch.sigmoid(score / max(self.tau, self.eps))
        return w.clamp(min=self.w_min, max=self.w_max).detach()

    def weights(self, hidden, attention_mask, idx, model, step):
        if self.signal == "embed":
            self.maybe_refresh(model, step)
            phi = self._pool(hidden, attention_mask)         # (B, d)
            score = phi @ self.anchor_centroid.to(phi.dtype)  # (B,)
        else:  # bm25
            score = self.bm25_scores.to(attention_mask.device)[idx]
        return self._gate(score)

    # ---- combine per-sample loss with weights -----------------------------
    def combine(self, per_sample_loss, w):
        weighted = w * per_sample_loss
        if self.weight_norm == "sum":            # paper Eq. 7 (absolute, no norm)
            return weighted.sum()
        if self.weight_norm == "zsum":           # Sec. 3.3 L* = (1/Z) sum w*loss
            return weighted.sum() / w.sum().clamp(min=self.eps)
        return weighted.mean()                   # default: keep mean-loss scale


# --------------------------------------------------------------------------- #
# Trainer: applies the per-sample weighted loss in the optimization loop
# --------------------------------------------------------------------------- #
class ADAPTTrainer(Trainer):
    """A :class:`~transformers.Trainer` that reweights the per-sample loss.

    One forward pass yields both the logits (for a per-sample LM loss) and the
    last-layer hidden states (for the ``embed`` signal), so reweighting adds only
    the lightweight in-loop scoring the paper calls ``F^on_metrics``.
    """

    def __init__(self, *args, reweighter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reweighter = reweighter

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        idx = inputs.pop("idx", None)
        labels = inputs["labels"]
        outputs = model(**inputs, output_hidden_states=self.reweighter.needs_hidden)

        # Per-sample LM loss: token-mean CE over the (unmasked) response tokens.
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        B, Tm1, V = logits.shape
        tok_loss = F.cross_entropy(
            logits.reshape(-1, V), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view(B, Tm1)
        valid = (shift_labels != -100).to(tok_loss.dtype)
        per_sample = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        hidden = outputs.hidden_states[-1] if self.reweighter.needs_hidden else None
        w = self.reweighter.weights(
            hidden, inputs["attention_mask"], idx, model, self.state.global_step
        )
        loss = self.reweighter.combine(per_sample, w)
        return (loss, outputs) if return_outputs else loss


class _IdxCollator:
    """Wraps a base collator, threading an integer ``idx`` through the batch.

    Because ADAPT runs with ``remove_unused_columns=False`` (to keep ``idx``
    alive), the collator only forwards the model-relevant keys to the base
    collator -- any extra columns (e.g. ``text``) are dropped.
    ``idx`` lets the BM25 signal look up precomputed per-example scores inside
    :meth:`ADAPTTrainer.compute_loss`.
    """

    MODEL_KEYS = ("input_ids", "attention_mask", "labels")

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        has_idx = "idx" in features[0]
        idx = [int(f["idx"]) for f in features] if has_idx else None
        clean = [{k: f[k] for k in self.MODEL_KEYS if k in f} for f in features]
        batch = self.base(clean)
        if has_idx:
            batch["idx"] = torch.tensor(idx, dtype=torch.long)
        return batch


# --------------------------------------------------------------------------- #
# Selector: returns the full dataset and wires up the reweighting trainer
# --------------------------------------------------------------------------- #
class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if model is None or tokenizer is None:
            raise ValueError("ADAPT needs a model and tokenizer.")
        self.reweighter = None

    # ADAPT keeps the full dataset; "selection" is the identity map.
    def select(self, train_dataset, val_dataset=None):
        if val_dataset is None or len(val_dataset) == 0:
            raise ValueError(
                "ADAPT requires a non-empty validation/anchor set (set "
                "dataset.validation_split > 0)."
            )
        return list(range(len(train_dataset)))

    def make_trainer(self, cfg, model, tokenizer, train_dataset, val_dataset):
        from utils.train_utils import build_training_args

        if val_dataset is None or len(val_dataset) == 0:
            raise ValueError("ADAPT requires a non-empty validation/anchor set.")

        reweighter = ADAPTReweighter(cfg, tokenizer, val_dataset)

        # Operate on LoRA adapters when enabled (matches the paper's LoRA setup).
        model = maybe_wrap_lora(cfg, model)
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        # Gradient checkpointing + LoRA needs inputs to require grad.
        if cfg.train.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if reweighter.signal == "bm25":
            reweighter.precompute_bm25(train_dataset)
            if "idx" not in train_dataset.column_names:
                train_dataset = train_dataset.add_column("idx", list(range(len(train_dataset))))
        else:  # embed: initialize anchors from theta_0 before the first step
            reweighter.maybe_refresh(model, step=0)

        args = build_training_args(cfg)
        # Keep 'idx'/'text' alive so the collator can thread them.
        args.remove_unused_columns = False
        collator = _IdxCollator(
            DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest",
                                   label_pad_token_id=-100)
        )
        self.reweighter = reweighter
        return ADAPTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            reweighter=reweighter,
        )


def add_args(parser):
    """Register ADAPT-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("ADAPT")
    g.add_argument("--adapt-signal", choices=["embed", "bm25"], default="embed",
                   dest="selection.signal",
                   help="Quality signal: 'embed' (model-state, ADAPT) or 'bm25' "
                        "(model-agnostic, ADAPT-BM25).")
    g.add_argument("--adapt-tau", type=float, default=1.0, dest="selection.tau",
                   help="Sigmoid temperature; larger -> flatter weights.")
    g.add_argument("--adapt-eps", type=float, default=1e-8, dest="selection.eps",
                   help="Numerical-stability constant.")
    g.add_argument("--adapt-refresh-interval", type=int, default=50,
                   dest="selection.refresh_interval",
                   help="Refresh the anchor embeddings every R steps (embed signal).")
    g.add_argument("--adapt-encode-batch", type=int, default=8,
                   dest="selection.encode_batch",
                   help="Batch size for anchor-embedding forward passes.")
    g.add_argument("--adapt-weight-norm", choices=["mean", "sum", "zsum"],
                   default="mean", dest="selection.weight_norm",
                   help="How weighted per-sample losses are combined: 'mean' "
                        "(keep mean-loss scale), 'sum' (paper Eq. 7), or 'zsum' "
                        "(normalize by sum of weights, Sec. 3.3).")
    g.add_argument("--adapt-w-min", type=float, default=0.0, dest="selection.w_min",
                   help="Lower clip on per-sample weights.")
    g.add_argument("--adapt-w-max", type=float, default=10.0, dest="selection.w_max",
                   help="Upper clip on per-sample weights (prevents LR explosion).")
    g.add_argument("--adapt-standardize-signal", action="store_true",
                   dest="selection.standardize_signal",
                   help="Standardize the raw score before the sigmoid "
                        "(recommended for the bm25 signal).")
