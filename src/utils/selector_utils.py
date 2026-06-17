"""Shared helpers for data-selection algorithms (``alg/<method>.py``).

Small utilities that several selectors would otherwise re-implement:

* ``tqdm``      -- progress-bar handle with a no-op fallback when tqdm is absent,
                   so modules can ``from utils.selector_utils import tqdm``.
* ``batched``   -- turn one tokenized id list into a ``(1, L)`` model input.
* ``model_inputs`` -- turn a tokenized example into the batched
                   ``input_ids``/``attention_mask``/``labels`` dict a forward
                   pass expects.
* ``mean_pool`` -- mean-pool token features over an attention mask.
"""

import torch

try:  # optional progress bars
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):
        return it


def batched(ids, device):
    """A single id list -> ``(1, L)`` tensor on ``device``."""
    return torch.tensor(ids, device=device).unsqueeze(0)


def model_inputs(example, device, keys=("input_ids", "attention_mask", "labels")):
    """Tokenized example -> dict of ``(1, L)`` tensors on ``device``.

    Only the ``keys`` actually present in ``example`` are included.
    """
    return {k: batched(example[k], device) for k in keys if k in example}


def mean_pool(hidden, mask):
    """Mean-pool token features ``(B, T, d)`` over the attention mask ``(B, T)``."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
