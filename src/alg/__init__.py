"""Selection-algorithm registry.

Each selection algorithm lives in ``alg/<method>.py`` and defines a class named
``Selector`` (a :class:`~alg.base.BaseSelector` subclass).
``cfg.selection.method`` picks which one to use.
"""

import importlib

from alg.base import BaseSelector

__all__ = ["BaseSelector", "get_selector"]


def get_selector(cfg, model=None, tokenizer=None):
    method = cfg.selection.method
    try:
        module = importlib.import_module(f"alg.{method}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown selection method {method!r}: expected a module alg/{method}.py"
        ) from e
    selector_cls = getattr(module, "Selector", None)
    if selector_cls is None:
        raise AttributeError(f"alg/{method}.py must define a `Selector` class")
    return selector_cls(cfg, model, tokenizer)
