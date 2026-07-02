# DataPan

![Logo](./assets/datapan.png)

**DataPan** pans for gold in your data — it sifts large instruction-tuning corpora down to the examples that actually make a LLM better. 
Not all training data is worth its weight: some examples teach the model nothing, some even hurt. 
DataPan gives you a single, principled pipeline to score every example, keep the valuable ones, and fine-tune on that distilled subset.

What sets it apart is its **modular** design. Instead of treating every selection
method as a monolithic black box, DataPan factors data selection into a small set
of orthogonal, swappable pieces — so you can recombine published methods from a
config file, or assemble entirely new ones, without rewriting the pipeline.

The end-to-end flow is always the same:

```
parse args → load model+tokenizer → load dataset → select subset
           → (optionally) fine-tune → (optionally) evaluate a benchmark
```

```bash
python main.py --scorer embedding --policy diversity --budget 0.05   # default method
python main.py --method less --budget 0.05 --benchmark gsm8k
python main.py --method random --no-train          # baseline, selection only
```

---

## Modular Design

We decompose *"which data is worth training on"* into three independent axes.
Each axis is a directory of interchangeable plugins; a run is just a choice of
one plugin per axis.

| Axis | Question it answers | Lives in | Plugins |
|------|--------------------|----------|---------|
| **Scorer** | What makes an example valuable? (signal × target × model) | `src/scorer/` | `bm25`, `embedding`, `ppl` |
| **Policy** | How do per-example scores become a subset / weights? | `src/policy/` | `hard`, `diversity`, `reweight`, `greats` |
| **Timing** | *When* does selection run? | (selector) | offline subset · online per-step |

- A **Scorer** maps a dataset to per-example scores, `score(train, val) -> (scores, features)`,
  following the convention *higher == more valuable*. It fuses the three things
  that don't vary independently in practice — the **scoring metric** (lexical,
  representation, perplexity, gradient…), the **comparison target** (a validation
  anchor, the corpus itself, or nothing), and the **scoring model** (none, the
  frozen base model, a LoRA influence model…).
- A **Policy** is blind to *what* a score means; it only turns a vector of scores
  (and optional features) into per-sample weights `wᵢ ≥ 0`. **Hard** selection
  (`{0,1}` top-k mask) and **soft** reweighting (continuous weights) are the two
  ends of one primitive — selection is just the binary special case.
- **Timing** is owned by the selector: *offline* methods score once and hand a
  subset to the generic trainer; *online* methods (ADAPT, GREATS) override
  `make_trainer` to score and reweight **each minibatch** inside the training loop.

This is why two methods that look unrelated on paper often share machinery here:
e.g. ADAPT's online reweighting is just `reweight` policy on the online timing,
and GREATS is the gradient-based, diversity-aware sibling of ADAPT.

```
src/
├── scorer/   # ①②③  what defines value   → get_scorer(cfg, model, tokenizer)
├── policy/   # ④     scores → subset/weights → get_policy(cfg)
├── alg/      # ⑤     selectors: glue scorer+policy, decide timing → get_selector(...)
├── dataset/  #       loaders: dataset/load_<name>.py → get_dataset(cfg, tokenizer)
└── main.py   #       the pipeline
```

Each axis has a name-based registry (`get_scorer` / `get_policy` / `get_selector`)
that imports `<axis>/<name>.py` by config, plus a `BaseScorer` / `BasePolicy` /
`BaseSelector` to subclass. Plugins declare their own CLI flags via an
`add_args(parser)` function, loaded dynamically — so method-specific knobs stay
out of the shared `config.yaml`. Discover them with `--help`:

```bash
python main.py --scorer bm25 --policy diversity --help
python main.py --method less --help
```

---

## Two Ways to Use It

### 1. Modular composition — pick a scorer + a policy (no code)

The `default` selector composes **any** scorer with **any** policy straight from
the config / CLI — it covers the "score the whole set once, then take a subset"
case with zero glue code. It is the **default method**, so `--method` can be
omitted; just choose a scorer and a policy:

```bash
# BM25 lexical relevance, plain top-k
python main.py --scorer bm25 --policy hard --budget 0.05

# perplexity signal, plain top-k
python main.py --scorer ppl --policy hard --budget 1000

# embedding similarity to a validation anchor, kept diverse via k-center coverage
python main.py --scorer embedding --policy diversity
```

Or set it once in `config.yaml`:

```yaml
selection:
  method: default       # the modular selector; alg/default.py
  scorer: embedding     # scorer/<scorer>.py  (null -> bm25)
  policy: diversity      # policy/<policy>.py
  budget: 0.05
```

`--scorer` selects `scorer/<name>.py`, `--policy` selects `policy/<name>.py`, and
both contribute their own tunables (`--bm25-k1`, etc.). Note that interaction-aware
policies like `diversity` need a scorer that exposes `features` (e.g. `embedding`).

> **`--scorer` / `--policy` only apply to the `default` method.** A custom method
> (mode 2 below) wires its own scorer *and* policy, so passing either alongside
> `--method <name>` is ignored and prints a warning.

### 2. DIY — write your own algorithm

When "one scorer → one policy" isn't enough — multiple scorers, gradient
plumbing, a custom reweighting trainer, online timing — drop a file in
`src/alg/<name>.py` that defines a `Selector(BaseSelector)`. You wire the scorer
and policy **in code** by importing the classes you want directly — exactly as
the scorer is imported — so the dependencies are explicit (`get_scorer`/
`get_policy` are reserved for the `default` selector):

```python
# src/alg/my_method.py
from alg.base import BaseSelector
from policy.diversity import Policy           # load the policy directly, like the scorer
from scorer.embedding import Scorer

DEFAULT_POLICY = "diversity"                  # mirror the import so utils.options loads
                                              # the policy's CLI flags; keep the two in sync

class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.scorer = Scorer(cfg, model, tokenizer)
        self.policy = Policy(cfg)             # your fixed ④ policy

    def select(self, train_dataset, val_dataset=None):
        scores, features = self.scorer.score(train_dataset, val_dataset)
        return self.apply_policy(scores, features=features)

    # optional: override for online per-step reweighting instead of an offline subset
    # def make_trainer(self, cfg, model, tokenizer, train_dataset, val_dataset): ...
```

Run it with `--method my_method`. Because the method owns its wiring, the CLI
`--scorer`/`--policy` don't reach it — the scorer and policy are whatever you
imported. If your policy exposes tunables (e.g. `diversity`, `reweight`), declare
a module-level `DEFAULT_POLICY` matching the import so its `add_args` flags load.
Expose method hyper-parameters by adding an `add_args(parser)` function (its
`dest` is the dotted config path it sets, e.g. `selection.warmup_steps`). The
published methods all live in `alg/` this way and are the best reference — see
`alg/less.py` (offline, gradient influence) and `alg/adapt.py` (online reweighting).

---

## Supported Algorithms

| Method | `--method` | Venue | Idea |
|--------|-----------|-------|------|
| **IFD** (Cherry LLM) | `ifd` | NAACL 2024 | Self-guided Instruction-Following Difficulty; quality over quantity. |
| **LESS** | `less` | ICML 2024 | Selecting influential data for *targeted* tuning via LoRA gradient influence. |
| **MIWV** | `miwv` | AAAI 2026 | Rank samples by the ICL-based Model Instruction Weakness Value (training-free). |
| **GREATS** | `greats` | NeurIPS 2024 | **Online** batch selection: keep the most useful *and diverse* size-k subset each step. |
| **ADAPT** | `adapt` | ICLR 2026 | **Online** per-sample reweighting instead of offline subset selection. |

Baselines (also usable directly, or via the default method with `--scorer …`):

- **Random** selection (`--method random`)
- **BM25** lexical relevance (`--method bm25` or `--scorer bm25`)
- **Embedding** similarity (`--method embedding` or `--scorer embedding`)
- **Perplexity** (`--method ppl` or `--scorer ppl`)

## Supported Datasets

Pick with `--dataset <name>`; each loads from `dataset/load_<name>.py` and caches
to `{data_dir}/<name>.jsonl` on first use.

- **Alpaca** — Stanford Alpaca 52k
- **WizardLM** — Evol-Instruct
- **LESS** — mixture of Flan V2, CoT, Dolly and Open Assistant
