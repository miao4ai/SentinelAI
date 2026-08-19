"""Compare fusion at different pipeline depths and report metrics.

The same three experts can be fused at different stages, and *where* you fuse is a
real modelling choice with different trade-offs. This module makes that choice
measurable: it trains one fusion model per depth on the SAME samples and reports
Precision / Recall / F1 / AUC side by side.

This sweep covers four of the five fusion positions (see ``docs/fusion.md``):

    early       one joint block mixing ALL modalities' raw signal -> deep MLP  (① input)
    embedding   concat each expert's encoded features             -> MLP       (③ feature)
    decision    concat each expert's final category scores        -> Decision Tree (④ decision)

plus an untrained **mean-voting baseline** (⑤ vote), to show what the training
actually buys over "just threshold the averaged scores".

**Early fusion** here is the real thing: the raw signals of all modalities are
entangled in one block, so its model must jointly *perceive and fuse* — which is
why it needs a deeper network. Position ② — CLIP-style **embedding model-level**
fusion (separate encoders aligned by contrastive learning, a *coordinated*
representation) — is a different mechanism, implemented in ``clip_screener.py`` and
verified on real video, so it is not part of this synthetic sweep.

Rule of thumb the comparison usually shows: earlier fusion keeps more information
but is higher-dimensional and harder to train; later fusion is low-dimensional,
fast and interpretable (a tree on 9 scores) but cannot recover lost detail.

``scikit-learn`` powers the classifiers (CPU, no GPU), so the whole comparison
runs on synthetic features without touching a real model — see ``synthetic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluate import binary_metrics

# The three feature "levels" a strategy can read, and the dataset attribute each
# one maps to. Keeping this explicit makes a strategy = (which level, which model).
_LEVEL_ATTR = {
    "early": "X_early",
    "embedding": "X_embedding",
    "decision": "X_decision",
}


@dataclass
class FusionDataset:
    """Multimodal features at all three depths plus the binary label.

    Each ``X_*`` is a (num_samples, dim) array holding the *concatenation across
    modalities* of that level's features; ``y`` is 1 for violating, 0 for safe.
    Holding all three levels for the same samples is what lets us compare depths
    fairly — only the input representation changes between strategies.
    """

    X_early: np.ndarray
    X_embedding: np.ndarray
    X_decision: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def split(self, test_frac: float = 0.25, seed: int = 0) -> tuple["FusionDataset", "FusionDataset"]:
        """Shuffle and split into (train, test). The same split feeds every strategy.

        A shared, seeded split is essential: comparing strategies on different
        test rows would measure luck, not the fusion depth.
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        cut = int(len(self) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        sub = lambda i: FusionDataset(
            self.X_early[i], self.X_embedding[i], self.X_decision[i], self.y[i]
        )
        return sub(tr), sub(te)


@dataclass(frozen=True)
class FusionResult:
    """Metrics for one fusion strategy on the held-out set.

    Adds AUC to the precision/recall/F1 triple. AUC (area under the ROC curve)
    needs a *score*, not a hard label, so it is computed from the classifier's
    predicted probabilities — it measures ranking quality independent of the 0.5
    threshold, which precision/recall depend on.
    """

    name: str
    level: str
    structure: str
    precision: float
    recall: float
    f1: float
    auc: float


class _MeanVoteBaseline:
    """Untrained decision-level baseline: flag if the loudest score crosses 0.5.

    Mimics ``sklearn``'s fit/predict/predict_proba API so it slots into the same
    harness. ``fit`` is a no-op; the "probability" is just the max score across the
    concatenated decision vector. This is the "no training at all" reference point.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MeanVoteBaseline":
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X.max(axis=1), 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._score(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._score(X)
        return np.stack([1 - p, p], axis=1)


@dataclass
class FusionStrategy:
    """One fusion approach: a classifier applied at one feature level.

    Attributes:
        name:      label for the results table (e.g. "embedding-mlp").
        level:     which depth's features to read ("embedding"/"logits"/"decision").
        estimator: any classifier exposing sklearn's fit / predict / predict_proba.
        structure: one-line description of the architecture, for the report.
    """

    name: str
    level: str
    estimator: Any
    structure: str = ""

    def _features(self, dataset: FusionDataset) -> np.ndarray:
        """Pull this strategy's input matrix out of the dataset by its level."""
        return getattr(dataset, _LEVEL_ATTR[self.level])

    def fit(self, train: FusionDataset) -> "FusionStrategy":
        """Train the classifier on this level's features. Returns self for chaining."""
        self.estimator.fit(self._features(train), train.y)
        return self

    def evaluate(self, test: FusionDataset) -> FusionResult:
        """Score on the held-out set: precision/recall/F1 from labels, AUC from probs."""
        from sklearn.metrics import roc_auc_score

        X = self._features(test)
        pred = self.estimator.predict(X)
        proba = self.estimator.predict_proba(X)[:, 1]   # P(violating)
        m = binary_metrics(test.y.astype(bool), pred.astype(bool))
        auc = float(roc_auc_score(test.y, proba))
        return FusionResult(self.name, self.level, self.structure, m.precision, m.recall, m.f1, auc)


def default_strategies() -> list[FusionStrategy]:
    """The trained strategies, each with its structure note.

    Model choice is matched to the tier: a shallow Decision Tree suits the tiny,
    interpretable decision vector; a plain MLP suits the clean encoded features; a
    deeper MLP is needed for the entangled early/input block. sklearn is imported
    here (lazily) so importing this module stays cheap.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    return [
        FusionStrategy(
            "early-fusion", "early",
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1200, random_state=0),
            structure="ONE joint block of all modalities' raw signal -> Linear -> ReLU(256) -> ReLU(128) -> 2",
        ),
        FusionStrategy(
            "embedding-mlp", "embedding",
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=800, random_state=0),
            structure="concat 3 experts' encoded features -> Linear -> ReLU(128) -> Linear -> 2",
        ),
        FusionStrategy(
            "decision-tree", "decision",
            DecisionTreeClassifier(max_depth=5, random_state=0),
            structure="concat 3 experts' final category scores -> DecisionTree(max_depth=5)",
        ),
    ]


def all_strategies() -> list[FusionStrategy]:
    """The trained strategies plus the untrained voting baseline, earliest -> latest.

    early-fusion uses a deeper MLP because its input block entangles all modalities
    through a nonlinearity — a shallow head can't perceive+untangle it, which is
    exactly why early fusion needs more model capacity in practice.
    """
    baseline = FusionStrategy(
        "mean-voting", "decision", _MeanVoteBaseline(),
        structure="untrained: flag if max of concatenated decision scores >= 0.5",
    )
    # earliest -> latest: early, embedding, decision(tree), decision(vote).
    return [*default_strategies(), baseline]


def compare_strategies(
    dataset: FusionDataset,
    strategies: list[FusionStrategy] | None = None,
    test_frac: float = 0.25,
    seed: int = 0,
) -> dict[str, FusionResult]:
    """Train every strategy on one shared split and return name -> FusionResult.

    This is the core experiment: identical train/test rows for all strategies, so
    any metric difference is attributable to the fusion depth + model, not the data.
    """
    strategies = strategies or default_strategies()
    train, test = dataset.split(test_frac=test_frac, seed=seed)
    return {s.name: s.fit(train).evaluate(test) for s in strategies}


def format_comparison(results: dict[str, FusionResult]) -> str:
    """Render the results as a metrics table followed by each strategy's structure."""
    header = f"{'strategy':<15}{'level':<11}{'precision':>10}{'recall':>8}{'f1':>7}{'auc':>7}"
    lines = [header, "-" * len(header)]
    for r in results.values():
        lines.append(
            f"{r.name:<15}{r.level:<11}{r.precision:>10.3f}{r.recall:>8.3f}{r.f1:>7.3f}{r.auc:>7.3f}"
        )
    lines.append("")
    lines.append("structures:")
    for r in results.values():
        lines.append(f"  {r.name:<15} {r.structure}")
    return "\n".join(lines)


def run_demo(n_samples: int = 3000, seed: int = 0) -> dict[str, FusionResult]:
    """Build a synthetic dataset, compare all strategies (+baseline), print the table.

    Entry point for ``python -m sentinelai.fusion.compare`` — lets you see the
    framework working end-to-end before any real features exist.
    """
    from .synthetic import make_synthetic_dataset

    dataset = make_synthetic_dataset(n_samples=n_samples, seed=seed)
    results = compare_strategies(dataset, strategies=all_strategies(), seed=seed)
    print(format_comparison(results))
    return results


if __name__ == "__main__":
    run_demo()
