"""Compare fusion at three pipeline depths and report metrics.

The same three experts can be fused at different stages, and *where* you fuse is a
real modelling choice with different trade-offs. This module makes that choice
measurable: it trains one fusion model per depth on the SAME samples and reports
Precision/Recall/F1 side by side.

The three depths (earliest → latest):

    embedding   concat each expert's backbone feature vector  -> MLP
    logits      concat each expert's pre-activation logits     -> MLP
    decision    concat each expert's final category scores      -> Decision Tree

Rule of thumb the comparison usually shows: earlier fusion keeps more information
(embeddings carry signal the logits/scores have thrown away) but is higher-
dimensional and can overfit with little data; later fusion is low-dimensional,
fast and interpretable (a tree on 9 scores) but cannot recover lost detail.

``scikit-learn`` powers the classifiers (CPU, no GPU), so the whole comparison
runs on synthetic features without touching a real model — see ``synthetic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluate import Metrics, binary_metrics

# The three feature "levels" a strategy can read, and the dataset attribute each
# one maps to. Keeping this explicit makes a strategy = (which level, which model).
_LEVEL_ATTR = {
    "embedding": "X_embedding",
    "logits": "X_logits",
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

    X_embedding: np.ndarray
    X_logits: np.ndarray
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
        sub = lambda i: FusionDataset(self.X_embedding[i], self.X_logits[i], self.X_decision[i], self.y[i])
        return sub(tr), sub(te)


@dataclass
class FusionStrategy:
    """One fusion approach: a sklearn classifier applied at one feature level.

    Attributes:
        name:      label for the results table (e.g. "embedding-mlp").
        level:     which depth's features to read ("embedding"/"logits"/"decision").
        estimator: any fitted-by-`.fit(X, y)` sklearn-style classifier.
    """

    name: str
    level: str
    estimator: Any

    def _features(self, dataset: FusionDataset) -> np.ndarray:
        """Pull this strategy's input matrix out of the dataset by its level."""
        return getattr(dataset, _LEVEL_ATTR[self.level])

    def fit(self, train: FusionDataset) -> "FusionStrategy":
        """Train the classifier on this level's features. Returns self for chaining."""
        self.estimator.fit(self._features(train), train.y)
        return self

    def evaluate(self, test: FusionDataset) -> Metrics:
        """Predict on the held-out set and score with the shared binary metrics."""
        pred = self.estimator.predict(self._features(test))
        return binary_metrics(test.y.astype(bool), pred.astype(bool))


def default_strategies() -> list[FusionStrategy]:
    """The three strategies from the brief: tree@decision, MLP@logits, MLP@embedding.

    Model choice is matched to dimensionality: a shallow Decision Tree suits the
    tiny, interpretable decision vector; MLPs suit the richer logit/embedding
    vectors. sklearn is imported here (lazily) so importing this module stays cheap.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    return [
        FusionStrategy(
            "decision-tree", "decision",
            DecisionTreeClassifier(max_depth=5, random_state=0),
        ),
        FusionStrategy(
            "logit-mlp", "logits",
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=0),
        ),
        FusionStrategy(
            "embedding-mlp", "embedding",
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=800, random_state=0),
        ),
    ]


def compare_strategies(
    dataset: FusionDataset,
    strategies: list[FusionStrategy] | None = None,
    test_frac: float = 0.25,
    seed: int = 0,
) -> dict[str, Metrics]:
    """Train every strategy on one shared split and return name -> Metrics.

    This is the core experiment: identical train/test rows for all strategies, so
    any metric difference is attributable to the fusion depth + model, not the data.
    """
    strategies = strategies or default_strategies()
    train, test = dataset.split(test_frac=test_frac, seed=seed)
    return {s.name: s.fit(train).evaluate(test) for s in strategies}


def format_comparison(results: dict[str, Metrics]) -> str:
    """Render the results dict as an aligned text table for quick reading."""
    header = f"{'strategy':<16}{'precision':>11}{'recall':>9}{'f1':>8}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(f"{name:<16}{m.precision:>11.3f}{m.recall:>9.3f}{m.f1:>8.3f}")
    return "\n".join(lines)


def run_demo(n_samples: int = 3000, seed: int = 0) -> dict[str, Metrics]:
    """Build a synthetic dataset, compare the three strategies, print the table.

    Entry point for ``python -m sentinelai.fusion.compare`` — lets you see the
    framework working end-to-end before any real features exist.
    """
    from .synthetic import make_synthetic_dataset

    dataset = make_synthetic_dataset(n_samples=n_samples, seed=seed)
    results = compare_strategies(dataset, seed=seed)
    print(format_comparison(results))
    return results


if __name__ == "__main__":
    run_demo()
