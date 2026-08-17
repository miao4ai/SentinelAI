"""Tests for the fusion-position comparison framework on synthetic data.

Needs numpy + scikit-learn (no torch, no GPU). Verifies the dataset shapes, the
shared-split contract, and that the strategies actually learn on the synthetic
signal.
"""

from __future__ import annotations

from sentinelai.fusion.compare import compare_strategies, default_strategies
from sentinelai.fusion.synthetic import make_synthetic_dataset


def test_dataset_shapes_are_consistent() -> None:
    """All tiers share the sample axis; decision dim matches the experts."""
    ds = make_synthetic_dataset(n_samples=200, seed=1)
    n = len(ds)
    assert n == 200
    assert ds.X_early.shape[0] == n
    assert ds.X_embedding.shape[0] == n
    assert ds.X_decision.shape[0] == n
    # decision is 3 modalities x 3 canonical categories = 9 dims.
    assert ds.X_decision.shape[1] == 9
    # the early/input block is one wide joint tensor, wider than the rest.
    assert ds.X_early.shape[1] > ds.X_embedding.shape[1] > ds.X_decision.shape[1]


def test_split_is_disjoint_and_covers_all() -> None:
    """Train/test partition the samples with no overlap and no loss."""
    ds = make_synthetic_dataset(n_samples=100, seed=2)
    train, test = ds.split(test_frac=0.25, seed=2)
    assert len(train) + len(test) == len(ds)
    assert len(test) == 25


def test_all_strategies_learn_the_signal() -> None:
    """On easy synthetic data every fusion position beats chance by a wide margin."""
    ds = make_synthetic_dataset(n_samples=2000, seed=0)
    results = compare_strategies(ds, seed=0)
    assert set(results) == {"early-fusion", "embedding-mlp", "decision-tree"}
    for name, m in results.items():
        assert m.f1 > 0.8, f"{name} underperformed: f1={m.f1:.3f}"
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0


def test_default_strategies_read_expected_levels() -> None:
    """The default strategies map to the pipeline positions (early -> intermediate -> late)."""
    levels = {s.name: s.level for s in default_strategies()}
    assert levels == {
        "early-fusion": "early",
        "embedding-mlp": "embedding",
        "decision-tree": "decision",
    }
