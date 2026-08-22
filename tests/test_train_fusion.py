"""Tests for the fusion-training synthetic data (ch.4 + early fusion).

Needs torch (CPU). The full training loops are verified via
`python -m sentinelai.train.train_fusion`.
"""

from __future__ import annotations

import torch

from sentinelai.train.train_fusion import (
    MODALITY_DIMS,
    TOKENS_PER,
    make_synthetic_tokens,
    pooled_features,
)


def test_token_shapes_and_labels() -> None:
    """Per-modality token tensors and multi-hot labels have consistent shapes."""
    tokens, labels = make_synthetic_tokens(n_samples=40, seed=3)
    assert labels.shape == (40, 3)
    assert set(labels.unique().tolist()) <= {0.0, 1.0}
    for m, dim in MODALITY_DIMS.items():
        assert tokens[m].shape == (40, TOKENS_PER[m], dim)


def test_pooled_features_concat_width() -> None:
    """Late-fusion view mean-pools each modality then concatenates -> sum of dims."""
    tokens, _ = make_synthetic_tokens(n_samples=10, seed=0)
    feats = pooled_features(tokens)
    assert feats.shape == (10, sum(MODALITY_DIMS.values()))   # 48+24+24 = 96
    assert feats.dtype == torch.float32
