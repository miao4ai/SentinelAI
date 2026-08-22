"""Shape/behaviour tests for CLIP-style coordinated fusion (position ②).

Needs torch (CPU). Verifies per-modality embeddings fuse into per-category logits.
"""

from __future__ import annotations

import torch

from sentinelai.coordinated_fusion import CoordinatedFusion


def _model() -> CoordinatedFusion:
    return CoordinatedFusion(
        modality_dims={"visual": 48, "audio": 24, "text": 24}, d_model=64, n_categories=3
    )


def test_forward_shape() -> None:
    """One embedding vector per modality -> (B, n_categories) logits."""
    model = _model()
    inputs = {"visual": torch.randn(4, 48), "audio": torch.randn(4, 24), "text": torch.randn(4, 24)}
    logits = model(inputs)
    assert logits.shape == (4, 3)


def test_predict_proba_in_unit_range() -> None:
    """Sigmoid probabilities lie in [0, 1]."""
    model = _model()
    inputs = {"visual": torch.randn(2, 48), "audio": torch.randn(2, 24), "text": torch.randn(2, 24)}
    proba = model.predict_proba(inputs)
    assert proba.shape == (2, 3)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_separate_encoders_per_modality() -> None:
    """Coordinated fusion keeps one encoder per modality (not a single joint one)."""
    model = _model()
    assert set(model.encoders.keys()) == {"visual", "audio", "text"}
