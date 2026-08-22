"""Shape/behaviour tests for the early-fusion joint Transformer.

Needs torch (CPU is fine). Verifies the forward contract: mixed-length,
mixed-dim per-modality token sequences fuse into one joint embedding + logits.
"""

from __future__ import annotations

import torch

from sentinelai.early_fusion import JointFusionTransformer


def _model() -> JointFusionTransformer:
    return JointFusionTransformer(
        modality_dims={"visual": 48, "audio": 24, "text": 24},
        d_model=64, n_heads=4, n_layers=2, n_categories=3,
    )


def test_forward_shapes_with_token_sequences() -> None:
    """Each modality is a (B, T_m, dim_m) sequence; output is (B, C) + (B, d_model)."""
    model = _model()
    inputs = {
        "visual": torch.randn(2, 8, 48),   # 8 image patches
        "audio": torch.randn(2, 5, 24),    # 5 spectrogram blocks
        "text": torch.randn(2, 6, 24),     # 6 text tokens
    }
    logits, joint = model(inputs)
    assert logits.shape == (2, 3)
    assert joint.shape == (2, 64)          # one shared joint embedding


def test_two_dim_modality_is_single_token() -> None:
    """A (B, dim) modality is accepted as one token and fuses with the rest."""
    model = _model()
    inputs = {
        "visual": torch.randn(3, 8, 48),
        "audio": torch.randn(3, 24),       # single audio vector
        "text": torch.randn(3, 6, 24),
    }
    logits, joint = model(inputs)
    assert logits.shape == (3, 3)
    assert joint.shape == (3, 64)


def test_predict_proba_in_unit_range() -> None:
    """Sigmoid probabilities lie in [0, 1] with the per-category shape."""
    model = _model()
    inputs = {m: torch.randn(2, 4, d) for m, d in {"visual": 48, "audio": 24, "text": 24}.items()}
    proba = model.predict_proba(inputs)
    assert proba.shape == (2, 3)
    assert (proba >= 0).all() and (proba <= 1).all()
