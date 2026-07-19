"""Shape/behaviour tests for the cross-attention fusion layer.

Needs torch (CPU is fine). Verifies the forward contract: logits and attention
maps have the right shapes, a 2-D guide is accepted, and probabilities are valid.
"""

from __future__ import annotations

import torch

from sentinelai.cross_attention import CrossAttentionFusion


def _model() -> CrossAttentionFusion:
    return CrossAttentionFusion(video_dim=2048, guide_dim=768, d_model=128, n_heads=4, n_categories=3)


def test_forward_shapes_with_sequence_guide() -> None:
    """logits are (B, C); attention is (B, S_guide, T_video)."""
    model = _model()
    video = torch.randn(2, 8, 2048)   # B=2, T=8 frames
    guide = torch.randn(2, 5, 768)    # B=2, S=5 guide tokens
    logits, attn = model(video, guide)
    assert logits.shape == (2, 3)
    assert attn.shape == (2, 5, 8)    # each guide token over each video frame


def test_two_dim_guide_is_treated_as_length_one() -> None:
    """A single guide vector per clip (B, D) works and yields S=1 attention."""
    model = _model()
    video = torch.randn(2, 8, 2048)
    guide = torch.randn(2, 768)       # no sequence axis
    logits, attn = model(video, guide)
    assert logits.shape == (2, 3)
    assert attn.shape == (2, 1, 8)


def test_attention_weights_sum_to_one_over_frames() -> None:
    """Attention is a distribution over video frames — each query row sums to 1.

    Checked in eval() mode: attention dropout during training deliberately
    breaks the sum-to-1 invariant.
    """
    model = _model().eval()
    video = torch.randn(1, 6, 2048)
    guide = torch.randn(1, 768)
    _, attn = model(video, guide)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 1), atol=1e-4)


def test_predict_proba_is_in_unit_range() -> None:
    """Sigmoid probabilities lie in [0, 1] with the per-category shape."""
    model = _model()
    proba = model.predict_proba(torch.randn(3, 8, 2048), torch.randn(3, 768))
    assert proba.shape == (3, 3)
    assert (proba >= 0).all() and (proba <= 1).all()
