"""Tests for the training data plumbing (V2 ch.6.3).

Covers the synthetic feature generator's shapes and label validity — needs torch
but not lightning, so it runs fast anywhere torch is installed. The full training
loop is verified separately via `python -m sentinelai.train.train`.
"""

from __future__ import annotations

import torch

from sentinelai.train.datamodule import make_synthetic_crossattn


def test_synthetic_shapes_and_labels() -> None:
    """Generator returns aligned (video, guide, labels) with valid multi-hot labels."""
    video, guide, labels = make_synthetic_crossattn(
        n_samples=50, video_dim=64, guide_dim=32, n_frames=6, n_categories=3, seed=1
    )
    assert video.shape == (50, 6, 64)   # (N, frames, video_dim)  -> K/V
    assert guide.shape == (50, 32)      # (N, guide_dim)          -> Q
    assert labels.shape == (50, 3)      # (N, n_categories)       multi-hot
    assert set(labels.unique().tolist()) <= {0.0, 1.0}
    assert video.dtype == torch.float32


def test_synthetic_is_deterministic() -> None:
    """Same seed reproduces the exact tensors (reproducible experiments)."""
    a = make_synthetic_crossattn(n_samples=20, seed=7)
    b = make_synthetic_crossattn(n_samples=20, seed=7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
