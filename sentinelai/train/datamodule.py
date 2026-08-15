"""Data plumbing for training the cross-attention fusion head.

Provides a Lightning ``DataModule`` over three tensors — video frame sequences
(K/V), guide vectors (Q), and multi-label targets — plus a synthetic generator so
the training loop can be validated on CPU before real features exist.

The synthetic task is designed to actually *need* cross-attention: each category
has a fixed signature in guide-space and in video-frame-space, and when a category
is active its signature is planted in the guide vector AND in a couple of random
"event frames". To predict the label the model must let the guide (which category
am I asking about?) attend to the frames that carry that category's signature —
exactly what the cross-attention layer is for.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:  # Lightning is optional at import time (pure helpers don't need it).
    import lightning.pytorch as pl
    _LightningDataModule = pl.LightningDataModule
except Exception:  # pragma: no cover - only hit when lightning is absent
    _LightningDataModule = object


def make_synthetic_crossattn(
    n_samples: int = 2400,
    video_dim: int = 64,
    guide_dim: int = 32,
    n_frames: int = 6,
    n_categories: int = 3,
    p_active: float = 0.3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (video_seq, guide, labels) where the label needs cross-attention.

    Returns:
        video: (N, n_frames, video_dim) float32 — frame features (K/V).
        guide: (N, guide_dim) float32 — audio/text query (Q).
        labels: (N, n_categories) float32 multi-hot targets.
    """
    rng = np.random.default_rng(seed)
    # Fixed per-category "signatures" in guide-space and in video-frame-space.
    guide_dirs = rng.normal(size=(n_categories, guide_dim))
    video_dirs = rng.normal(size=(n_categories, video_dim))

    labels = (rng.random((n_samples, n_categories)) < p_active).astype(np.float32)
    guide = rng.normal(0.0, 1.0, size=(n_samples, guide_dim)).astype(np.float32)
    video = rng.normal(0.0, 1.0, size=(n_samples, n_frames, video_dim)).astype(np.float32)

    for i in range(n_samples):
        for c in range(n_categories):
            if labels[i, c]:
                # Plant the category signature in the guide...
                guide[i] += guide_dirs[c] * 2.0
                # ...and in 1-2 random "event frames" of the video.
                k = int(rng.integers(1, 3))
                frames = rng.choice(n_frames, size=k, replace=False)
                video[i, frames] += video_dirs[c] * 2.0

    return torch.from_numpy(video), torch.from_numpy(guide), torch.from_numpy(labels)


class FusionFeaturesDataModule(_LightningDataModule):
    """Wrap in-memory (video, guide, labels) tensors into train/val DataLoaders.

    Swapping in real data later means constructing this with cached expert features
    instead of the synthetic tensors — the training code above it does not change.
    """

    def __init__(
        self,
        video: torch.Tensor,
        guide: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 64,
        val_frac: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.video = video
        self.guide = guide
        self.labels = labels
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        """Deterministically split the samples into train / val TensorDatasets."""
        n = len(self.labels)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)
        cut = int(n * (1 - self.val_frac))
        tr, va = perm[:cut], perm[cut:]
        self.train_ds = TensorDataset(self.video[tr], self.guide[tr], self.labels[tr])
        self.val_ds = TensorDataset(self.video[va], self.guide[va], self.labels[va])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size)
