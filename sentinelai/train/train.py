"""Entry point: train the cross-attention fusion head (V2 ch.6.3).

Runs end-to-end on synthetic features so the loop can be verified with no GPU and
no real data — ``python -m sentinelai.train.train``. To train for real, replace
``make_synthetic_crossattn`` with cached expert features of the same shapes; the
LightningModule and Trainer below stay identical.
"""

from __future__ import annotations

import lightning.pytorch as pl

from .datamodule import FusionFeaturesDataModule, make_synthetic_crossattn
from .lit_module import LitCrossAttention

# Small dims keep the synthetic run fast on CPU; real features would be larger
# (e.g. video_dim=2048 from I3D/VideoMAE, guide_dim=768 from AST/text).
VIDEO_DIM = 64
GUIDE_DIM = 32
N_FRAMES = 6
N_CATEGORIES = 3


def train(max_epochs: int = 15, seed: int = 0) -> LitCrossAttention:
    """Build synthetic data, fit the cross-attention head, return the trained module."""
    pl.seed_everything(seed, verbose=False)

    video, guide, labels = make_synthetic_crossattn(
        video_dim=VIDEO_DIM, guide_dim=GUIDE_DIM, n_frames=N_FRAMES,
        n_categories=N_CATEGORIES, seed=seed,
    )
    dm = FusionFeaturesDataModule(video, guide, labels, batch_size=64, seed=seed)
    model = LitCrossAttention(
        video_dim=VIDEO_DIM, guide_dim=GUIDE_DIM, n_categories=N_CATEGORIES
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",           # GPU if present, else CPU
        logger=False,                 # no external logging for the demo
        enable_checkpointing=False,
        enable_progress_bar=False,    # we print our own tidy per-epoch line
        enable_model_summary=False,
    )
    trainer.fit(model, dm)
    return model


if __name__ == "__main__":
    train()
