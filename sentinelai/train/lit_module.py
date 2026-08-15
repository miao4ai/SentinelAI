"""LightningModule that trains the cross-attention fusion head (V2 ch.6.3).

Wraps ``CrossAttentionFusion`` with the training machinery: a multi-label BCE
loss, AdamW, and per-epoch validation metrics (loss, element-wise accuracy, and
macro AUC). Keeping this thin means the fusion architecture stays in one place
(``cross_attention.py``) and only the optimisation lives here.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn

from ..cross_attention import CrossAttentionFusion


class LitCrossAttention(pl.LightningModule):
    """Train/validate the cross-attention head for multi-label violation scoring."""

    def __init__(
        self,
        video_dim: int,
        guide_dim: int,
        n_categories: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = CrossAttentionFusion(
            video_dim=video_dim,
            guide_dim=guide_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_categories=n_categories,
        )
        # BCE-with-logits: multi-label, one independent sigmoid per category.
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, video: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(video, guide)
        return logits

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One gradient step: forward, BCE loss, log it. Lightning handles backward."""
        video, guide, y = batch
        logits = self(video, guide)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Score a val batch; stash probs/labels for a proper epoch-level AUC."""
        video, guide, y = batch
        logits = self(video, guide)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        acc = ((probs >= 0.5).float() == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self._val_probs.append(probs.detach().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Compute macro AUC over the whole val set, then print a tidy epoch line."""
        from sklearn.metrics import roc_auc_score

        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        self._val_probs.clear()
        self._val_labels.clear()

        # AUC per category (skip degenerate columns with a single class), then average.
        aucs = [
            roc_auc_score(labels[:, c], probs[:, c])
            for c in range(labels.shape[1])
            if len(set(labels[:, c])) == 2
        ]
        auc = float(sum(aucs) / len(aucs)) if aucs else float("nan")
        self.log("val_auc", auc, prog_bar=True)
        metrics = self.trainer.callback_metrics
        print(
            f"epoch {self.current_epoch:>2}  "
            f"val_loss={metrics.get('val_loss', float('nan')):.4f}  "
            f"val_acc={metrics.get('val_acc', float('nan')):.3f}  "
            f"val_auc={auc:.3f}"
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
