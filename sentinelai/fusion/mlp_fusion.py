"""4.2 (alternative) Trainable MLP fusion over concatenated modality features.

Instead of hand-weighted voting, this learns the combination: take each expert's
last-layer **embedding or logits**, concatenate them into one long vector, and let
a small multi-layer perceptron learn the mapping to violation categories.

Trade-off vs. voting:
  * voting  — works immediately, fully interpretable, but weights are guessed.
  * MLP     — can learn cross-modal interactions (e.g. "violent *sound* only counts
              when the *frame* is also dynamic"), but must be **trained** on labelled
              clips before its outputs mean anything (ROADMAP ch.4 data).

This module is intentionally separate from ``fusion.py`` so the voting path stays
torch-free. Import it explicitly: ``from sentinelai.fusion.mlp_fusion import MLPFusion``.
"""

from __future__ import annotations

import torch
from torch import nn

from .signals import CANONICAL_CATEGORIES


class MLPFusion(nn.Module):
    """A small MLP that maps concatenated modality features to category logits.

    Architecture::

        [cv_feat | audio_feat | text_feat]  (input_dim,)
            -> Linear -> ReLU -> Dropout
            -> Linear -> logits  (num_categories,)

    The caller is responsible for building the input vector by concatenating the
    experts' embeddings/logits in a *fixed order* (and padding with zeros when a
    modality is missing, so ``input_dim`` is constant).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_categories: int = len(CANONICAL_CATEGORIES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # One hidden layer is plenty for late fusion of already-rich features;
        # deeper nets just overfit the (small) labelled fusion set.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),          # regularises the small fusion head
            nn.Linear(hidden_dim, num_categories),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map a (batch, input_dim) feature batch to (batch, num_categories) logits.

        Returns raw logits (no sigmoid/softmax) so the caller can choose the loss
        (BCEWithLogits for multi-label, CrossEntropy for single-label).
        """
        return self.net(features)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience inference: per-category probabilities via sigmoid (multi-label).

        Sigmoid (not softmax) because a clip can be several violation types at once,
        mirroring how the audio/text experts treat their labels.
        """
        self.eval()
        return torch.sigmoid(self.forward(features))
