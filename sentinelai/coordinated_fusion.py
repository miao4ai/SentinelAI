"""Embedding model-level fusion (position ②): coordinated representation, CLIP-style.

Unlike early fusion (one joint encoder) or feature fusion (concat → MLP), the
**coordinated** style keeps a **separate encoder per modality** and fuses by
putting them into a **shared embedding space** and comparing by cosine similarity —
exactly CLIP's idea, at "embedding model-level".

CLIP aligns image↔text pairs with a contrastive loss and classifies zero-shot by
comparing an image embedding to *text-prompt* embeddings. The trainable classifier
analogue here:

    each modality → own encoder → L2-normalise ─┐
                                                 ├─ cosine sim to learned CLASS
    learned class prototypes → L2-normalise ─────┘   prototypes (× temperature)
                                                 → average similarities over modalities → logits

So every modality lands in one shared space and is scored against the same learned
class anchors (the prototypes play the role of CLIP's text embeddings). The
encoders stay **separate** (coordinated), and the "alignment" is what training
learns. (CLIP proper also adds a contrastive pair loss to align modalities to each
other; that is an optional extension on top of this.)

Input is one **embedding vector per modality** ``{m: (B, dim_m)}`` (e.g. each
expert's pooled backbone feature) — embedding-level, not raw tokens.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CoordinatedFusion(nn.Module):
    """CLIP-style coordinated fusion: per-modality encoders → shared space → prototypes.

    Args:
        modality_dims: {modality: embedding_dim} — per-modality input width & order.
        d_model:       shared embedding space width.
        n_categories:  number of violation categories (learned class prototypes).
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        d_model: int = 128,
        n_categories: int = 3,
    ) -> None:
        super().__init__()
        self.modalities = list(modality_dims)
        # SEPARATE encoder per modality (this is what makes it "coordinated", not
        # one shared joint encoder). Each maps its modality into the shared space.
        self.encoders = nn.ModuleDict(
            {
                m: nn.Sequential(nn.Linear(dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
                for m, dim in modality_dims.items()
            }
        )
        # Learned class anchors in the shared space — the role CLIP's text-prompt
        # embeddings play. A modality is "class c" if it points at prototype c.
        self.prototypes = nn.Parameter(torch.randn(n_categories, d_model) * 0.02)
        # Learned temperature (CLIP's logit_scale): sharpens the cosine similarities.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score each modality against the class prototypes, average → logits.

        Args:
            inputs: {modality: (B, dim_m)} one embedding vector per modality.

        Returns:
            logits: (B, n_categories) — mean over modalities of the temperature-
                    scaled cosine similarity to each class prototype.
        """
        scale = self.logit_scale.exp().clamp(max=100.0)
        protos = F.normalize(self.prototypes, dim=-1)          # (C, d) unit anchors

        per_modality = []
        for modality in self.modalities:
            emb = self.encoders[modality](inputs[modality])    # (B, d)
            emb = F.normalize(emb, dim=-1)                     # onto the unit sphere
            per_modality.append(scale * emb @ protos.t())      # (B, C) cosine sims
        # Fuse: average each modality's vote in the shared space.
        return torch.stack(per_modality, dim=0).mean(dim=0)    # (B, C)

    @torch.no_grad()
    def predict_proba(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-category violation probabilities (sigmoid — multi-label)."""
        return torch.sigmoid(self.forward(inputs))
