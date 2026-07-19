"""V2 ch.6 — cross-attention deep fusion (audio/text guides video).

V1 fused the experts *after* each had decided (late fusion). That throws away
*where* and *when* a cue occurred. V2 fuses earlier and deeper: a cross-attention
layer lets one modality **attend into** another's per-frame features.

The guiding idea (ROADMAP 6.2):

    Query  (Q) = audio / text features   — "what am I looking for?"
    Key/Value (K,V) = video frame sequence — "where in the video is it?"

So an audio scream (Q) attends over the video frames (K/V) and pulls out the frames
that matter — "when I hear a scream, focus on *these* frames". Because K/V is the
**temporal** frame sequence (from VideoMAE / TimeSformer, ch.6.1), the model sees
motion across frames and can tell "cutting vegetables" from "stabbing" — the
confusion a single frame cannot resolve.

This module is the fusion layer only; it consumes pre-extracted features:
    * ``video_seq`` (B, T, video_dim)   — per-frame temporal features (K/V)
    * ``guide``     (B, guide_dim) or (B, S, guide_dim) — audio/text features (Q)
and emits per-category violation logits plus the attention map (which frames the
guide attended to — free explainability). Training it (ch.6.3, PyTorch Lightning)
is a separate step.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """One cross-attention block: guide (Q) attends into the video sequence (K/V).

    Args:
        video_dim:    feature dim of each video frame (K/V source).
        guide_dim:    feature dim of the audio/text guide (Q source).
        d_model:      shared attention width both sides are projected to.
        n_heads:      number of attention heads.
        n_categories: number of violation categories to score.
        dropout:      dropout inside attention and the feed-forward block.
    """

    def __init__(
        self,
        video_dim: int,
        guide_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_categories: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Project the two modalities into one shared space so attention can compare
        # them: video frames become the keys/values, the guide becomes the query.
        self.video_proj = nn.Linear(video_dim, d_model)
        self.guide_proj = nn.Linear(guide_dim, d_model)

        # Cross-attention: query attends over the video frames. batch_first so
        # tensors stay (batch, seq, dim). Returns attention weights for free.
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # A standard transformer feed-forward block after attention.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Classify the fused representation into per-category violation logits.
        self.head = nn.Linear(d_model, n_categories)

    def forward(
        self, video_seq: torch.Tensor, guide: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse and classify.

        Args:
            video_seq: (B, T, video_dim) temporal frame features -> keys/values.
            guide:     (B, guide_dim) or (B, S, guide_dim) audio/text query. A 2-D
                       guide (one vector per clip) is treated as a length-1 sequence.

        Returns:
            logits:    (B, n_categories) — pre-sigmoid violation scores.
            attention: (B, S, T) — how much each query token attends to each video
                       frame. Averaged over heads; useful to *see* which frames the
                       guide focused on.
        """
        if guide.dim() == 2:
            guide = guide.unsqueeze(1)  # (B, guide_dim) -> (B, 1, guide_dim)

        v = self.video_proj(video_seq)  # (B, T, d_model)  keys & values
        q = self.guide_proj(guide)      # (B, S, d_model)  queries

        # Query attends over the frames. attn_out has the query's shape; attn_w
        # tells us, per query token, the weight on each video frame.
        attn_out, attn_w = self.attn(q, v, v, need_weights=True)
        x = self.norm1(q + attn_out)          # residual + norm
        x = self.norm2(x + self.ffn(x))       # feed-forward + residual + norm

        pooled = x.mean(dim=1)                 # (B, d_model) collapse query tokens
        logits = self.head(pooled)             # (B, n_categories)
        return logits, attn_w

    @torch.no_grad()
    def predict_proba(self, video_seq: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Per-category violation probabilities (sigmoid — multi-label, like the experts)."""
        logits, _ = self.forward(video_seq, guide)
        return torch.sigmoid(logits)
