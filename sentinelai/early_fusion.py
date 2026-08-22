"""Early fusion (position ①): tokenize every modality → one joint Transformer.

True early fusion fuses *before* per-modality encoders finish: all modalities are
turned into tokens, poured into ONE shared Transformer, and cross-modal attention
happens from the first layer. The pooled ``[CLS]`` output is a single **joint
embedding** — one shared representation, not three aligned ones (that would be
CLIP's coordinated style, position ②; see ``docs/fusion.md``).

    image → patches ┐
    audio → spec-blocks ┼─ +modality-type emb → [CLS]+tokens → Transformer → joint emb → logits
    text  → tokens ┘        (one sequence)          ↑ cross-modal attention from layer 1

Heterogeneity (pixels vs waveform vs tokens) is handled by giving each modality its
own thin **projection** into a shared width ``d_model`` — the only per-modality
step. Everything after is joint. This module takes each modality already tokenised
to ``(B, T_m, dim_m)`` (e.g. ViT patches / spectrogram blocks / token embeddings)
and does the fusion; producing those tokens is the caller's job.
"""

from __future__ import annotations

import torch
from torch import nn


class JointFusionTransformer(nn.Module):
    """Fuse per-modality token sequences into one joint embedding + violation logits.

    Args:
        modality_dims: {modality_name: token_dim} — the per-modality token width,
                       and (by insertion order) the fixed order tokens are laid out.
        d_model:       shared Transformer width every modality is projected to.
        n_heads:       attention heads.
        n_layers:      number of joint Transformer encoder layers.
        n_categories:  number of violation categories to score.
        dropout:       dropout inside the encoder.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_categories: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = list(modality_dims)   # fixed token order

        # The ONLY per-modality step: a thin projection turning each modality's
        # tokens into the shared width, so heterogeneous inputs become comparable.
        self.projections = nn.ModuleDict(
            {m: nn.Linear(dim, d_model) for m, dim in modality_dims.items()}
        )
        # A learned "which modality is this token" vector, added to every token so
        # the joint Transformer knows a token's source.
        self.modality_emb = nn.Embedding(len(modality_dims), d_model)
        # A learned [CLS] token; its output after the encoder is the joint embedding.
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # One SHARED encoder over the concatenated token sequence — this is where
        # early fusion happens: every layer attends across all modalities at once.
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_categories)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse and classify.

        Args:
            inputs: {modality: tokens}. Each tensor is ``(B, T_m, dim_m)``; a 2-D
                    ``(B, dim_m)`` is accepted as a single (T_m=1) token.

        Returns:
            logits:         ``(B, n_categories)`` pre-sigmoid violation scores.
            joint_embedding: ``(B, d_model)`` the fused [CLS] representation.
        """
        batch = next(iter(inputs.values())).shape[0]
        # Start the sequence with the shared [CLS] token.
        seq = [self.cls.expand(batch, -1, -1)]

        for idx, modality in enumerate(self.modalities):
            x = inputs[modality]
            if x.dim() == 2:                     # (B, dim) -> one token (B, 1, dim)
                x = x.unsqueeze(1)
            x = self.projections[modality](x)    # (B, T_m, d_model)
            # Add this modality's type embedding to each of its tokens.
            ids = torch.full(x.shape[:2], idx, dtype=torch.long, device=x.device)
            seq.append(x + self.modality_emb(ids))

        tokens = torch.cat(seq, dim=1)           # (B, 1 + ΣT_m, d_model)
        encoded = self.encoder(tokens)           # joint cross-modal attention
        joint = self.norm(encoded[:, 0])         # [CLS] output = joint embedding
        return self.head(joint), joint

    @torch.no_grad()
    def predict_proba(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-category violation probabilities (sigmoid — multi-label)."""
        logits, _ = self.forward(inputs)
        return torch.sigmoid(logits)
