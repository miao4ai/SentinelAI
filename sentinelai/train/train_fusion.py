"""Train the chapter-4 late-fusion MLP and the early-fusion Transformer.

Both are trained with PyTorch Lightning on ONE synthetic token dataset, viewed two
ways, so the run also *compares* them fairly on the same signal:

* **late fusion** (`MLPFusion`, ch.4) sees each modality's tokens **mean-pooled**
  then **concatenated** — a single feature vector (position ③, but pooled to a
  fixed length like V1's late fusion input).
* **early fusion** (`JointFusionTransformer`, position ①) sees the **raw token
  sequences** and fuses them with cross-modal attention from layer 1.

The label signal is planted in a few specific tokens, so early fusion (which can
attend to those tokens) has a fair shot at beating pooled late fusion. Swap the
synthetic generator for real tokenised features to train for real.

Run: ``python -m sentinelai.train.train_fusion``
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..early_fusion import JointFusionTransformer
from ..fusion.mlp_fusion import MLPFusion

MODALITY_DIMS = {"visual": 48, "audio": 24, "text": 24}
TOKENS_PER = {"visual": 8, "audio": 5, "text": 6}
N_CATEGORIES = 3


# -- synthetic data ---------------------------------------------------------

def make_synthetic_tokens(
    n_samples: int = 2400, p_active: float = 0.3, seed: int = 0
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Per-modality token sequences whose labels are readable only from a few tokens.

    Each active category plants its signature into ONE random token of every
    modality, so a model must find those tokens (attention) or pool cleverly.
    Returns ``(tokens, labels)`` — tokens[m] is (N, T_m, dim_m); labels is (N, C).
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random((n_samples, N_CATEGORIES)) < p_active).astype(np.float32)
    tokens: dict[str, torch.Tensor] = {}
    for modality, dim in MODALITY_DIMS.items():
        n_tok = TOKENS_PER[modality]
        x = rng.normal(0.0, 1.0, size=(n_samples, n_tok, dim)).astype(np.float32)
        signatures = rng.normal(size=(N_CATEGORIES, dim))          # per-category token signature
        for i in range(n_samples):
            for c in range(N_CATEGORIES):
                if labels[i, c]:
                    t = int(rng.integers(0, n_tok))
                    x[i, t] += signatures[c] * 2.0
        tokens[modality] = torch.from_numpy(x)
    return tokens, torch.from_numpy(labels)


def pooled_features(tokens: dict[str, torch.Tensor]) -> torch.Tensor:
    """Late-fusion view: mean-pool each modality's tokens, then concatenate."""
    return torch.cat([t.mean(dim=1) for t in tokens.values()], dim=1)   # (N, Σdim)


class _TokenDataset(Dataset):
    """Yields ({modality: tokens_i}, label_i); default collate batches the dict."""

    def __init__(self, tokens: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        self.tokens = tokens
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {m: t[i] for m, t in self.tokens.items()}, self.labels[i]


def _split(n: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:]


# -- shared Lightning base --------------------------------------------------

class _LitFusionBase(pl.LightningModule):
    """Common BCE training + val loss/acc/macro-AUC. Subclasses map a batch->logits."""

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._lr = lr
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _logits_labels(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        logits, y = self._logits_labels(batch)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        logits, y = self._logits_labels(batch)
        probs = torch.sigmoid(logits)
        self.log("val_loss", self.loss_fn(logits, y), prog_bar=True)
        self.log("val_acc", ((probs >= 0.5).float() == y).float().mean(), prog_bar=True)
        self._val_probs.append(probs.detach().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        from sklearn.metrics import roc_auc_score

        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        self._val_probs.clear()
        self._val_labels.clear()
        aucs = [
            roc_auc_score(labels[:, c], probs[:, c])
            for c in range(labels.shape[1])
            if len(set(labels[:, c])) == 2
        ]
        auc = float(sum(aucs) / len(aucs)) if aucs else float("nan")
        self.log("val_auc", auc, prog_bar=True)
        m = self.trainer.callback_metrics
        print(
            f"[{type(self).__name__}] epoch {self.current_epoch:>2}  "
            f"val_loss={m.get('val_loss', float('nan')):.4f}  "
            f"val_acc={m.get('val_acc', float('nan')):.3f}  val_auc={auc:.3f}"
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self._lr)


class LitMLPFusion(_LitFusionBase):
    """Chapter-4 late fusion: MLP over pooled+concatenated modality features."""

    def __init__(self, input_dim: int, n_categories: int = N_CATEGORIES, lr: float = 1e-3) -> None:
        super().__init__(lr)
        self.model = MLPFusion(input_dim=input_dim, num_categories=n_categories)

    def _logits_labels(self, batch):
        features, y = batch
        return self.model(features), y


class LitEarlyFusion(_LitFusionBase):
    """Early fusion: joint Transformer over per-modality token sequences."""

    def __init__(self, modality_dims: dict[str, int], n_categories: int = N_CATEGORIES, lr: float = 1e-3) -> None:
        super().__init__(lr)
        self.model = JointFusionTransformer(
            modality_dims, d_model=128, n_layers=2, n_categories=n_categories
        )

    def _logits_labels(self, batch):
        tokens, y = batch
        logits, _ = self.model(tokens)
        return logits, y


# -- training entry points --------------------------------------------------

def _trainer(max_epochs: int) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs, accelerator="auto", logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )


def train_both(n_samples: int = 2400, max_epochs: int = 15, seed: int = 0) -> None:
    """Train late fusion (ch.4) and early fusion on the same synthetic tokens."""
    pl.seed_everything(seed, verbose=False)
    tokens, labels = make_synthetic_tokens(n_samples=n_samples, seed=seed)
    tr, va = _split(n_samples, val_frac=0.2, seed=seed)

    # ---- chapter 4: late fusion (pooled + concat -> MLP) ----
    feats = pooled_features(tokens)
    from torch.utils.data import TensorDataset
    late_tr = DataLoader(TensorDataset(feats[tr], labels[tr]), batch_size=64, shuffle=True)
    late_va = DataLoader(TensorDataset(feats[va], labels[va]), batch_size=64)
    print("=== Chapter 4 — late fusion (MLPFusion) ===")
    _trainer(max_epochs).fit(
        LitMLPFusion(input_dim=feats.shape[1]), late_tr, late_va
    )

    # ---- early fusion (token sequences -> joint Transformer) ----
    sub = lambda idx: {m: t[idx] for m, t in tokens.items()}
    early_tr = DataLoader(_TokenDataset(sub(tr), labels[tr]), batch_size=64, shuffle=True)
    early_va = DataLoader(_TokenDataset(sub(va), labels[va]), batch_size=64)
    print("=== Early fusion (JointFusionTransformer) ===")
    _trainer(max_epochs).fit(
        LitEarlyFusion(MODALITY_DIMS), early_tr, early_va
    )


if __name__ == "__main__":
    train_both()
