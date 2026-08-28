"""Experiment 6 — real cross-attention deep fusion on XD-Violence (V2 ch.6).

The V2 thesis (ROADMAP 6.1/6.2): a single frame can't tell "cutting vegetables"
from "stabbing" — you need motion across frames, and you need the audio/text cue
to point at the *right* frames. So:

    Query  (Q) = audio (and text) features   — "what am I looking for?"
    Key/Value  = I3D per-frame temporal seq   — "where/when in the video is it?"

I3D is a 3D CNN, so its (T, 2048) sequence already encodes short-term motion
(the ch.6.1 temporal features; VideoMAE/TimeSformer would be a drop-in swap that
needs re-extraction from raw frames). The cross-attention layer (ch.6.2,
`CrossAttentionFusion`) lets the guide attend over those frames; we train it with
the existing PyTorch-Lightning module (ch.6.3, `LitCrossAttention`).

Evaluated with the SAME movie-grouped 5-fold CV as exp2/exp3, so its F1/AUC drop
straight into the §7.1 (2-modal) / §8 (3-modal) comparison tables.

    --modal 2  video K/V + audio Q          (4502 clips, cf. §7.1)
    --modal 3  video K/V + [audio,text] Q    (788 clips,  cf. §8)

Run on the GPU box: python scripts/exp6_cross_attention.py --modal 2
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl

from sentinelai.train.lit_module import LitCrossAttention

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
AUDIO_FULL = f"{DATA}/audio_full"
TEXT = f"{DATA}/text_features"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
V_TOK = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def is_violent(n): return 0 if "_label_A" in n else 1


def resample(x, n):
    if len(x) == 0:
        return np.zeros((n, x.shape[1]), dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return x[idx].astype(np.float32)


def load_i3d_seq():
    out = {}
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            a = np.load(f)
            if a.ndim == 3:
                a = a.mean(axis=1)
            out[base(os.path.basename(f))] = a.astype(np.float32)
    return out


def load_npz_dict(folder):
    out = {}
    for f in glob.glob(f"{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out


def score(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {"P": precision_score(y, pred, zero_division=0),
            "R": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, proba)}


def train_fold(Vtr, Gtr, ytr, Vte, Gte, epochs, seed):
    """Train one LitCrossAttention on a fold, return test-fold violation probs."""
    pl.seed_everything(seed, verbose=False)
    lit = LitCrossAttention(video_dim=Vtr.shape[-1], guide_dim=Gtr.shape[-1],
                            n_categories=1, d_model=128, n_heads=4, lr=1e-3)
    ds = TensorDataset(torch.tensor(Vtr), torch.tensor(Gtr),
                       torch.tensor(ytr[:, None], dtype=torch.float32))
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="auto", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0)
    trainer.fit(lit, DataLoader(ds, batch_size=128, shuffle=True))
    lit.eval()
    dev = lit.device
    with torch.no_grad():
        probs = lit.model.predict_proba(
            torch.tensor(Vte, device=dev), torch.tensor(Gte, device=dev))
    return probs[:, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal", type=int, choices=[2, 3], default=2)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    print(f"device={DEVICE}; loading features (modal={args.modal})...")
    vis = load_i3d_seq()
    aud = load_npz_dict(AUDIO_FULL)
    txt = load_npz_dict(TEXT) if args.modal == 3 else {}

    keys = [k for k in vis if k in aud and (args.modal == 2 or k in txt)]
    y = np.array([is_violent(k) for k in keys])
    groups = np.array([k.split("__")[0] for k in keys])
    print(f"{len(keys)} clips, {len(set(groups))} movies ({y.mean():.0%} violent); "
          f"video K/V = I3D {V_TOK} frames; "
          f"guide Q = {'audio' if args.modal == 2 else 'audio+text'}\n")

    # video K/V: I3D temporal frames (N, T, 2048)
    V = np.stack([resample(vis[k], V_TOK) for k in keys])
    # guide Q: audio (N, 768) as one query token, or [audio, text] as two tokens
    A = np.stack([aud[k] for k in keys])
    if args.modal == 2:
        G = A                                     # (N, 768) -> length-1 query
    else:
        T = np.stack([txt[k] for k in keys])
        G = np.stack([A, T], axis=1)              # (N, 2, 768) two query tokens

    folds = []
    for tr, te in GroupKFold(n_splits=5).split(np.arange(len(keys)), y, groups):
        p = train_fold(V[tr], G[tr], y[tr], V[te], G[te], args.epochs, seed=0)
        folds.append(score(y[te], p))

    f1 = np.array([m["F1"] for m in folds]); auc = np.array([m["AUC"] for m in folds])
    print(f"{'model':<30}{'F1 (mean±std)':>16}{'AUC (mean±std)':>16}")
    print("-" * 62)
    name = f"⑥ cross-attn ({'V+A' if args.modal == 2 else 'V+A+T'})"
    print(f"{name:<30}{f'{f1.mean():.3f}±{f1.std():.3f}':>16}{f'{auc.mean():.3f}±{auc.std():.3f}':>16}")


if __name__ == "__main__":
    main()
