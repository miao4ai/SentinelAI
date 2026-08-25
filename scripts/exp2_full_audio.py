"""Experiment 2 (full-audio) — fusion positions on the EXPANDED paired set.

Also adds ④ decision-level fusion (a GBDT meta-learner stacked on the per-modality
probabilities, trained out-of-fold) — the learned counterpart to ⑤'s fixed average.

Same comparison as exp2_all_positions.py, but audio now comes from
`data/xd-violence/audio_full/` — AST features we extracted ourselves for *every*
I3D clip (see scripts/extract_audio_features.py), lifting the paired visual+audio
set from 788 to ~4500 clips. The question: does more paired data let the learned
fusions (①②③) finally beat the simple ⑤ late average?

Difference from exp2: audio_full stores one clip-level *mean* AST vector (768-d),
not per-snippet tokens. So ② ③ ⑤ (pooled) are unchanged, and ① early gets the
audio as a single summary token alongside the 32 visual temporal tokens. Visual
I3D still provides real temporal tokens.

Run on the GPU box: python scripts/exp2_full_audio.py
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from sentinelai.coordinated_fusion import CoordinatedFusion
from sentinelai.early_fusion import JointFusionTransformer

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
AUDIO_FULL = f"{DATA}/audio_full"
AUDIO_SEQ = f"{DATA}/audio_seq"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
V_TOK, A_TOK = 32, 16          # fixed token counts for the early-fusion transformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def is_violent(n): return 0 if "_label_A" in n else 1


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample/pad a (T, d) sequence to exactly (n, d)."""
    if len(x) == 0:
        return np.zeros((n, x.shape[1]), dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return x[idx].astype(np.float32)


def load_i3d_seq() -> dict[str, np.ndarray]:
    """Per clip: crop-mean I3D -> (T, 2048) temporal token sequence."""
    out = {}
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            a = np.load(f)
            if a.ndim == 3:
                a = a.mean(axis=1)                 # (T, 2048)
            out[base(os.path.basename(f))] = a.astype(np.float32)
    return out


def load_audio_pooled() -> dict[str, np.ndarray]:
    """Per clip: our self-extracted clip-level mean AST embedding -> (768,)."""
    out = {}
    for f in glob.glob(f"{AUDIO_FULL}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out


def load_audio_seq() -> dict[str, np.ndarray]:
    """Per clip: the full AST token sequence -> (num_windows, 768). Used when --seq
    is set so early fusion (①) attends over real audio tokens, not one mean vector.
    The pooled positions (②③④⑤) still use its mean, identical to load_audio_pooled."""
    out = {}
    for f in glob.glob(f"{AUDIO_SEQ}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["key"])] = z["sequence"].astype(np.float32)
    return out


def score(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {"P": precision_score(y, pred, zero_division=0),
            "R": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, proba)}


def sk_probas(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def oof_probs(X, y, g):
    """Out-of-fold LR probabilities on the training set (inner movie-grouped 3-fold),
    so the ④ meta-learner never sees a modality's prediction on data it trained on."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = list(GroupKFold(n_splits=3).split(X, y, g))
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def torch_probas(model, tr_inputs, ytr, te_inputs, epochs=200, lr=1e-3):
    """Full-batch train a torch fusion model (binary BCE), return test probabilities."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y = torch.tensor(ytr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    tr = {m: torch.tensor(v, device=DEVICE) for m, v in tr_inputs.items()}
    te = {m: torch.tensor(v, device=DEVICE) for m, v in te_inputs.items()}
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(tr)
        logits = out[0] if isinstance(out, tuple) else out
        loss_fn(logits, y).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(te)
        logits = out[0] if isinstance(out, tuple) else out
        return torch.sigmoid(logits).squeeze(1).cpu().numpy()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    # --seq: give ① early fusion the full AST token sequence (from audio_seq/)
    # instead of a single mean token. ②③④⑤ still use the mean either way.
    ap.add_argument("--seq", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE}; loading features (audio={'seq tokens' if args.seq else 'mean'})...")
    vis = load_i3d_seq()
    aud = load_audio_seq() if args.seq else load_audio_pooled()
    keys = [k for k in vis if k in aud]
    y = np.array([is_violent(k) for k in keys])
    # Group by source movie (key before "__#") so no movie's clips straddle a
    # fold — avoids the movie-level leak that inflates a random per-clip split.
    groups = np.array([k.split("__")[0] for k in keys])
    print(f"paired {len(keys)} clips, {len(set(groups))} movies ({y.mean():.0%} violent); "
          f"5-fold GroupKFold (movie-disjoint)\n")

    # features computed once; folds index into them. With --seq, aud[k] is a
    # (num_windows, 768) sequence, so Ap is its mean and As resamples it to A_TOK
    # real tokens; without, aud[k] is already the (768,) mean and As is one token.
    Vp = np.stack([vis[k].mean(0) for k in keys])                  # (N, 2048)
    Ap = np.stack([aud[k].mean(0) if args.seq else aud[k] for k in keys])   # (N, 768)
    Vs = np.stack([resample(vis[k], V_TOK) for k in keys])         # (N, 32, 2048)
    if args.seq:
        As = np.stack([resample(aud[k], A_TOK) for k in keys])    # (N, 16, 768) real tokens
    else:
        As = Ap[:, None, :]                                       # (N, 1, 768) single audio token
    VpAp = np.concatenate([Vp, Ap], 1)

    per_model: dict[str, list[dict]] = defaultdict(list)
    for tr, te in GroupKFold(n_splits=5).split(np.arange(len(keys)), y, groups):
        ytr, yte = y[tr], y[te]
        pv = sk_probas(Vp[tr], ytr, Vp[te])
        pa = sk_probas(Ap[tr], ytr, Ap[te])
        p3 = sk_probas(VpAp[tr], ytr, VpAp[te])
        p5 = (pv + pa) / 2
        # ④ decision-level: a GBDT meta-learner over the two modality probabilities,
        # trained on out-of-fold train probs (avoids the meta-learner seeing leaked
        # in-sample predictions), then applied to the test-fold probs [pv, pa].
        gtr = groups[tr]
        meta_tr = np.c_[oof_probs(Vp[tr], ytr, gtr), oof_probs(Ap[tr], ytr, gtr)]
        meta = GradientBoostingClassifier(random_state=0).fit(meta_tr, ytr)
        p4 = meta.predict_proba(np.c_[pv, pa])[:, 1]
        torch.manual_seed(0)
        p1 = torch_probas(
            JointFusionTransformer({"visual": 2048, "audio": 768}, d_model=128, n_layers=2, n_categories=1),
            {"visual": Vs[tr], "audio": As[tr]}, ytr, {"visual": Vs[te], "audio": As[te]})
        torch.manual_seed(0)
        p2 = torch_probas(
            CoordinatedFusion({"visual": 2048, "audio": 768}, d_model=128, n_categories=1),
            {"visual": Vp[tr], "audio": Ap[tr]}, ytr, {"visual": Vp[te], "audio": Ap[te]})
        for name, pr in [("visual only (I3D)", pv), ("audio only (AST)", pa),
                         ("① early (joint transf.)", p1), ("② coordinated (CLIP)", p2),
                         ("③ feature-concat", p3), ("④ decision (GBDT stack)", p4),
                         ("⑤ late-fusion (avg)", p5)]:
            per_model[name].append(score(yte, pr))

    # mean ± std across the 5 folds
    print(f"{'model':<26}{'F1 (mean±std)':>16}{'AUC (mean±std)':>16}")
    print("-" * 58)
    for name, folds in per_model.items():
        f1 = np.array([m["F1"] for m in folds]); auc = np.array([m["AUC"] for m in folds])
        print(f"{name:<26}{f'{f1.mean():.3f}±{f1.std():.3f}':>16}{f'{auc.mean():.3f}±{auc.std():.3f}':>16}")


if __name__ == "__main__":
    main()
