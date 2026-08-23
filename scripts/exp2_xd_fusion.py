"""Experiment 2 — real 2-modality fusion on XD-Violence (I3D visual + AST audio).

Pairs the pre-extracted I3D RGB features (visual) with AST audio features by
video_id, pools each to a video-level vector, and compares single-modality
baselines against fusion. Answers: does audio+visual fusion beat one modality,
and which fusion (feature-concat ③ vs late-vote ⑤) wins — on REAL data.

Run on the GPU box (has the data): python scripts/exp2_xd_fusion.py
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
AUDIO = f"{DATA}/audio_ast"
TRAIN_DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954"]


def base(name: str) -> str:
    """Strip extension so i3d '<clip>.npy' and audio '<clip>.mp4' share a key."""
    return name.replace(".npy", "").replace(".mp4", "")


def is_violent(name: str) -> int:
    """XD-Violence: '_label_A' is normal (0); B1/B2/B4/B5/B6/G are violent (1)."""
    return 0 if "_label_A" in name else 1


def load_i3d(dirs: list[str]) -> dict[str, np.ndarray]:
    """Video-level visual feature per clip: mean over crops, then mean over time -> (2048,)."""
    feats: dict[str, np.ndarray] = {}
    for d in dirs:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            a = np.load(f)                       # (T, 5, 2048)
            if a.ndim == 3:
                a = a.mean(axis=1)               # crop-mean -> (T, 2048)
            feats[base(os.path.basename(f))] = a.mean(axis=0).astype(np.float32)
    return feats


def load_audio(split: str) -> dict[str, np.ndarray]:
    """Video-level audio feature: pool AST snippet features (mean) per video_id."""
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for p in glob.glob(f"{AUDIO}/{split}/**/*.parquet", recursive=True):
        tbl = pq.read_table(p, columns=["video_id", "feature_vector"]).to_pandas()
        for vid, fv in zip(tbl["video_id"], tbl["feature_vector"]):
            acc[base(vid)].append(np.asarray(fv, dtype=np.float32).ravel())
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def pair(vis: dict, aud: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep clips present in BOTH modalities -> (X_visual, X_audio, y)."""
    xv, xa, y = [], [], []
    for k in vis:
        if k in aud:
            xv.append(vis[k]); xa.append(aud[k]); y.append(is_violent(k))
    return np.array(xv), np.array(xa), np.array(y)


def _probas(Xtr, ytr, Xte) -> np.ndarray:
    """Train a standardized logistic-regression classifier, return test P(violent)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def score(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "P": precision_score(y_true, pred, zero_division=0),
        "R": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AUC": roc_auc_score(y_true, proba),
    }


def main() -> None:
    print("loading features...")
    vtr, atr = load_i3d(TRAIN_DIRS), load_audio("train")
    vte, ate = load_i3d(["test_videos"]), load_audio("test")
    Xv_tr, Xa_tr, y_tr = pair(vtr, atr)
    Xv_te, Xa_te, y_te = pair(vte, ate)
    print(f"paired train: {len(y_tr)} clips ({y_tr.mean():.0%} violent) | "
          f"test: {len(y_te)} clips ({y_te.mean():.0%} violent)")
    print(f"dims: visual={Xv_tr.shape[1]}  audio={Xa_tr.shape[1]}\n")

    # per-modality probabilities (also reused for late fusion)
    pv = _probas(Xv_tr, y_tr, Xv_te)
    pa = _probas(Xa_tr, y_tr, Xa_te)
    # ③ feature-level fusion: concat then one classifier
    pc = _probas(np.concatenate([Xv_tr, Xa_tr], 1), y_tr, np.concatenate([Xv_te, Xa_te], 1))
    # ⑤ late fusion: average the two per-modality probabilities
    pl = (pv + pa) / 2

    rows = [
        ("visual only (I3D)", score(y_te, pv)),
        ("audio only (AST)", score(y_te, pa)),
        ("③ feature-concat fusion", score(y_te, pc)),
        ("⑤ late-fusion (avg)", score(y_te, pl)),
    ]
    print(f"{'model':<26}{'P':>7}{'R':>7}{'F1':>7}{'AUC':>7}")
    print("-" * 54)
    for name, m in rows:
        print(f"{name:<26}{m['P']:>7.3f}{m['R']:>7.3f}{m['F1']:>7.3f}{m['AUC']:>7.3f}")


if __name__ == "__main__":
    main()
