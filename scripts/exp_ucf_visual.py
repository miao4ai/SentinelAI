"""UCF-Crime visual-only baseline (I3D) — the only modality UCF has.

UCF-Crime on GCS is **I3D ten-crop features only** (no audio / text / raw video),
so the multimodal fusion positions don't apply here — this is the visual-only row
of the cross-dataset results matrix. Task: normal vs anomaly (filename `Normal_*`
= normal, everything else = anomaly), using UCF's own train/test split.

Each clip: (T, 10, 2048) ten-crop I3D -> mean over crops -> mean over time ->
(2048) -> LogisticRegression. Report F1 / P / R / AUC on the test split.

Run on the GPU box:  python scripts/exp_ucf_visual.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = os.path.expanduser("~/documents/SentinelAI/data/ucf-crime")
TRAIN = f"{DATA}/UCF_Train_ten_crop_i3d"
TEST = f"{DATA}/UCF_Test_ten_crop_i3d"


def is_anomaly(name: str) -> int:
    return 0 if "Normal" in os.path.basename(name) else 1


def load(folder: str):
    """Each .npy -> one pooled (2048,) vector + its normal/anomaly label."""
    X, y = [], []
    for f in sorted(glob.glob(f"{folder}/*.npy")):
        a = np.load(f)
        if a.ndim == 3:               # (T, 10 crops, 2048) -> mean over crops
            a = a.mean(axis=1)
        X.append(a.mean(axis=0).astype(np.float32))   # pool over time
        y.append(is_anomaly(f))
    return np.stack(X), np.array(y)


def main() -> None:
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    print(f"UCF-Crime I3D — train {len(ytr)} ({ytr.mean():.0%} anomaly), "
          f"test {len(yte)} ({yte.mean():.0%} anomaly); feature dim {Xtr.shape[1]}")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    pred = (p >= 0.5).astype(int)

    print(f"\n{'model':<26}{'F1':>8}{'P':>8}{'R':>8}{'AUC':>8}")
    print("-" * 58)
    print(f"{'visual only (I3D)':<26}"
          f"{f1_score(yte, pred, zero_division=0):>8.3f}"
          f"{precision_score(yte, pred, zero_division=0):>8.3f}"
          f"{recall_score(yte, pred, zero_division=0):>8.3f}"
          f"{roc_auc_score(yte, p):>8.3f}")


if __name__ == "__main__":
    main()
