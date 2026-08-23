"""Inspect false positives of the visual-only (I3D) violence classifier.

Out-of-fold (movie-grouped CV) predictions for every clip, then list the NORMAL
clips (label_A) the visual model most confidently mislabels as violent. The clip
id encodes movie + timestamp, so you can tell which scene fooled it.

Run on the GPU box: python scripts/inspect_visual_fp.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
AUDIO = f"{DATA}/audio_ast"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def is_violent(n): return 0 if "_label_A" in n else 1


def main() -> None:
    import pyarrow.parquet as pq

    # only the clips also used in the fusion experiment (have audio too)
    aud = set()
    for p in glob.glob(f"{AUDIO}/**/*.parquet", recursive=True):
        for v in pq.read_table(p, columns=["video_id"]).to_pandas()["video_id"]:
            aud.add(base(v))

    keys, X = [], []
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            k = base(os.path.basename(f))
            if k in aud:
                a = np.load(f)
                if a.ndim == 3:
                    a = a.mean(axis=1)
                keys.append(k); X.append(a.mean(axis=0).astype(np.float32))
    X = np.stack(X)
    y = np.array([is_violent(k) for k in keys])
    groups = np.array([k.split("__")[0] for k in keys])

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(5), groups=groups,
                              method="predict_proba")[:, 1]

    fps = sorted(((keys[i], proba[i]) for i in range(len(keys)) if y[i] == 0 and proba[i] >= 0.5),
                 key=lambda t: -t[1])
    n_normal = int((y == 0).sum())
    print(f"visual-only false positives: {len(fps)} / {n_normal} normal clips "
          f"({len(fps) / n_normal:.0%})\n")
    print("most-confident false positives (normal clip predicted violent):")
    for k, p in fps[:10]:
        print(f"  p(violent)={p:.3f}   {k}")


if __name__ == "__main__":
    main()
