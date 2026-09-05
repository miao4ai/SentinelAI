"""Cross-dataset: mix XD-Violence + UCF-Crime (shared I3D/AST/XLM-R spaces).

XD (violent vs normal) and UCF (anomaly vs normal) share the same feature spaces —
I3D visual (2048), AST audio (768), XLM-R text (768) — so we can pool them under
one binary label **unsafe (violence/anomaly) = 1 vs normal = 0**:

  A. mixed training  — union of both, grouped 80/20 split, one model; per-source F1.
  B. domain transfer — train on ALL of one, test on ALL of the other (both ways).

Modalities: visual (I3D), audio (AST), text (XLM-R); fusions ⑤ late-avg, ② coordinated.
Run on the GPU box: python scripts/exp_cross_dataset.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

from sentinelai.coordinated_fusion import CoordinatedFusion

HOME = os.path.expanduser("~/documents/SentinelAI/data")
XD = f"{HOME}/xd-violence"
UCF = f"{HOME}/ucf-crime"
XD_DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def load_npz(folder):
    out = {}
    for f in glob.glob(f"{folder}/*.npz"):
        z = np.load(f, allow_pickle=True); out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out


def pooled_i3d_xd():
    cache = f"{XD}/i3d_pooled.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); return {k: z[k] for k in z.files}
    out = {}
    for d in XD_DIRS:
        for f in glob.glob(f"{XD}/data/i3d_rgb/{d}/*.npy"):
            a = np.load(f); a = a.mean(1) if a.ndim == 3 else a
            out[base(os.path.basename(f))] = a.mean(0).astype(np.float32)
    np.savez(cache, **out); return out


def pooled_i3d_ucf():
    out = {}
    for folder in [f"{UCF}/UCF_Train_ten_crop_i3d_pooled.npz", f"{UCF}/UCF_Test_ten_crop_i3d_pooled.npz"]:
        if os.path.exists(folder):
            z = np.load(folder, allow_pickle=True); out.update({k: z[k].astype(np.float32) for k in z.files})
    return out


def build(vis, aud, txt, label_fn, group_fn):
    """Rows with all three modalities -> (V, A, T, y, groups, keys)."""
    ks = [k for k in vis if k in aud and k in txt]
    V = np.stack([vis[k] for k in ks]); A = np.stack([aud[k] for k in ks])
    T = np.stack([txt[k] for k in ks]); y = np.array([label_fn(k) for k in ks])
    g = np.array([group_fn(k) for k in ks])
    return V, A, T, y, g, ks


def probs_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def probs_coord(tr, ytr, te):
    torch.manual_seed(0)
    m = CoordinatedFusion({"visual": 2048, "audio": 768, "text": 768}, d_model=128, n_categories=1).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
    y = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)[:, None]
    T = {k: torch.tensor(v, device=DEVICE) for k, v in tr.items()}
    E = {k: torch.tensor(v, device=DEVICE) for k, v in te.items()}
    for _ in range(300):
        m.train(); opt.zero_grad(); o = m(T); lo = o[0] if isinstance(o, tuple) else o
        lf(lo, y).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o = m(E); lo = o[0] if isinstance(o, tuple) else o
        return torch.sigmoid(lo).squeeze(1).cpu().numpy()


def scores(y, p):
    return f1_score(y, (p >= 0.5).astype(int), zero_division=0), roc_auc_score(y, p)


def run(Vtr, Atr, Ttr, ytr, Vte, Ate, Tte, yte, tag, src_te=None):
    pv = probs_lr(Vtr, ytr, Vte); pa = probs_lr(Atr, ytr, Ate); pt = probs_lr(Ttr, ytr, Tte)
    p5 = (pv + pa + pt) / 3
    p2 = probs_coord({"visual": Vtr, "audio": Atr, "text": Ttr}, ytr,
                     {"visual": Vte, "audio": Ate, "text": Tte})
    print(f"\n=== {tag} ===  (test n={len(yte)}, {yte.mean():.0%} unsafe)")
    for name, p in [("visual (I3D)", pv), ("audio (AST)", pa), ("text (XLM-R)", pt),
                    ("⑤ late-avg", p5), ("② coordinated", p2)]:
        f1, auc = scores(yte, p)
        line = f"  {name:<16} F1={f1:.3f}  AUC={auc:.3f}"
        if src_te is not None:                       # per-source AUC on the mixed test set
            for s in ("xd", "ucf"):
                m = src_te == s
                if m.sum() and len(set(yte[m])) == 2:
                    line += f"  [{s} AUC={roc_auc_score(yte[m], p[m]):.3f}]"
        print(line)


def main():
    print("loading features...")
    xv, xa, xt = pooled_i3d_xd(), load_npz(f"{XD}/audio_full"), load_npz(f"{XD}/text_features")
    uv, ua, ut = pooled_i3d_ucf(), load_npz(f"{UCF}/audio"), load_npz(f"{UCF}/text")
    Vx, Ax, Tx, yx, gx, kx = build(xv, xa, xt, lambda k: 0 if "_label_A" in k else 1, lambda k: "XD_" + k.split("__")[0])
    Vu, Au, Tu, yu, gu, ku = build(uv, ua, ut, lambda k: 0 if "Normal" in k else 1, lambda k: "UCF_" + k)
    print(f"XD {len(yx)} ({yx.mean():.0%} unsafe) | UCF {len(yu)} ({yu.mean():.0%} unsafe)")

    # B. domain transfer — train all of one, test all of the other
    run(Vx, Ax, Tx, yx, Vu, Au, Tu, yu, "B1: train XD -> test UCF")
    run(Vu, Au, Tu, yu, Vx, Ax, Tx, yx, "B2: train UCF -> test XD")

    # A. mixed training — union, movie/video-grouped 80/20 split, per-source AUC
    V = np.concatenate([Vx, Vu]); A = np.concatenate([Ax, Au]); T = np.concatenate([Tx, Tu])
    y = np.concatenate([yx, yu]); g = np.concatenate([gx, gu])
    src = np.array(["xd"] * len(yx) + ["ucf"] * len(yu))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(V, y, g))
    run(V[tr], A[tr], T[tr], y[tr], V[te], A[te], T[te], y[te], "A: mixed XD+UCF (grouped 80/20)", src[te])


if __name__ == "__main__":
    main()
