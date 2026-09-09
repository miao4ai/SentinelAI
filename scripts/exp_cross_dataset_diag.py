"""Diagnostics for §4.8 — is "zero transfer" real, and does mixing actually help?

Two suspicions raised about exp_cross_dataset.py's conclusions:

  1. B's below-random transfer (visual AUC 0.41-0.46) may be an ARTIFACT: the
     StandardScaler is fit on the source domain and applied to the target, but the
     two I3D feature sets come from different extraction pipelines — a scale/offset
     mismatch alone can wreck (even invert) the decision boundary.
  2. A's mixed-training 0.971 may be a domain-conditional SHORTCUT (two classifiers
     in one), not cross-domain generalisation.

Four checks, all cheap (pooled features + LR):
  ① feature statistics        — how different are the two domains' feature scales?
  ② domain probe              — held-out AUC of "XD vs UCF?" (grouped split)
  ③ per-domain z-scored B     — standardise each domain with ITS OWN stats
                                (unsupervised, no target labels), re-run transfer
  ④ same-split ablation       — mixed vs single-domain training, evaluated on the
                                IDENTICAL per-domain test portions

Run on the GPU box: python scripts/exp_cross_dataset_diag.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from exp_cross_dataset import (XD, UCF, build, load_npz, pooled_i3d_ucf,   # noqa: E402
                               pooled_i3d_xd, probs_coord, probs_lr)


def zscore(X):
    """Standardise a domain with its OWN statistics (unsupervised — no labels used)."""
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def auc_lr(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main():
    print("loading features...")
    xv, xa, xt = pooled_i3d_xd(), load_npz(f"{XD}/audio_full"), load_npz(f"{XD}/text_features")
    uv, ua, ut = pooled_i3d_ucf(), load_npz(f"{UCF}/audio"), load_npz(f"{UCF}/text")
    Vx, Ax, Tx, yx, gx, _ = build(xv, xa, xt, lambda k: 0 if "_label_A" in k else 1, lambda k: "XD_" + k.split("__")[0])
    Vu, Au, Tu, yu, gu, _ = build(uv, ua, ut, lambda k: 0 if "Normal" in k else 1, lambda k: "UCF_" + k)
    print(f"XD {len(yx)} | UCF {len(yu)}\n")

    # ── ① feature statistics: are the two I3D pipelines even on the same scale? ──
    print("=== ① 特征统计（同为 I3D，但来自不同抽取流水线）===")
    for name, A, B in [("visual", Vx, Vu), ("audio ", Ax, Au), ("text  ", Tx, Tu)]:
        na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        # 逐维均值的错位程度，除以联合 std 做个无量纲的 shift 指标
        shift = np.abs(A.mean(0) - B.mean(0)) / (np.concatenate([A, B]).std(0) + 1e-6)
        print(f"  {name}  L2范数 XD {na.mean():7.2f}±{na.std():6.2f} | UCF {nb.mean():7.2f}±{nb.std():6.2f}"
              f" | 逐维均值错位(中位) {np.median(shift):.2f} σ")

    # ── ② domain probe: held-out "which dataset is this clip from?" ──
    print("\n=== ② 域探针（视觉特征判断 XD vs UCF，分组 80/20 留出）===")
    V = np.concatenate([Vx, Vu]); d = np.array([0] * len(yx) + [1] * len(yu))
    g = np.concatenate([gx, gu])
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(V, d, g))
    print(f"  probe AUC = {auc_lr(V[tr], d[tr], V[te], d[te]):.3f}   (≈1.0 = 域身份白送)")

    # ── ③ transfer, before/after per-domain z-scoring ──
    print("\n=== ③ 域迁移：源域 scaler（原做法） vs 按域各自标准化 ===")
    zVx, zVu, zAx, zAu = zscore(Vx), zscore(Vu), zscore(Ax), zscore(Au)
    rows = [
        ("视觉 XD→UCF", auc_lr(Vx, yx, Vu, yu), auc_lr(zVx, yx, zVu, yu)),
        ("视觉 UCF→XD", auc_lr(Vu, yu, Vx, yx), auc_lr(zVu, yu, zVx, yx)),
        ("音频 XD→UCF", auc_lr(Ax, yx, Au, yu), auc_lr(zAx, yx, zAu, yu)),
        ("音频 UCF→XD", auc_lr(Au, yu, Ax, yx), auc_lr(zAu, yu, zAx, yx)),
    ]
    print(f"  {'':<12}{'源域scaler':>12}{'按域z-score':>12}")
    for n, a, b in rows:
        print(f"  {n:<12}{a:>12.3f}{b:>12.3f}")

    # ── ④ same-split ablation: does mixing beat single-domain training? ──
    print("\n=== ④ 同 split 消融（同一测试集上：混合训练 vs 单域训练）===")
    Vall = np.concatenate([Vx, Vu]); Aall = np.concatenate([Ax, Au]); Tall = np.concatenate([Tx, Tu])
    yall = np.concatenate([yx, yu]); gall = np.concatenate([gx, gu])
    src = np.array(["xd"] * len(yx) + ["ucf"] * len(yu))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(Vall, yall, gall))
    tr_xd = tr[src[tr] == "xd"]; tr_ucf = tr[src[tr] == "ucf"]
    te_xd = te[src[te] == "xd"]; te_ucf = te[src[te] == "ucf"]
    print(f"  test: XD {len(te_xd)} / UCF {len(te_ucf)}")

    def eval_regime(name, idx):
        pv_xd = probs_lr(Vall[idx], yall[idx], Vall[te_xd])
        pv_ucf = probs_lr(Vall[idx], yall[idx], Vall[te_ucf])
        p2_xd = probs_coord({"visual": Vall[idx], "audio": Aall[idx], "text": Tall[idx]}, yall[idx],
                            {"visual": Vall[te_xd], "audio": Aall[te_xd], "text": Tall[te_xd]})
        p2_ucf = probs_coord({"visual": Vall[idx], "audio": Aall[idx], "text": Tall[idx]}, yall[idx],
                             {"visual": Vall[te_ucf], "audio": Aall[te_ucf], "text": Tall[te_ucf]})
        print(f"  {name:<16} 视觉: XD测 {roc_auc_score(yall[te_xd], pv_xd):.3f} / UCF测 {roc_auc_score(yall[te_ucf], pv_ucf):.3f}"
              f"   ②coord: XD测 {roc_auc_score(yall[te_xd], p2_xd):.3f} / UCF测 {roc_auc_score(yall[te_ucf], p2_ucf):.3f}")

    eval_regime("混合训练", tr)
    eval_regime("只用 XD 训练", tr_xd)
    eval_regime("只用 UCF 训练", tr_ucf)
    print("\n  读法：若「混合」≈「单域」各自主场成绩 → 混合只是容量，无协同；若更高 → 存在真迁移。")


if __name__ == "__main__":
    main()
