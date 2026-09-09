"""§4.7 follow-up — quantify "calibrate the threshold" vs "add the labels".

Two critiques of the original §4.7 table:
  * p@top10% across different base rates (49% vs 20%) is confounded — replace it
    with lift@10% (= p@top10% / base rate), which is comparable.
  * FP@0.5 conflates ranking quality with a badly-placed threshold — report the
    BEST-threshold F1 too, i.e. the *upper bound* of what pure calibration can
    recover (oracle threshold, so slightly optimistic by construction).

Reads the cached per-clip scores from exp7 (add_normals/*.npz: src, label,
s_def = default prompts, s_exp = +5 Kinetics safe prompts).

Run anywhere with the cache present: python scripts/exp7_threshold_analysis.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

OUT = os.path.expanduser("~/documents/SentinelAI/data/xd-violence/add_normals")


def metrics(y, s):
    auc = roc_auc_score(y, s)
    order = np.argsort(-s)
    k = max(1, int(0.10 * len(y)))
    p10 = y[order[:k]].mean()
    lift = p10 / y.mean()
    f05 = f1_score(y, (s >= 0.5).astype(int), zero_division=0)
    # oracle best threshold — the ceiling pure calibration could reach
    ths = np.linspace(0.30, 0.99, 140)
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in ths]
    bi = int(np.argmax(f1s)); bt = ths[bi]
    pred = (s >= bt).astype(int)
    return dict(auc=auc, p10=p10, lift=lift, f05=f05, bt=bt, fbest=f1s[bi],
                pbest=precision_score(y, pred, zero_division=0),
                rbest=recall_score(y, pred, zero_division=0))


def main():
    rows = [np.load(f, allow_pickle=True) for f in glob.glob(f"{OUT}/*.npz")]
    src = np.array([str(z["src"]) for z in rows])
    y = np.array([int(z["label"]) for z in rows])
    sd = np.array([float(z["s_def"]) for z in rows])
    se = np.array([float(z["s_exp"]) for z in rows])
    xd = src == "xd"
    settings = [
        ("基线: XD only, 默认prompt", y[xd], sd[xd]),
        ("(B) 加样本, 默认prompt", y, sd),
        ("(A) 加label: XD only", y[xd], se[xd]),
        ("(A+B) 加样本+label", y, se),
    ]
    print(f"{'设置':<26}{'基率':>5}{'AUC':>7}{'p@10%':>7}{'lift':>6}{'F1@0.5':>8}"
          f"{'最优阈':>7}{'F1@最优':>8}{'P':>6}{'R':>6}")
    print("-" * 86)
    for name, yy, ss in settings:
        m = metrics(yy, ss)
        print(f"{name:<26}{yy.mean():>5.0%}{m['auc']:>7.3f}{m['p10']:>7.3f}{m['lift']:>6.2f}"
              f"{m['f05']:>8.3f}{m['bt']:>7.2f}{m['fbest']:>8.3f}{m['pbest']:>6.3f}{m['rbest']:>6.3f}")
    print("\n注：F1@最优 用的是在评测集上选的 oracle 阈值 —— 它是「纯校准」能达到的上限，实际部署")
    print("要在独立验证集上定阈值，只会更低。lift = p@top10% / 基率（跨基率可比）。")


if __name__ == "__main__":
    main()
