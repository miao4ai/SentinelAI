"""Experiment 3 — real 3-modality fusion on XD-Violence (I3D + AST + text/ASR).

Adds the TEXT modality (from `extract_text_features.py`) to the visual+audio
comparison, on clips that have ALL THREE modalities. Same positions as exp2:
single-modality baselines, ① early (joint transformer), ② coordinated (CLIP),
③ feature-concat, ④ decision-level (GBDT stacking over the 3 modality probs),
⑤ late-avg.

Prereq: run `scripts/extract_text_features.py` first to build data/.../text_features.
Run on the GPU box: python scripts/exp3_three_modal.py
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
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
AUDIO = f"{DATA}/audio_ast"
TEXT = f"{DATA}/text_features"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
V_TOK, A_TOK = 32, 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIMS = {"visual": 2048, "audio": 768, "text": 768}


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


def load_audio_seq():
    acc = defaultdict(list)
    for p in glob.glob(f"{AUDIO}/**/*.parquet", recursive=True):
        tbl = pq.read_table(p, columns=["video_id", "feature_vector"]).to_pandas()
        for vid, fv in zip(tbl["video_id"], tbl["feature_vector"]):
            acc[base(vid)].append(np.mean([np.asarray(x, np.float32) for x in fv], 0))
    return {k: np.stack(v).astype(np.float32) for k, v in acc.items()}


def load_text():
    """{key: 768-d transcript embedding} from cached npz files."""
    out = {}
    for f in glob.glob(f"{TEXT}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out


def score(y, proba):
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


def torch_probas(model, tr, ytr, te, epochs=200, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y = torch.tensor(ytr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    tr = {m: torch.tensor(v, device=DEVICE) for m, v in tr.items()}
    te = {m: torch.tensor(v, device=DEVICE) for m, v in te.items()}
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


def main():
    print(f"device={DEVICE}; loading features...")
    vis, aud, txt = load_i3d_seq(), load_audio_seq(), load_text()
    keys = [k for k in vis if k in aud and k in txt]
    if not keys:
        print("no clips with all 3 modalities — run extract_text_features.py first.")
        return
    y = np.array([is_violent(k) for k in keys])
    groups = np.array([k.split("__")[0] for k in keys])   # source movie
    speech = np.mean([bool(np.any(txt[k])) for k in keys])
    print(f"3-modal clips {len(keys)}, {len(set(groups))} movies "
          f"({y.mean():.0%} violent, {speech:.0%} have speech); "
          f"5-fold GroupKFold (movie-disjoint)\n")

    Vp = np.stack([vis[k].mean(0) for k in keys])
    Ap = np.stack([aud[k].mean(0) for k in keys])
    Tp = np.stack([txt[k] for k in keys])
    Vs = np.stack([resample(vis[k], V_TOK) for k in keys])
    As = np.stack([resample(aud[k], A_TOK) for k in keys])
    Ts = Tp[:, None, :]                       # text = one token
    allp = np.concatenate([Vp, Ap, Tp], 1)

    per_model: dict[str, list[dict]] = defaultdict(list)
    for tr, te in GroupKFold(n_splits=5).split(np.arange(len(keys)), y, groups):
        ytr, yte = y[tr], y[te]
        pv = sk_probas(Vp[tr], ytr, Vp[te])
        pa = sk_probas(Ap[tr], ytr, Ap[te])
        pt = sk_probas(Tp[tr], ytr, Tp[te])
        p3 = sk_probas(allp[tr], ytr, allp[te])
        p5 = (pv + pa + pt) / 3
        # ④ decision-level: a GBDT meta-learner over the THREE modality probabilities,
        # trained on out-of-fold train probs. With 3 inputs incl. a weak text modality,
        # the tree can actually learn to downweight text (unlike the 2-modal case).
        gtr = groups[tr]
        meta_tr = np.c_[oof_probs(Vp[tr], ytr, gtr), oof_probs(Ap[tr], ytr, gtr),
                        oof_probs(Tp[tr], ytr, gtr)]
        meta = GradientBoostingClassifier(random_state=0).fit(meta_tr, ytr)
        p4 = meta.predict_proba(np.c_[pv, pa, pt])[:, 1]
        torch.manual_seed(0)
        p1 = torch_probas(JointFusionTransformer(DIMS, d_model=128, n_layers=2, n_categories=1),
                          {"visual": Vs[tr], "audio": As[tr], "text": Ts[tr]}, ytr,
                          {"visual": Vs[te], "audio": As[te], "text": Ts[te]})
        torch.manual_seed(0)
        p2 = torch_probas(CoordinatedFusion(DIMS, d_model=128, n_categories=1),
                          {"visual": Vp[tr], "audio": Ap[tr], "text": Tp[tr]}, ytr,
                          {"visual": Vp[te], "audio": Ap[te], "text": Tp[te]})
        for name, pr in [("visual only", pv), ("audio only", pa), ("text only", pt),
                         ("① early (joint transf.)", p1), ("② coordinated (CLIP)", p2),
                         ("③ feature-concat", p3), ("④ decision (GBDT stack)", p4),
                         ("⑤ late-fusion (avg)", p5)]:
            per_model[name].append(score(yte, pr))

    print(f"{'model':<26}{'F1 (mean±std)':>16}{'AUC (mean±std)':>16}")
    print("-" * 58)
    for name, folds in per_model.items():
        f1 = np.array([m["F1"] for m in folds]); auc = np.array([m["AUC"] for m in folds])
        print(f"{name:<26}{f'{f1.mean():.3f}±{f1.std():.3f}':>16}{f'{auc.mean():.3f}±{auc.std():.3f}':>16}")


if __name__ == "__main__":
    main()
