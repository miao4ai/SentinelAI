"""UCF-Crime multimodal fusion (I3D visual + AST audio + XLM-R text).

Extends the visual-only UCF baseline (exp_ucf_visual.py) once audio/text are
extracted from the raw videos (extract_ucf_multimodal.py). Task: normal vs anomaly,
on UCF's official train/test split. Single modalities + fusion positions ②③⑤,
so UCF joins the multimodal matrix.

Run on the GPU box: python scripts/exp_ucf_multimodal.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from sentinelai.coordinated_fusion import CoordinatedFusion

DATA = os.path.expanduser("~/documents/SentinelAI/data/ucf-crime")
TRAIN_I3D = f"{DATA}/UCF_Train_ten_crop_i3d"
TEST_I3D = f"{DATA}/UCF_Test_ten_crop_i3d"
AUDIO, TEXT = f"{DATA}/audio", f"{DATA}/text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_anomaly(name): return 0 if "Normal" in name else 1
def i3d_key(f): return os.path.basename(f).replace("_i3d.npy", "").replace(".npy", "")


def load_i3d(folder):
    out = {}
    for f in glob.glob(f"{folder}/*.npy"):
        a = np.load(f)
        if a.ndim == 3:
            a = a.mean(axis=1)
        out[i3d_key(f)] = a.mean(axis=0).astype(np.float32)
    return out


def load_npz(folder):
    out = {}
    for f in glob.glob(f"{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out


def sk_probas(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def torch_coord(tr, ytr, te):
    dims = {"visual": 2048, "audio": 768, "text": 768}
    torch.manual_seed(0)
    m = CoordinatedFusion(dims, d_model=128, n_categories=1).to(DEVICE)
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


def report(y, probs):
    print(f"\n{'model':<24}{'F1':>8}{'P':>8}{'R':>8}{'AUC':>8}")
    print("-" * 56)
    for name, p in probs.items():
        pred = (p >= 0.5).astype(int)
        print(f"{name:<24}{f1_score(y,pred,zero_division=0):>8.3f}{precision_score(y,pred,zero_division=0):>8.3f}"
              f"{recall_score(y,pred,zero_division=0):>8.3f}{roc_auc_score(y,p):>8.3f}")


def main():
    vtr, vte = load_i3d(TRAIN_I3D), load_i3d(TEST_I3D)
    aud, txt = load_npz(AUDIO), load_npz(TEXT)
    has_text = len(txt) > 0
    speech = np.mean([bool(np.any(v)) for v in txt.values()]) if has_text else 0.0

    def prep(vis):
        ks = [k for k in vis if k in aud]                 # require audio
        V = np.stack([vis[k] for k in ks]); A = np.stack([aud[k] for k in ks])
        T = np.stack([txt.get(k, np.zeros(768, np.float32)) for k in ks])
        y = np.array([is_anomaly(k) for k in ks])
        return ks, V, A, T, y

    ktr, Vtr, Atr, Ttr, ytr = prep(vtr)
    kte, Vte, Ate, Tte, yte = prep(vte)
    print(f"UCF 多模态: train {len(ktr)} ({ytr.mean():.0%} 异常), test {len(kte)} ({yte.mean():.0%} 异常); "
          f"文本可用={has_text} (有语音 {speech:.0%})")

    pv = sk_probas(Vtr, ytr, Vte)
    pa = sk_probas(Atr, ytr, Ate)
    probs = {"visual only (I3D)": pv, "audio only (AST)": pa}
    late_parts = [pv, pa]
    concat_tr, concat_te = [Vtr, Atr], [Vte, Ate]
    if has_text:
        pt = sk_probas(Ttr, ytr, Tte); probs["text only (XLM-R)"] = pt
        late_parts.append(pt); concat_tr.append(Ttr); concat_te.append(Tte)
    probs["③ feature-concat"] = sk_probas(np.concatenate(concat_tr, 1), ytr, np.concatenate(concat_te, 1))
    probs["⑤ late-fusion (avg)"] = np.mean(late_parts, axis=0)
    probs["② coordinated"] = torch_coord(
        {"visual": Vtr, "audio": Atr, "text": Ttr}, ytr, {"visual": Vte, "audio": Ate, "text": Tte})
    report(yte, probs)


if __name__ == "__main__":
    main()
