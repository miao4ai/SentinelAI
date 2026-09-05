"""End-to-end fine-tune VideoMAE on XD violence — the real attempt to beat I3D.

Frozen VideoMAE features lost to I3D (§4.4). This fine-tunes the whole ViT on the
violence task (gradients flow through the encoder), so its features become
task-aligned — the one thing frozen features and off-the-shelf checkpoints lack.
Head-to-head with frozen I3D visual-only on the SAME movie-grouped split.

    frames16/<key>.npz (16,224,224,3) --VideoMAEForVideoClassification (full FT)--> violent?

Run on the GPU box: python scripts/finetune_videomae.py [--epochs 4] [--bs 4] [--accum 4]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
I3D = f"{DATA}/data/i3d_rgb"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
MODEL = "MCG-NJU/videomae-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def is_violent(n): return 0 if "_label_A" in n else 1


class FramesDS(Dataset):
    def __init__(self, files, proc):
        self.files, self.proc = files, proc

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        frames = list(z["frames"])                       # 16 x (224,224,3) uint8
        px = self.proc(frames, return_tensors="pt")["pixel_values"][0]   # (16,3,224,224)
        return px, is_violent(str(z["key"])), str(z["key"])


def collate(batch):
    px = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch])
    keys = [b[2] for b in batch]
    return px, y, keys


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps, ys, ks = [], [], []
    for px, y, keys in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pixel_values=px.to(DEVICE)).logits
        ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        ys.append(y.numpy()); ks += keys
    return np.concatenate(ps), np.concatenate(ys), ks


def i3d_baseline(train_keys, test_keys):
    """Frozen I3D visual-only on the same split, for the head-to-head."""
    pooled = {}
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            k = base(os.path.basename(f))
            if k in train_keys or k in test_keys:
                a = np.load(f); a = a.mean(1) if a.ndim == 3 else a
                pooled[k] = a.mean(0).astype(np.float32)
    def XY(keys):
        ks = [k for k in keys if k in pooled]
        return np.stack([pooled[k] for k in ks]), np.array([is_violent(k) for k in ks])
    Xtr, ytr = XY(train_keys); Xte, yte = XY(test_keys)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return f1_score(yte, (p >= 0.5).astype(int)), roc_auc_score(yte, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    files = sorted(glob.glob(f"{FRAMES}/*.npz"))
    keys = [base(os.path.basename(f)) for f in files]
    groups = np.array([k.split("__")[0] for k in keys])
    y = np.array([is_violent(k) for k in keys])
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(files, y, groups))
    tr_files = [files[i] for i in tr]; te_files = [files[i] for i in te]
    print(f"{len(files)} clips ({y.mean():.0%} violent); train {len(tr)} / test {len(te)} (movie-grouped)")

    proc = VideoMAEImageProcessor.from_pretrained(MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)

    tr_loader = DataLoader(FramesDS(tr_files, proc), batch_size=args.bs, shuffle=True,
                           num_workers=6, collate_fn=collate, drop_last=True)
    te_loader = DataLoader(FramesDS(te_files, proc), batch_size=args.bs, shuffle=False,
                           num_workers=6, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    lossf = torch.nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train(); opt.zero_grad(); tot = 0.0
        for i, (px, yb, _) in enumerate(tr_loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=px.to(DEVICE)).logits
                loss = lossf(logits, yb.to(DEVICE)) / args.accum
            loss.backward(); tot += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                opt.step(); opt.zero_grad()
        p, yte, kte = predict(model, te_loader)
        print(f"epoch {ep}: loss={tot/max(len(tr_loader),1):.3f}  "
              f"VideoMAE-FT F1={f1_score(yte,(p>=0.5).astype(int)):.3f} AUC={roc_auc_score(yte,p):.3f}")

    p, yte, kte = predict(model, te_loader)
    f1_v, auc_v = f1_score(yte, (p >= 0.5).astype(int)), roc_auc_score(yte, p)
    f1_i, auc_i = i3d_baseline(set(base(os.path.basename(f)) for f in tr_files),
                               set(base(os.path.basename(f)) for f in te_files))
    print("\n=== visual-only, same movie-grouped test split ===")
    print(f"  frozen I3D          F1={f1_i:.3f}  AUC={auc_i:.3f}")
    print(f"  VideoMAE end2end FT F1={f1_v:.3f}  AUC={auc_v:.3f}")


if __name__ == "__main__":
    main()
