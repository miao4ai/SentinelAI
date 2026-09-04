"""Experiment 7 — does adding many NORMAL videos (Kinetics) change the results?

Real moderation is a needle-in-haystack: violence is rare, most content is normal.
This tests two "adding normal" moves on the ch.5 CLIP zero-shot violence screener,
using the 450-clip Kinetics subset (5 benign action classes: archery, bowling,
flying_kite, high_jump, marching — all clearly non-violent):

  A. add labels — put those 5 Kinetics action classes into the prompt pool as extra
     SAFE prompts, re-score the XD clips, see if AUC / precision moves.
  B. add samples — add the 450 Kinetics clips (all label=normal) to the eval set,
     see how precision / FP-rate change under the realistic base-rate shift.
  A+B. add both — do the matching safe labels rescue the added normals from FPs?

Every clip: extract 8 frames -> CLIP zero-shot -> max-frame violation prob, scored
under BOTH the default and the label-expanded prompt pool. Cached + shardable.

Run on the GPU box: python scripts/exp7_add_normals.py --n 400 [--shard i --nshards N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from exp5_clip_zeroshot import (all_i3d_keys, extract_frames, is_violent, safe,
                                sample_keys, video_repo_map, HF_REPO)

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
KIN = os.path.expanduser("~/documents/SentinelAI/data/kinetics400/extracted")
OUT = f"{DATA}/add_normals"

# A: the 5 Kinetics action classes, written as SAFE prompts to add to the pool.
KIN_SAFE = (
    "a photo of a person doing archery",
    "a photo of people bowling",
    "a photo of a person flying a kite",
    "a photo of an athlete doing a high jump",
    "a photo of people marching",
)


def clip_score(screener, frames):
    fs = screener.score_frames(frames)
    return max((f.violation_prob for f in fs), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="XD clips to sample (class-balanced)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from sentinelai.clip_screener import (ClipScreener, build_prompts,
                                          SAFE_PROMPTS, VIOLATION_PROMPTS)

    # Two screeners over the same CLIP: default pool vs label-expanded pool (adds
    # the Kinetics action classes as safe prompts). Prompt text is pre-encoded once.
    scr_def = ClipScreener()
    scr_exp = ClipScreener(prompts=build_prompts(VIOLATION_PROMPTS, SAFE_PROMPTS + KIN_SAFE))

    # Build the work list: XD sample (real labels) + all Kinetics clips (label=normal).
    xd_keys = sample_keys(all_i3d_keys(), args.n, 0)
    # Cache the HF repo file listing to disk — listing it per shard hits the API's
    # rate limit (429). Built once (below), every shard reads the json.
    import json
    vmap_cache = f"{OUT}/vmap.json"
    if os.path.exists(vmap_cache):
        vmap = json.load(open(vmap_cache))
    else:
        vmap = video_repo_map()
        json.dump(vmap, open(vmap_cache, "w"))
    xd_items = [("xd", k, vmap.get(k)) for k in xd_keys if k in vmap]
    kin_files = sorted(glob.glob(f"{KIN}/**/*.mp4", recursive=True))
    kin_items = [("kin", os.path.splitext(os.path.basename(f))[0], f) for f in kin_files]
    items = xd_items + kin_items
    if args.nshards > 1:
        items = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    todo = [it for it in items if not os.path.exists(f"{OUT}/{it[0]}__{safe(it[1])}.npz")]
    print(f"XD {len(xd_items)} + Kinetics {len(kin_items)} = {len(items)} this shard; {len(todo)} to score")

    done = 0
    for src, key, path in todo:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                vp = hf_hub_download(HF_REPO, path, repo_type="dataset", local_dir=tmp) if src == "xd" else path
                frames = extract_frames(vp, tmp, args.k)
                if not frames:
                    continue
                sd = clip_score(scr_def, frames)
                se = clip_score(scr_exp, frames)
            label = is_violent(key) if src == "xd" else 0
            np.savez(f"{OUT}/{src}__{safe(key)}.npz", src=src, key=key,
                     label=np.int64(label), s_def=np.float32(sd), s_exp=np.float32(se))
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
        except Exception as e:
            print(f"  skip {src}:{key[:34]}: {e}")
    print(f"done: {done} -> {OUT}")

    if args.nshards == 1:
        report()


def report():
    from sklearn.metrics import roc_auc_score
    rows = [np.load(f, allow_pickle=True) for f in glob.glob(f"{OUT}/*.npz")]
    src = np.array([str(z["src"]) for z in rows])
    y = np.array([int(z["label"]) for z in rows])
    sd = np.array([float(z["s_def"]) for z in rows])
    se = np.array([float(z["s_exp"]) for z in rows])
    xd = src == "xd"; kin = src == "kin"

    def block(mask, s, tag):
        yy, ss = y[mask], s[mask]
        auc = roc_auc_score(yy, ss) if len(set(yy)) == 2 else float("nan")
        order = np.argsort(-ss)
        p10 = yy[order[:max(1, int(0.10 * len(yy)))]].mean()
        fp = int(((ss >= 0.5) & (yy == 0)).sum())
        print(f"  {tag:<34} n={len(yy):<4} 违反率={yy.mean():.0%}  AUC={auc:.3f}  "
              f"p@top10%={p10:.3f}  FP@0.5={fp}")

    print("\n=== 加正常样本/label 的影响（CLIP 零样本暴力初筛）===")
    print("[基线] 只有 XD：")
    block(xd, sd, "XD only, 默认 prompt")
    print("\n[B 加样本] XD + 450 个 Kinetics 正常片段（默认 prompt）：")
    block(xd | kin, sd, "XD+Kin, 默认 prompt")
    print(f"    其中 Kinetics 正常片段被误报(FP)：{int(((sd[kin]>=0.5)).sum())}/{kin.sum()}"
          f"，Kinetics 违规分中位={np.median(sd[kin]):.2f}")
    print("\n[A 加 label] 只有 XD，但 prompt 池加了 5 个 Kinetics 动作 safe prompt：")
    block(xd, se, "XD only, 扩充 prompt")
    print("\n[A+B] XD + Kinetics，且用扩充 prompt：")
    block(xd | kin, se, "XD+Kin, 扩充 prompt")
    print(f"    Kinetics FP：默认 {int((sd[kin]>=0.5).sum())} → 扩充 {int((se[kin]>=0.5).sum())}"
          f"（加了对应 safe label 后的变化）")


if __name__ == "__main__":
    main()
