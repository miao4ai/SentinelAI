"""Experiment 5 — validate CLIP zero-shot screening on real XD-Violence clips.

Chapter 5's claim: we can flag violent frames with **no training** by scoring them
against a text prompt pool in CLIP's shared space (`sentinelai/clip_screener.py`).
This proves it on real data: does the zero-shot score actually separate violent
from normal clips, and is it good enough for a fast pre-filter?

Pipeline (per sampled clip):
    raw video (HF) --ffmpeg--> K evenly-spaced frames --CLIP zero-shot--> per-frame
    violation_prob --max over frames--> one clip-level screening score

We take the **max** over frames because a pre-filter should flag a clip if *any*
frame looks violent. Zero-shot means nothing is fitted, so there is no train/test
split to leak — we score the sampled clips directly and report:
  - AUC (threshold-free separability — the honest headline for a screener),
  - F1 / precision / recall at a 0.5 threshold,
  - precision@k, the metric that matters for a pre-filter (of the clips it flags
    most confidently, how many are truly violent).

Run on the GPU box: python scripts/exp5_clip_zeroshot.py --n 400 --k 8
Shardable like the audio job: --shard i --nshards N (writes per-clip score cache).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
OUT = f"{DATA}/clip_zeroshot"          # per-clip score cache (resumable)
HF_REPO = "jherng/xd-violence"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def is_violent(n): return 0 if "_label_A" in n else 1
def safe(key): return "".join(c if c.isalnum() or c in "._-#" else "_" for c in key)


def all_i3d_keys() -> list[str]:
    return sorted(base(os.path.basename(f))
                  for d in DIRS for f in glob.glob(f"{I3D}/{d}/*.npy"))


def video_repo_map() -> dict[str, str]:
    from huggingface_hub import list_repo_files
    files = list_repo_files(HF_REPO, repo_type="dataset")
    return {base(os.path.basename(f)): f for f in files if f.endswith(".mp4")}


def sample_keys(keys: list[str], n: int, seed: int = 0) -> list[str]:
    """Class-balanced sample of n keys (n/2 violent, n/2 normal), deterministic."""
    rng = np.random.default_rng(seed)
    vio = [k for k in keys if is_violent(k)]
    norm = [k for k in keys if not is_violent(k)]
    half = n // 2
    pick = lambda pool: [pool[i] for i in rng.permutation(len(pool))[:half]]
    return pick(vio) + pick(norm)


def extract_frames(video_path: str, out_dir: str, k: int) -> list[str]:
    """Dump up to k evenly-spaced JPEG frames from the video via ffmpeg."""
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", video_path]).strip() or 0.0)
    except Exception:
        dur = 0.0
    dur = dur if dur > 0 else 10.0
    fps = max(k / dur, 0.001)          # k frames spread across the whole clip
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={fps}", "-frames:v", str(k), f"{out_dir}/f_%03d.jpg"],
        check=True)
    return sorted(glob.glob(f"{out_dir}/f_*.jpg"))


def report(y: np.ndarray, s: np.ndarray) -> None:
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)
    pred = (s >= 0.5).astype(int)
    print(f"\nzero-shot screening on {len(y)} clips ({y.mean():.0%} violent)")
    print(f"  AUC              {roc_auc_score(y, s):.3f}   (threshold-free separability)")
    print(f"  F1 @0.5          {f1_score(y, pred, zero_division=0):.3f}")
    print(f"  precision @0.5   {precision_score(y, pred, zero_division=0):.3f}")
    print(f"  recall @0.5      {recall_score(y, pred, zero_division=0):.3f}")
    # precision@k: of the top-scored clips (what a pre-filter would escalate first)
    order = np.argsort(-s)
    for frac in (0.1, 0.25, 0.5):
        kk = max(1, int(frac * len(y)))
        print(f"  precision@top-{int(frac*100):>2}%  {y[order[:kk]].mean():.3f}  "
              f"(of the {kk} highest-scored clips)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="clips to sample (class-balanced)")
    ap.add_argument("--k", type=int, default=8, help="frames per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default=None, help="override CLIP model (e.g. OFA-Sys/chinese-clip-vit-base-patch16)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from huggingface_hub import hf_hub_download

    from sentinelai.clip_screener import ClipScreener

    keys = sample_keys(all_i3d_keys(), args.n, args.seed)
    if args.nshards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.nshards == args.shard]
    todo = [k for k in keys if not os.path.exists(f"{OUT}/{safe(k)}.npz")]
    print(f"sampled {args.n} clips; this shard handles {len(keys)}, {len(todo)} to score")

    screener = ClipScreener(**({"model_name": args.model} if args.model else {}))
    vmap = video_repo_map()

    done = 0
    for key in todo:
        if key not in vmap:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                vp = hf_hub_download(HF_REPO, vmap[key], repo_type="dataset", local_dir=tmp)
                frames = extract_frames(vp, tmp, args.k)
                if not frames:
                    continue
                fs = screener.score_frames(frames)
            clip_score = max(f.violation_prob for f in fs)     # flag if ANY frame looks violent
            np.savez(f"{OUT}/{safe(key)}.npz", key=key, score=np.float32(clip_score),
                     label=np.int64(is_violent(key)))
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
        except Exception as e:
            print(f"  skip {key[:40]}: {e}")
    print(f"done: scored {done} clips -> {OUT}")

    # Aggregate whatever is cached so far (all shards write to the same dir).
    if args.nshards == 1:
        cached = [np.load(f) for f in glob.glob(f"{OUT}/*.npz")]
        y = np.array([int(z["label"]) for z in cached])
        s = np.array([float(z["score"]) for z in cached])
        if len(y):
            report(y, s)


if __name__ == "__main__":
    main()
