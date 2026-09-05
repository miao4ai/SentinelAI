"""Extract 16 raw frames (224x224) per XD clip — inputs for end-to-end VideoMAE FT.

The VideoMAE experiments so far used *frozen* features and lost to I3D. To actually
beat I3D the video encoder has to be fine-tuned end-to-end on the violence task,
which needs raw frames in the training loop. This caches 16 evenly-spaced RGB
frames (VideoMAE's clip length) per clip as compressed uint8, so fine-tuning reads
from disk instead of re-decoding video every epoch.

    raw video (HF) --ffmpeg(16 frames @224)--> (16, 224, 224, 3) uint8 -> frames16/<key>.npz

Sharded + resumable. Run: python scripts/extract_xd_frames.py [--shard i --nshards N]
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
OUT = f"{DATA}/frames16"
HF_REPO = "jherng/xd-violence"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
N_FRAMES, SIZE = 16, 224


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def safe(k): return "".join(c if c.isalnum() or c in "._-#" else "_" for c in k)


def all_i3d_keys():
    return sorted(base(os.path.basename(f)) for d in DIRS for f in glob.glob(f"{I3D}/{d}/*.npy"))


def video_repo_map():
    from huggingface_hub import list_repo_files
    files = list_repo_files(HF_REPO, repo_type="dataset")
    return {base(os.path.basename(f)): f for f in files if f.endswith(".mp4")}


def extract16(video_path, tmp):
    from PIL import Image
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", video_path]).strip() or 0.0)
    except Exception:
        dur = 0.0
    dur = dur if dur > 0 else 10.0
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video_path, "-vf",
                    f"fps={N_FRAMES/dur},scale={SIZE}:{SIZE}", "-frames:v", str(N_FRAMES),
                    f"{tmp}/f_%02d.jpg"], check=True)
    imgs = [np.asarray(Image.open(f).convert("RGB").resize((SIZE, SIZE))) for f in sorted(glob.glob(f"{tmp}/f_*.jpg"))]
    if not imgs:
        return None
    while len(imgs) < N_FRAMES:
        imgs.append(imgs[-1])
    return np.stack(imgs[:N_FRAMES]).astype(np.uint8)      # (16, 224, 224, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from huggingface_hub import hf_hub_download

    keys = all_i3d_keys()
    if args.nshards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.nshards == args.shard]
    todo = [k for k in keys if not os.path.exists(f"{OUT}/{safe(k)}.npz")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} clips to extract (this shard)")
    vmap = video_repo_map()

    done = 0
    for k in todo:
        if k not in vmap:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                vp = hf_hub_download(HF_REPO, vmap[k], repo_type="dataset", local_dir=tmp)
                frames = extract16(vp, tmp)
            if frames is None:
                continue
            np.savez_compressed(f"{OUT}/{safe(k)}.npz", key=k, frames=frames)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}")
        except Exception as e:
            print(f"  skip {k[:40]}: {e}")
    print(f"done: {done} -> {OUT}")


if __name__ == "__main__":
    main()
