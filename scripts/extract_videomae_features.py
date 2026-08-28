"""Extract VideoMAE temporal features for XD clips (V2 ch.6.1, proper temporal).

exp6 used I3D (a 3D CNN) as the video K/V. ch.6.1 asks for a transformer temporal
encoder — VideoMAE / TimeSformer — that models frame-to-frame motion, so the
cross-attention can tell "cutting vegetables" from "stabbing". This extracts a
per-window VideoMAE sequence per clip so we can swap it in for I3D and compare.

    raw video (HF) --ffmpeg--> 128 frames --> 8 windows x 16 frames
                    --VideoMAE (mean over patch tokens)--> (8, 768) temporal seq

Only the clips that already have audio (the exp6 pairing universe) are extracted.
Sharded + resumable, like extract_audio_features.py.

Run: python scripts/extract_videomae_features.py [--shard i --nshards N] [--limit N]
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
AUDIO_FULL = f"{DATA}/audio_full"
OUT = f"{DATA}/videomae_seq"
HF_REPO = "jherng/xd-violence"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
MODEL = "MCG-NJU/videomae-base"
T_WINDOWS, FRAMES_PER = 8, 16          # 8 temporal windows, each a 16-frame VideoMAE clip


def base(n): return n.replace(".npy", "").replace(".mp4", "")
def safe(key): return "".join(c if c.isalnum() or c in "._-#" else "_" for c in key)


def paired_keys() -> list[str]:
    """Clips that have BOTH an I3D feature and our audio feature (exp6's universe)."""
    # audio_full filenames are safe(key); read the real key stored inside each npz.
    audio = {str(np.load(f, allow_pickle=True)["key"]) for f in glob.glob(f"{AUDIO_FULL}/*.npz")}
    i3d = {base(os.path.basename(f)) for d in DIRS for f in glob.glob(f"{I3D}/{d}/*.npy")}
    return sorted(i3d & audio)


def video_repo_map() -> dict[str, str]:
    from huggingface_hub import list_repo_files
    files = list_repo_files(HF_REPO, repo_type="dataset")
    return {base(os.path.basename(f)): f for f in files if f.endswith(".mp4")}


def extract_frames(video_path: str, out_dir: str, n: int) -> list:
    """Dump ~n evenly-spaced frames via ffmpeg and load them as RGB arrays."""
    from PIL import Image
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", video_path]).strip() or 0.0)
    except Exception:
        dur = 0.0
    dur = dur if dur > 0 else 10.0
    fps = max(n / dur, 0.001)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={fps}", "-frames:v", str(n), f"{out_dir}/f_%03d.jpg"], check=True)
    return [np.array(Image.open(f).convert("RGB")) for f in sorted(glob.glob(f"{out_dir}/f_*.jpg"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = VideoMAEImageProcessor.from_pretrained(MODEL)
    model = VideoMAEModel.from_pretrained(MODEL).to(device).eval()

    keys = paired_keys()
    if args.nshards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.nshards == args.shard]
    todo = [k for k in keys if not os.path.exists(f"{OUT}/{safe(k)}.npz")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(paired_keys())} paired clips; this shard {len(keys)}, {len(todo)} to extract")
    vmap = video_repo_map()

    need = T_WINDOWS * FRAMES_PER
    done = 0
    for k in todo:
        if k not in vmap:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                vp = hf_hub_download(HF_REPO, vmap[k], repo_type="dataset", local_dir=tmp)
                imgs = extract_frames(vp, tmp, need)
            if not imgs:
                continue
            while len(imgs) < need:                 # pad short clips by repeating the last frame
                imgs.append(imgs[-1])
            imgs = imgs[:need]
            windows = [imgs[i * FRAMES_PER:(i + 1) * FRAMES_PER] for i in range(T_WINDOWS)]
            inputs = proc(windows, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)               # last_hidden_state (8, 1568, 768)
            seq = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)  # (8, 768)
            np.savez(f"{OUT}/{safe(k)}.npz", key=k, sequence=seq)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
        except Exception as e:
            print(f"  skip {k[:40]}: {e}")
    print(f"done: {done} clips -> {OUT}")


if __name__ == "__main__":
    main()
