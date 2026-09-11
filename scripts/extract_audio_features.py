"""Extract AST audio features for ALL XD-Violence clips (more training data).

We only had audio for 788 clips (a 20% subset); visual I3D covers all ~4747. This
runs our own AST audio expert over every clip so the paired visual+audio set jumps
to the full ~4747, giving the learned fusions enough data to actually compete.

    raw video (HF) --(ffmpeg inside AudioExpert)--> 16kHz audio --AST--> 768-d
                    --mean over windows--> one clip-level audio embedding (cached)

Resumable (skips cached), cleans up temp videos. Runs the AST model on GPU.
Run: python scripts/extract_audio_features.py [--limit N]
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile

import numpy as np

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
OUT = f"{DATA}/audio_full"
OUT_SEQ = f"{DATA}/audio_seq"
HF_REPO = "jherng/xd-violence"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]


def base(n: str) -> str:
    return n.replace(".npy", "").replace(".mp4", "")


def safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "._-#" else "_" for c in key)


def all_i3d_keys() -> list[str]:
    """Every clip that has an I3D visual feature (what we want audio to pair with)."""
    return sorted(
        base(os.path.basename(f))
        for d in DIRS for f in glob.glob(f"{I3D}/{d}/*.npy")
    )


def video_repo_map() -> dict[str, str]:
    # Shared cached listing (add_normals/vmap.json) so parallel shards don't hit
    # the HF API rate limit (429). Built once by whichever process gets there first.
    import json
    cache = f"{DATA}/add_normals/vmap.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from huggingface_hub import list_repo_files

    files = list_repo_files(HF_REPO, repo_type="dataset")
    vmap = {base(os.path.basename(f)): f for f in files if f.endswith(".mp4")}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(vmap, open(cache, "w"))
    return vmap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    # Shard the work across N parallel workers (downloads are network-bound, so
    # a few processes sharing one GPU finish far faster than one). Worker i takes
    # every clip whose index % nshards == i, so shards never touch the same clip.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    # (--seq kept for compat but is now a no-op: every clip is downloaded ONCE and
    # both outputs are written — the clip mean to audio_full/ and the full
    # (num_windows, 768) AST sequence to audio_seq/.)
    ap.add_argument("--seq", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_SEQ, exist_ok=True)

    from huggingface_hub import hf_hub_download

    from sentinelai.audio_expert import AudioExpert

    expert = AudioExpert()   # AST, uses GPU if available

    keys = all_i3d_keys()
    if args.nshards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.nshards == args.shard]
    todo = [k for k in keys
            if not (os.path.exists(f"{OUT}/{safe(k)}.npz") and os.path.exists(f"{OUT_SEQ}/{safe(k)}.npz"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(keys)} i3d clips; {len(todo)} need audio features")
    vmap = video_repo_map()

    done = 0
    for k in todo:
        if k not in vmap:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                vp = hf_hub_download(HF_REPO, vmap[k], repo_type="dataset", local_dir=tmp)
                # AudioExpert decodes the audio track from the video via ffmpeg itself.
                feats = expert.extract_features(vp)          # (num_windows, 768)
            seq = feats.astype(np.float32) if len(feats) else np.zeros((1, 768), np.float32)
            emb = feats.mean(axis=0) if len(feats) else np.zeros(768, dtype=np.float32)
            np.savez(f"{OUT}/{safe(k)}.npz", key=k, embedding=emb.astype(np.float32))
            np.savez(f"{OUT_SEQ}/{safe(k)}.npz", key=k, sequence=seq)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}")
        except Exception as e:
            print(f"  skip {k[:40]}: {e}")

    print(f"done: {done} clips -> {OUT} + {OUT_SEQ}")


if __name__ == "__main__":
    main()
