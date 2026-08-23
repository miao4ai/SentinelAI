"""Extract the TEXT modality for XD-Violence via ASR (experiment 3 feature pipeline).

Visual (I3D) and audio (AST) came pre-extracted; TEXT does not — we build it here:

    raw video (HF) --ffmpeg--> 16kHz wav --faster-whisper--> transcript
                 --XLM-R (mean-pooled)--> 768-d text embedding  (cached)

Only clips that already have BOTH visual and audio are processed (the pairable set).
Clips with no/negligible speech (action scenes) get a zero embedding and
has_speech=False. Resumable (skips cached clips) and cleans up temp media.

Deps: `pip install faster-whisper` (transformers/ffmpeg already present).
Run on the GPU box:  python scripts/extract_text_features.py --limit 300
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq
import torch

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
I3D = f"{DATA}/data/i3d_rgb"
AUDIO = f"{DATA}/audio_ast"
OUT = f"{DATA}/text_features"
HF_REPO = "jherng/xd-violence"          # has the raw videos under data/video/
TEXT_MODEL = "unitary/multilingual-toxic-xlm-roberta"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]


def base(n: str) -> str:
    return n.replace(".npy", "").replace(".mp4", "")


def safe(key: str) -> str:
    """Filesystem-safe cache name for a clip key."""
    return "".join(c if c.isalnum() or c in "._-#" else "_" for c in key)


def pairable_keys() -> list[str]:
    """Clips that already have BOTH visual (I3D) and audio (AST) features."""
    i3d = {base(os.path.basename(f)) for d in DIRS for f in glob.glob(f"{I3D}/{d}/*.npy")}
    aud = set()
    for p in glob.glob(f"{AUDIO}/**/*.parquet", recursive=True):
        for v in pq.read_table(p, columns=["video_id"]).to_pandas()["video_id"]:
            aud.add(base(v))
    return sorted(i3d & aud)


def video_repo_map() -> dict[str, str]:
    """Map clip key -> its .mp4 path inside the HF repo (robust to folder layout)."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(HF_REPO, repo_type="dataset")
    return {base(os.path.basename(f)): f for f in files if f.endswith(".mp4")}


def extract_wav(video_path: str, wav_path: str) -> bool:
    """ffmpeg: video -> 16kHz mono wav. Returns False if there is no audio stream."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video_path,
         "-ac", "1", "-ar", "16000", wav_path],
        capture_output=True,
    )
    return r.returncode == 0 and os.path.getsize(wav_path) > 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N new clips (0 = all)")
    ap.add_argument("--whisper", default="base", help="faster-whisper model size")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from faster_whisper import WhisperModel
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    asr = WhisperModel(args.whisper, device=device, compute_type="float16" if device == "cuda" else "int8")
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL, output_hidden_states=True
    ).to(device).eval()

    def embed(text: str) -> tuple[np.ndarray, bool]:
        """Mean-pooled XLM-R embedding of the transcript; zeros if no speech."""
        if not text.strip():
            return np.zeros(768, dtype=np.float32), False
        enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            h = mdl(**enc).hidden_states[-1].mean(dim=1).squeeze(0)
        return h.cpu().numpy().astype(np.float32), True

    keys = pairable_keys()
    todo = [k for k in keys if not os.path.exists(f"{OUT}/{safe(k)}.npz")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(keys)} pairable clips; {len(todo)} to process")
    vmap = video_repo_map()

    done = 0
    for k in todo:
        if k not in vmap:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                vp = hf_hub_download(HF_REPO, vmap[k], repo_type="dataset", local_dir=tmp)
                wav = os.path.join(tmp, "a.wav")
                if extract_wav(vp, wav):
                    segments, _ = asr.transcribe(wav)
                    text = " ".join(s.text for s in segments).strip()
                else:
                    text = ""
            emb, has_speech = embed(text)
            np.savez(f"{OUT}/{safe(k)}.npz", key=k, embedding=emb, text=text, has_speech=has_speech)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(todo)}  last='{text[:50]}' speech={has_speech}")
        except Exception as e:  # keep going on any single-clip failure
            print(f"  skip {k[:40]}: {e}")

    print(f"done: {done} clips -> {OUT}")


if __name__ == "__main__":
    main()
