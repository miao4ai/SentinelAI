"""Extract audio (AST) + text (Whisper->XLM-R) for UCF-Crime videos.

UCF-Crime on GCS was I3D-only; the raw videos (jinmang2/ucf_crime zips) mostly DO
have audio, so we can add the audio and text modalities and run real multimodal
fusion on UCF (§ cross-dataset). Reads the unzipped local videos, caches a 768-d
AST audio embedding and a 768-d XLM-R text embedding per clip (resumable, shardable).

    data/ucf-crime/videos/**/*.mp4  --AST-->        audio/<key>.npz
                                    --Whisper->XLM-R--> text/<key>.npz

Run on the GPU box: python scripts/extract_ucf_multimodal.py [--audio-only] [--shard i --nshards N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from extract_text_features import TEXT_MODEL, extract_wav, safe  # noqa: E402

DATA = os.path.expanduser("~/documents/SentinelAI/data/ucf-crime")
VIDEOS = f"{DATA}/videos"
A_OUT = f"{DATA}/audio"
T_OUT = f"{DATA}/text"


def key_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-only", action="store_true", help="skip the slow Whisper text pass")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(A_OUT, exist_ok=True)
    os.makedirs(T_OUT, exist_ok=True)

    import torch

    from sentinelai.audio_expert import AudioExpert

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; audio_only={args.audio_only}")
    aexp = AudioExpert()

    asr = tok = mdl = None
    if not args.audio_only:
        from faster_whisper import WhisperModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        asr = WhisperModel(args.whisper, device=device, compute_type="float16" if device == "cuda" else "int8")
        tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL, output_hidden_states=True).to(device).eval()

    def embed(text: str):
        if not text.strip():
            return np.zeros(768, dtype=np.float32), False
        enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            h = mdl(**enc).hidden_states[-1].mean(dim=1).squeeze(0)
        return h.cpu().numpy().astype(np.float32), True

    vids = sorted(glob.glob(f"{VIDEOS}/**/*.mp4", recursive=True))
    if args.nshards > 1:
        vids = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
    print(f"{len(vids)} UCF videos this shard")

    done = 0
    for vp in vids:
        k = key_of(vp)
        a_path, t_path = f"{A_OUT}/{safe(k)}.npz", f"{T_OUT}/{safe(k)}.npz"
        need_a = not os.path.exists(a_path)
        need_t = (not args.audio_only) and (not os.path.exists(t_path))
        if not need_a and not need_t:
            continue
        try:
            if need_a:
                feats = aexp.extract_features(vp)                 # (num_windows, 768)
                emb = feats.mean(axis=0) if len(feats) else np.zeros(768, np.float32)
                np.savez(a_path, key=k, embedding=emb.astype(np.float32))
            if need_t:
                with tempfile.TemporaryDirectory() as tmp:
                    wav = os.path.join(tmp, "a.wav")
                    text = ""
                    if extract_wav(vp, wav):
                        segments, _ = asr.transcribe(wav)
                        text = " ".join(s.text for s in segments).strip()
                temb, has = embed(text)
                np.savez(t_path, key=k, embedding=temb, text=text, has_speech=has)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(vids)}")
        except Exception as e:
            print(f"  skip {k[:40]}: {e}")
    print(f"done: {done} -> {A_OUT}, {T_OUT}")


if __name__ == "__main__":
    main()
