"""V3 zero-shot, cross-dataset — can world knowledge cross the domain gap features couldn't?

§4.8 showed feature-based transfer XD<->UCF is at chance. But a VLM was trained on
neither domain and carries general world knowledge. This scores UCF-Crime (balanced
sample) and the 450 Kinetics normals with BOTH Qwen2.5-VL and CLIP on the SAME 8
frames per video (seek-based extraction, so long CCTV files don't need full decode):

  * UCF   — zero-shot AUC vs the at-chance feature transfer and the supervised 0.958.
  * Kinetics — the §4.7 stress test, VLM edition: how many benign clips get flagged?
               (CLIP with the default pool flagged 365/450.)

Caches one npz per clip (key, label, q=Qwen P(yes), c=CLIP max-frame score); resumable.
Run on the GPU box: python scripts/exp_v3_qwen_crossdataset.py [--n-ucf 300]
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np

HOME = os.path.expanduser("~/documents/SentinelAI/data")
UCF_VIDEOS = f"{HOME}/ucf-crime/videos"
KIN_VIDEOS = f"{HOME}/kinetics400/extracted"
UCF_OUT = f"{HOME}/ucf-crime/qwen_zs"
KIN_OUT = f"{HOME}/kinetics400/qwen_zs"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMG = 8

PROMPT = ("You are given {n} frames evenly sampled from one video. Does the video depict "
          "physical violence (fighting, assault, weapons used against people, blood or injury)? "
          "Answer with exactly one word: yes or no.")


def safe(k): return "".join(c if c.isalnum() or c in "._-#" else "_" for c in k)


def frames8(video_path):
    """8 seek-extracted 224px frames — fast even for long CCTV files."""
    from PIL import Image
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", video_path]).strip() or 0.0)
    except Exception:
        dur = 0.0
    dur = dur if dur > 0 else 10.0
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(N_IMG):
            t = dur * (i + 0.5) / N_IMG
            out = f"{tmp}/f_{i}.jpg"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}",
                            "-i", video_path, "-frames:v", "1", "-vf", "scale=224:224", out],
                           check=False)
            if os.path.exists(out):
                imgs.append(Image.open(out).convert("RGB").copy())
    return imgs


def first_token_ids(tok, word):
    ids = set()
    for v in (word, word.capitalize(), word.upper(), " " + word, " " + word.capitalize()):
        t = tok.encode(v, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ucf", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(UCF_OUT, exist_ok=True)
    os.makedirs(KIN_OUT, exist_ok=True)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from sentinelai.clip_screener import ClipScreener
    from sklearn.metrics import f1_score, roc_auc_score

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                        device_map="auto").eval()
    yes_ids = first_token_ids(proc.tokenizer, "yes")
    no_ids = first_token_ids(proc.tokenizer, "no")
    clip = ClipScreener()

    def p_yes(imgs):
        msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in imgs],
                                             {"type": "text", "text": PROMPT.format(n=len(imgs))}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        logits = out.scores[0][0].float()
        ly = torch.logsumexp(logits[yes_ids], 0)
        ln = torch.logsumexp(logits[no_ids], 0)
        return torch.softmax(torch.stack([ly, ln]), 0)[0].item()

    # ── build work lists ──
    rng = np.random.default_rng(0)
    ucf_all = sorted(glob.glob(f"{UCF_VIDEOS}/**/*.mp4", recursive=True))
    ucf_norm = [f for f in ucf_all if "Normal" in os.path.basename(f)]
    ucf_anom = [f for f in ucf_all if "Normal" not in os.path.basename(f)]
    half = args.n_ucf // 2
    pick = lambda pool: [pool[i] for i in rng.permutation(len(pool))[:half]]
    jobs = [("ucf", f, 0 if "Normal" in os.path.basename(f) else 1, UCF_OUT)
            for f in pick(ucf_anom) + pick(ucf_norm)]
    jobs += [("kin", f, 0, KIN_OUT) for f in sorted(glob.glob(f"{KIN_VIDEOS}/**/*.mp4", recursive=True))]
    print(f"UCF sample {min(args.n_ucf, len(ucf_norm)+len(ucf_anom))} + Kinetics "
          f"{len([j for j in jobs if j[0]=='kin'])}")

    done = 0
    for src, vp, label, outdir in jobs:
        key = os.path.splitext(os.path.basename(vp))[0]
        dst = f"{outdir}/{safe(key)}.npz"
        if os.path.exists(dst):
            continue
        try:
            imgs = frames8(vp)
            if len(imgs) < 2:
                continue
            q = p_yes(imgs)
            c = max(r.violation_prob for r in clip.score_frames([np.asarray(im) for im in imgs]))
            np.savez(dst, key=key, label=np.int64(label), q=np.float32(q), c=np.float32(c))
            done += 1
            if done % 25 == 0:
                print(f"  {done}", flush=True)
        except Exception as e:
            print(f"  skip {key[:36]}: {e}")
    print(f"scored {done} new")

    # ── reports ──
    def load(outdir):
        rows = [np.load(f, allow_pickle=True) for f in glob.glob(f"{outdir}/*.npz")]
        return (np.array([int(z["label"]) for z in rows]),
                np.array([float(z["q"]) for z in rows]),
                np.array([float(z["c"]) for z in rows]))

    y, q, c = load(UCF_OUT)
    print(f"\n=== UCF 零样本（{len(y)} clips, {y.mean():.0%} anomaly, 同帧对比）===")
    for tag, s in [("Qwen2.5-VL", q), ("CLIP      ", c)]:
        o = np.argsort(-s); k = max(1, len(y) // 10)
        print(f"  {tag}: AUC={roc_auc_score(y,s):.3f}  F1@0.5={f1_score(y,(s>=0.5).astype(int)):.3f}"
              f"  p@top10%={y[o[:k]].mean():.3f} (lift {y[o[:k]].mean()/y.mean():.2f})")

    yk, qk, ck = load(KIN_OUT)
    print(f"\n=== Kinetics 450 正常片段（§4.7 压测的 VLM 版）===")
    print(f"  Qwen 误报@0.5: {int((qk>=0.5).sum())}/{len(qk)}  中位 P(yes)={np.median(qk):.3f}")
    print(f"  CLIP 误报@0.5: {int((ck>=0.5).sum())}/{len(ck)}  中位分={np.median(ck):.3f}"
          f"   (exp7 默认池当年: 365/450, 中位 0.81)")


if __name__ == "__main__":
    main()
