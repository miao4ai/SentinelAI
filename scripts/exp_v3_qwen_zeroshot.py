"""V3 step 1 — Qwen2.5-VL zero-shot violence screening (the VLM upgrade of exp5).

Same protocol as the CLIP zero-shot screener (§4.5): the SAME class-balanced clip
sample (seed 0), no training, score each clip and report AUC / F1 / lift. But the
scorer is a 7B vision-language model reading 8 frames, and its answer comes with a
probability from a forced yes/no choice:

    frames16/<key>.npz --8 frames--> Qwen2.5-VL --1st-token logits--> P("yes")

If the old CLIP per-clip cache (clip_zeroshot/) is present, also reports CLIP's AUC
on the intersection for a strictly fair side-by-side. --explain N additionally
generates free-text justifications for N clips — the thing no earlier method could do.

Run on the GPU box: python scripts/exp_v3_qwen_zeroshot.py --n 400 [--explain 6] [--load-4bit]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from exp5_clip_zeroshot import all_i3d_keys, is_violent, safe, sample_keys  # noqa: E402

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
OUT = f"{DATA}/qwen_zs"
CLIP_CACHE = f"{DATA}/clip_zeroshot"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMG = 8

PROMPT = ("You are given {n} frames evenly sampled from one video. Does the video depict "
          "physical violence (fighting, assault, weapons used against people, blood or injury)? "
          "Answer with exactly one word: yes or no.")
PROMPT_EXPLAIN = ("You are given {n} frames evenly sampled from one video. Does the video depict "
                  "physical violence? Start your answer with 'yes' or 'no', then explain in one "
                  "or two sentences what you see that supports your answer.")


def load_frames(key):
    p = f"{FRAMES}/{safe(key)}.npz"
    if not os.path.exists(p):
        return None
    from PIL import Image
    arr = np.load(p, allow_pickle=True)["frames"]          # (16, 224, 224, 3) uint8
    return [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr) // N_IMG)][:N_IMG]


def first_token_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, w.capitalize(), w.upper(), " " + w, " " + w.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--explain", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28)
    kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs = dict(device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = AutoModelForImageTextToText.from_pretrained(MODEL, **kwargs).eval()
    yes_ids = first_token_ids(proc.tokenizer, ["yes"])
    no_ids = first_token_ids(proc.tokenizer, ["no"])

    def ask(imgs, prompt, max_new=1):
        msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in imgs],
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        return out, inputs

    def p_yes(imgs):
        out, _ = ask(imgs, PROMPT.format(n=len(imgs)), max_new=1)
        logits = out.scores[0][0].float()
        ly = torch.logsumexp(logits[yes_ids], 0)
        ln = torch.logsumexp(logits[no_ids], 0)
        return torch.softmax(torch.stack([ly, ln]), 0)[0].item()

    keys = [k for k in sample_keys(all_i3d_keys(), args.n, 0)
            if os.path.exists(f"{FRAMES}/{safe(k)}.npz")]
    todo = [k for k in keys if not os.path.exists(f"{OUT}/{safe(k)}.npz")]
    print(f"{len(keys)} clips with frames; {len(todo)} to score")

    done = 0
    for k in todo:
        try:
            imgs = load_frames(k)
            if not imgs:
                continue
            s = p_yes(imgs)
            np.savez(f"{OUT}/{safe(k)}.npz", key=k, label=np.int64(is_violent(k)), score=np.float32(s))
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
        except Exception as e:
            print(f"  skip {k[:40]}: {e}")
    print(f"scored {done} new clips")

    # ── report ──
    from sklearn.metrics import f1_score, roc_auc_score
    rows = [np.load(f, allow_pickle=True) for f in glob.glob(f"{OUT}/*.npz")]
    kk = [str(z["key"]) for z in rows]
    y = np.array([int(z["label"]) for z in rows])
    s = np.array([float(z["score"]) for z in rows])
    order = np.argsort(-s); k10 = max(1, int(0.10 * len(y)))
    print(f"\n=== Qwen2.5-VL zero-shot ({len(y)} clips, {y.mean():.0%} violent) ===")
    print(f"  AUC={roc_auc_score(y, s):.3f}  F1@0.5={f1_score(y, (s >= 0.5).astype(int)):.3f}"
          f"  p@top10%={y[order[:k10]].mean():.3f} (lift {y[order[:k10]].mean()/y.mean():.2f})")

    # fair side-by-side with CLIP zero-shot on the intersection, if its cache exists
    clip = {}
    for f in glob.glob(f"{CLIP_CACHE}/*.npz"):
        z = np.load(f, allow_pickle=True); clip[str(z["key"])] = (int(z["label"]), float(z["score"]))
    inter = [i for i, key in enumerate(kk) if key in clip]
    if len(inter) > 50:
        yi = y[inter]; sq = s[inter]; sc = np.array([clip[kk[i]][1] for i in inter])
        print(f"  同片段对比 ({len(inter)} clips):  Qwen AUC={roc_auc_score(yi, sq):.3f}"
              f"  vs  CLIP AUC={roc_auc_score(yi, sc):.3f}")

    # qualitative: free-text explanations
    if args.explain:
        idx = list(np.argsort(-s)[: args.explain // 2]) + list(np.argsort(s)[: args.explain - args.explain // 2])
        print("\n=== 解释示例（VLM 独有）===")
        for i in idx:
            imgs = load_frames(kk[i])
            out, inputs = ask(imgs, PROMPT_EXPLAIN.format(n=len(imgs)), max_new=96)
            txt = proc.tokenizer.decode(out.sequences[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            print(f"\n[{kk[i][:55]}] 真实={'暴力' if y[i] else '正常'} P(yes)={s[i]:.2f}\n  {txt.strip()}")


if __name__ == "__main__":
    main()
