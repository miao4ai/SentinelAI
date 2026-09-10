"""V3 close-out — does the XD-tuned QLoRA adapter keep its world knowledge?

Two loose ends from §4.10:
  A. Explanations — the adapter was trained to answer bare yes/no; can it still
     produce grounded free-text justifications when asked?
  B. Cross-domain — §4.9 celebrated zero-shot domain robustness (UCF 0.879,
     Kinetics 0/450 false alarms), §4.4 showed task-fine-tuned encoders transfer
     WORSE (k400 collapse). Which fate does our own fine-tune meet? Re-score the
     same UCF/Kinetics clips with the adapter ON and OFF (both under 4-bit, so the
     pairing is exact).

Caches per-clip (resumable), saves a summary, meant to run under the self-shutdown
driver. Run: python scripts/exp_v3_qlora_tails.py [--explain 6]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from exp_v3_qwen_crossdataset import KIN_VIDEOS, UCF_VIDEOS, frames8, safe  # noqa: E402

DATA = os.path.expanduser("~/documents/SentinelAI/data")
ADAPTER = f"{DATA}/xd-violence/qwen_qlora_adapter"
XD_FRAMES = f"{DATA}/xd-violence/frames16"
UCF_ZS = f"{DATA}/ucf-crime/qwen_zs"          # zero-shot caches from exp_v3_qwen_crossdataset
KIN_ZS = f"{DATA}/kinetics400/qwen_zs"
UCF_OUT = f"{DATA}/ucf-crime/qwen_qlora_zs"
KIN_OUT = f"{DATA}/kinetics400/qwen_qlora_zs"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMG = 8
PROMPT = ("You are given {n} frames evenly sampled from one video. Does the video depict "
          "physical violence (fighting, assault, weapons used against people, blood or injury)? "
          "Answer with exactly one word: yes or no.")
PROMPT_EXPLAIN = ("You are given {n} frames evenly sampled from one video. Does the video depict "
                  "physical violence? Start your answer with 'yes' or 'no', then explain in one "
                  "or two sentences what you see that supports your answer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(UCF_OUT, exist_ok=True)
    os.makedirs(KIN_OUT, exist_ok=True)

    import torch
    from peft import PeftModel
    from sklearn.metrics import roc_auc_score
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig)

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(MODEL, quantization_config=bnb,
                                                       device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    yes_ids = sorted({proc.tokenizer.encode(v, add_special_tokens=False)[0]
                      for v in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({proc.tokenizer.encode(v, add_special_tokens=False)[0]
                     for v in ("no", "No", " no", " No")})

    def gen(imgs, prompt, max_new=1):
        msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in imgs],
                                             {"type": "text", "text": prompt.format(n=len(imgs))}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        return out, enc

    def p_yes(imgs):
        out, _ = gen(imgs, PROMPT, 1)
        logits = out.scores[0][0].float()
        ly = torch.logsumexp(logits[yes_ids], 0)
        ln = torch.logsumexp(logits[no_ids], 0)
        return torch.softmax(torch.stack([ly, ln]), 0)[0].item()

    # ── A. explanations from the tuned model ──
    if args.explain:
        from PIL import Image
        files = sorted(glob.glob(f"{XD_FRAMES}/*.npz"))
        vio = [f for f in files if "_label_A" not in f][: args.explain // 2]
        nor = [f for f in files if "_label_A" in f][: args.explain - args.explain // 2]
        print("=== A. QLoRA 模型的解释能力 ===")
        for f in vio + nor:
            arr = np.load(f, allow_pickle=True)["frames"]
            imgs = [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr) // N_IMG)][:N_IMG]
            out, enc = gen(imgs, PROMPT_EXPLAIN, 96)
            txt = proc.tokenizer.decode(out.sequences[0][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            lab = "正常" if "_label_A" in f else "暴力"
            print(f"\n[{os.path.basename(f)[:52]}] 真实={lab}\n  {txt.strip()}", flush=True)

    # ── B. paired cross-domain re-score (adapter on/off, both 4-bit) ──
    vmap = {}
    for root in (UCF_VIDEOS, KIN_VIDEOS):
        for p in glob.glob(f"{root}/**/*.mp4", recursive=True):
            vmap[os.path.splitext(os.path.basename(p))[0]] = p

    def rescore(zs_dir, out_dir):
        for f in sorted(glob.glob(f"{zs_dir}/*.npz")):
            key = str(np.load(f, allow_pickle=True)["key"])
            dst = f"{out_dir}/{safe(key)}.npz"
            if os.path.exists(dst) or key not in vmap:
                continue
            try:
                imgs = frames8(vmap[key])
                if len(imgs) < 2:
                    continue
                s_on = p_yes(imgs)
                with model.disable_adapter():
                    s_off = p_yes(imgs)
                z = np.load(f, allow_pickle=True)
                np.savez(dst, key=key, label=z["label"], s_on=np.float32(s_on),
                         s_off=np.float32(s_off))
            except Exception as e:
                print(f"  skip {key[:36]}: {e}", flush=True)

    print("\n=== B. 跨域配对重打分 ===", flush=True)
    rescore(UCF_ZS, UCF_OUT)
    print("UCF done", flush=True)
    rescore(KIN_ZS, KIN_OUT)
    print("Kinetics done", flush=True)

    def load(d):
        rows = [np.load(f, allow_pickle=True) for f in glob.glob(f"{d}/*.npz")]
        return (np.array([int(z["label"]) for z in rows]),
                np.array([float(z["s_on"]) for z in rows]),
                np.array([float(z["s_off"]) for z in rows]))

    y, on, off = load(UCF_OUT)
    print(f"\nUCF ({len(y)} clips, {y.mean():.0%} anomaly):")
    for tag, s in [("零样本(4bit, off)", off), ("QLoRA (on)      ", on)]:
        o = np.argsort(-s); k = max(1, len(y) // 10)
        print(f"  {tag}: AUC={roc_auc_score(y, s):.3f}  p@top10%={y[o[:k]].mean():.3f}")
    yk, onk, offk = load(KIN_OUT)
    print(f"\nKinetics 良性 ({len(yk)}):")
    print(f"  零样本(off): 误报@0.5 {int((offk>=0.5).sum())}/{len(offk)}  中位 {np.median(offk):.3f}")
    print(f"  QLoRA (on) : 误报@0.5 {int((onk>=0.5).sum())}/{len(onk)}  中位 {np.median(onk):.3f}")


if __name__ == "__main__":
    main()
