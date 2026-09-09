"""V3 step 2 — QLoRA fine-tune Qwen2.5-VL-7B on XD violence (close 0.889 -> 0.951?).

Zero-shot Qwen already ranks at AUC 0.889 (§4.9). This fine-tunes it on the task:
instruction pairs built from the cached frames16 clips —

    8 frames + "does the video depict physical violence? answer yes or no" -> "yes"/"no"

4-bit NF4 base + LoRA (r=16, LLM attn+MLP only, vision tower frozen), loss on the
answer tokens only, movie-grouped 80/20 split. Evaluation scores every held-out
test clip TWICE — adapter ON (QLoRA) and adapter OFF (zero-shot) — so the
comparison is exactly paired. Saves the adapter + per-clip scores.

Run on the GPU box:
    python scripts/finetune_qwen_qlora.py [--epochs 1 --accum 8 --lr 1e-4 --eval-n 400]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
ADAPTER_OUT = f"{DATA}/qwen_qlora_adapter"
SCORES_OUT = f"{DATA}/qwen_qlora_scores.npz"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMG = 8
PROMPT = ("You are given {n} frames evenly sampled from one video. Does the video depict "
          "physical violence (fighting, assault, weapons used against people, blood or injury)? "
          "Answer with exactly one word: yes or no.")


def is_violent(n): return 0 if "_label_A" in n else 1


def load_frames(path):
    from PIL import Image
    arr = np.load(path, allow_pickle=True)["frames"]
    return [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr) // N_IMG)][:N_IMG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap train samples")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig)

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, quantization_config=bnb,
                                                        device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    def chat_prompt(imgs):
        msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in imgs],
                                             {"type": "text", "text": PROMPT.format(n=len(imgs))}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def train_batch(imgs, answer):
        """(inputs, labels): LM loss on the answer tokens only."""
        ptext = chat_prompt(imgs)
        enc_p = proc(text=[ptext], images=imgs, return_tensors="pt")
        enc = proc(text=[ptext + answer + "<|im_end|>"], images=imgs, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[:, : enc_p["input_ids"].shape[1]] = -100
        return enc.to(model.device), labels.to(model.device)

    yes_ids = sorted({proc.tokenizer.encode(v, add_special_tokens=False)[0]
                      for v in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({proc.tokenizer.encode(v, add_special_tokens=False)[0]
                     for v in ("no", "No", " no", " No")})

    def p_yes(imgs):
        enc = proc(text=[chat_prompt(imgs)], images=imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        logits = out.scores[0][0].float()
        ly = torch.logsumexp(logits[yes_ids], 0)
        ln = torch.logsumexp(logits[no_ids], 0)
        return torch.softmax(torch.stack([ly, ln]), 0)[0].item()

    # ── movie-grouped split over all cached-frame clips ──
    files = sorted(glob.glob(f"{FRAMES}/*.npz"))
    keys = [os.path.splitext(os.path.basename(f))[0] for f in files]
    y = np.array([is_violent(k) for k in keys])
    groups = np.array([k.split("__")[0] for k in keys])
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(files, y, groups))
    print(f"{len(files)} clips ({y.mean():.0%} violent): train {len(tr)} / test {len(te)} (movie-grouped)")

    # ── training ──
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    for ep in range(args.epochs):
        order = rng.permutation(tr)
        if args.limit:
            order = order[: args.limit]
        running = []
        for j, i in enumerate(order):
            imgs = load_frames(files[i])
            enc, labels = train_batch(imgs, "yes" if y[i] else "no")
            loss = model(**enc, labels=labels).loss / args.accum
            loss.backward()
            running.append(loss.item() * args.accum)
            if (j + 1) % args.accum == 0:
                opt.step(); opt.zero_grad(); step += 1
            if (j + 1) % 200 == 0:
                print(f"  ep{ep} {j+1}/{len(order)}  loss={np.mean(running[-200:]):.4f}", flush=True)
        print(f"epoch {ep} done  mean_loss={np.mean(running):.4f}", flush=True)
    os.makedirs(ADAPTER_OUT, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT)
    print(f"adapter saved -> {ADAPTER_OUT}")

    # ── paired evaluation: adapter ON vs OFF on the same held-out clips ──
    model.eval()
    ev = te if args.eval_n == 0 else rng.permutation(te)[: args.eval_n]
    ys, s_ft, s_zs, kk = [], [], [], []
    for c, i in enumerate(ev):
        try:
            imgs = load_frames(files[i])
            s1 = p_yes(imgs)
            with model.disable_adapter():
                s0 = p_yes(imgs)
            ys.append(int(y[i])); s_ft.append(s1); s_zs.append(s0); kk.append(keys[i])
            if (c + 1) % 50 == 0:
                print(f"  eval {c+1}/{len(ev)}", flush=True)
        except Exception as e:
            print(f"  eval skip {keys[i][:36]}: {e}")
    ys, s_ft, s_zs = np.array(ys), np.array(s_ft), np.array(s_zs)
    np.savez(SCORES_OUT, keys=np.array(kk), y=ys, s_ft=s_ft, s_zs=s_zs)

    print(f"\n=== 同一留出集成对对比 ({len(ys)} clips, {ys.mean():.0%} violent) ===")
    for tag, s in [("零样本 (adapter off)", s_zs), ("QLoRA  (adapter on) ", s_ft)]:
        o = np.argsort(-s); k = max(1, len(ys) // 10)
        print(f"  {tag}: AUC={roc_auc_score(ys, s):.3f}  F1@0.5={f1_score(ys, (s >= 0.5).astype(int)):.3f}"
              f"  p@top10%={ys[o[:k]].mean():.3f}")


if __name__ == "__main__":
    main()
