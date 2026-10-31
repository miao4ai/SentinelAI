"""第十章收尾 A/B/C：把模型排序做扎实，补完两个没跑成的实验。

A  统计严谨性：295 条单切分分不开 V1/V2（V2 跨种子 AUC 0.985±0.005，单次低至 0.950）。
   这里做三件事——
   A1 V1/V2 在全部 1189 片段上做 5 折电影分组 CV（V2 每折 3 个种子）；
   A2 V2 在固定切分上跑 10 个种子，量化纯种子方差；
   A3 **电影分组配对自助法**：在 295 条上对所有模型两两求差的 95% 置信区间。
   ⚠️ 限制：V3 的 adapter 是在固定 80/20 的训练侧上微调的，做 5 折 CV 需要重训 5 次
   （约 15 小时），本轮不做；V3 只在它合法的 295 条留出集上报，用自助法给置信区间。
   零样本 V3 不需要训练，**可以在全部 1189 条上打分**，给一个更大样本的估计。

B  CUDA graph：上一轮在 gpu_memory_utilization=0.90 下引擎初始化失败，这里降到 0.82 重试。

C  约束解码的判定质量重测：上一轮用的 K_te[:60] 全是正常片段（基率混淆），
   这里换**类别均衡**子样本（30 暴力 + 30 正常）重测判定准确率与 CoT-logits 一致率。

Run: PYTHONPATH=~/documents/SentinelAI ~/vllm-env/bin/python scripts/exp11_rigor.py
"""

from __future__ import annotations

import gc
import glob
import json
import os
import time

import numpy as np

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
ADAPTER = f"{DATA}/qwen_qlora_adapter"
I3D = f"{DATA}/data/i3d_rgb"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT_NPZ = f"{DATA}/exp11_scores.npz"
OUT_JSON = os.path.expanduser("~/exp11_results.json")
N_IMG = 8
R = {}


def iv(k): return 0 if "_label_A" in k else 1
def lc(k): return k.split("_label_")[1].split("-")[0] if "_label_" in k else "?"
def log(*a): print(*a, flush=True)


def load_npz_dir(d, field="embedding"):
    out = {}
    for f in glob.glob(f"{d}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["key"])] = z[field].astype(np.float32)
    return out


def resample(x, n):
    if len(x) == 0:
        return np.zeros((n, x.shape[1]), np.float32)
    return x[np.linspace(0, len(x) - 1, n).round().astype(int)].astype(np.float32)


def metrics(y, p, thr=0.5):
    from sklearn.metrics import f1_score, fbeta_score, roc_auc_score
    pred = (p >= thr).astype(int)
    return (float(roc_auc_score(y, p)), float(f1_score(y, pred)),
            float(fbeta_score(y, pred, beta=2)))


# ═════════════════════════════ A: 统计严谨性 ═════════════════════════════
def experiment_A():
    import torch
    from torch import nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sentinelai.early_fusion import JointFusionTransformer

    files = sorted(glob.glob(f"{FRAMES}/*.npz"))
    fk = [os.path.splitext(os.path.basename(f))[0] for f in files]
    vis = {}
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            a = np.load(f); a = a.mean(1) if a.ndim == 3 else a
            vis[os.path.basename(f)[:-4]] = a.astype(np.float32)
    ap = load_npz_dir(f"{DATA}/audio_full")
    asq = load_npz_dir(f"{DATA}/audio_seq", "sequence")
    tp = load_npz_dir(f"{DATA}/text_features")
    K = [k for k in fk if k in vis and k in ap and k in asq and k in tp]
    y = np.array([iv(k) for k in K]); g = np.array([k.split("__")[0] for k in K])
    log(f"A: 可用片段 {len(K)}（三模态 ∩ 有帧）, {len(set(g))} 部电影, 暴力 {y.mean():.0%}")

    Vp = np.stack([vis[k].mean(0) for k in K]); Ap = np.stack([ap[k] for k in K])
    Tp = np.stack([tp[k] for k in K])
    Vs = np.stack([resample(vis[k], 32) for k in K]); As = np.stack([resample(asq[k], 16) for k in K])
    X = np.concatenate([Vp, Ap, Tp], 1)
    dev = "cuda"

    def fit_v1(tr, te):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
        return m.predict_proba(X[te])[:, 1]

    def fit_v2(tr, te, seed):
        torch.manual_seed(seed)
        m = JointFusionTransformer({"visual": 2048, "audio": 768, "text": 768},
                                   d_model=128, n_layers=2, n_categories=1).to(dev)
        o = torch.optim.AdamW(m.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
        T = {"visual": torch.tensor(Vs[tr], device=dev), "audio": torch.tensor(As[tr], device=dev),
             "text": torch.tensor(Tp[tr][:, None, :], device=dev)}
        E = {"visual": torch.tensor(Vs[te], device=dev), "audio": torch.tensor(As[te], device=dev),
             "text": torch.tensor(Tp[te][:, None, :], device=dev)}
        yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)[:, None]
        for _ in range(200):
            m.train(); o.zero_grad(); out = m(T); lo = out[0] if isinstance(out, tuple) else out
            lf(lo, yt).backward(); o.step()
        m.eval()
        with torch.no_grad():
            out = m(E); lo = out[0] if isinstance(out, tuple) else out
            p = torch.sigmoid(lo).squeeze(1).cpu().numpy()
        del T, E, yt, o, m; gc.collect(); torch.cuda.empty_cache()
        return p

    # ── A1: 5 折 GroupKFold（V2 每折 3 个种子）──
    log("\n=== A1: 5 折电影分组 CV（全部 1189 片段）===")
    cv = list(GroupKFold(n_splits=5).split(X, y, g))
    v1_folds, v2_folds = [], []
    for fi, (tr, te) in enumerate(cv):
        a1 = metrics(y[te], fit_v1(tr, te))
        v1_folds.append(a1)
        for sd in range(3):
            v2_folds.append(metrics(y[te], fit_v2(tr, te, sd)))
        log(f"  fold {fi}: V1 AUC={a1[0]:.4f} F1={a1[1]:.4f} | "
            f"V2 AUC={np.mean([v2_folds[-3+i][0] for i in range(3)]):.4f}")
    v1a = np.array([r[0] for r in v1_folds]); v1f = np.array([r[1] for r in v1_folds])
    v2a = np.array([r[0] for r in v2_folds]); v2f = np.array([r[1] for r in v2_folds])
    log(f"  V1 (5 折):        AUC {v1a.mean():.4f} ± {v1a.std():.4f}   F1 {v1f.mean():.4f} ± {v1f.std():.4f}")
    log(f"  V2 (5 折×3 种子): AUC {v2a.mean():.4f} ± {v2a.std():.4f}   F1 {v2f.mean():.4f} ± {v2f.std():.4f}")
    R["A1_cv"] = {"V1_AUC": [round(v1a.mean(), 4), round(v1a.std(), 4)],
                  "V1_F1": [round(v1f.mean(), 4), round(v1f.std(), 4)],
                  "V2_AUC": [round(v2a.mean(), 4), round(v2a.std(), 4)],
                  "V2_F1": [round(v2f.mean(), 4), round(v2f.std(), 4)], "n": len(K)}

    # ── A2: 固定切分上 V2 跑 10 个种子 + V1，并把分数留给 A3 ──
    log("\n=== A2: 固定 80/20 切分（与第十章一致），V2 跑 10 个种子 ===")
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, g))
    K_te = [K[i] for i in te]; y_te = y[te]
    p_v1 = fit_v1(tr, te)
    v2_scores = np.stack([fit_v2(tr, te, sd) for sd in range(10)])
    v2m = np.array([metrics(y_te, v2_scores[i]) for i in range(10)])
    log(f"  V1: AUC={metrics(y_te,p_v1)[0]:.4f} F1={metrics(y_te,p_v1)[1]:.4f} (确定性)")
    log(f"  V2 跨 10 种子: AUC {v2m[:,0].mean():.4f} ± {v2m[:,0].std():.4f} "
        f"(min {v2m[:,0].min():.4f} max {v2m[:,0].max():.4f})")
    log(f"              F1  {v2m[:,1].mean():.4f} ± {v2m[:,1].std():.4f}")
    R["A2_seeds"] = {"V1_AUC": round(float(metrics(y_te, p_v1)[0]), 4),
                     "V2_AUC_mean": round(float(v2m[:, 0].mean()), 4),
                     "V2_AUC_std": round(float(v2m[:, 0].std()), 4),
                     "V2_AUC_min": round(float(v2m[:, 0].min()), 4),
                     "V2_AUC_max": round(float(v2m[:, 0].max()), 4),
                     "V2_F1_mean": round(float(v2m[:, 1].mean()), 4),
                     "V2_F1_std": round(float(v2m[:, 1].std()), 4)}
    del vis, ap, asq, tp; gc.collect(); torch.cuda.empty_cache()
    return K, y, g, K_te, y_te, p_v1, v2_scores.mean(0), np.array(te)


# ═════════════════════════════ B+C+打分 ═════════════════════════════
def experiment_BC(K_all, y_all, K_te, y_te):
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    proc = AutoProcessor.from_pretrained(MODEL)
    PROMPT = ("You are given 8 frames evenly sampled from one video. Does the video depict "
              "physical violence (fighting, assault, weapons used against people, blood or injury)? "
              "Answer with exactly one word: yes or no.")

    def frames(k):
        arr = np.load(f"{FRAMES}/{k}.npz", allow_pickle=True)["frames"]
        return [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr) // N_IMG)][:N_IMG]

    def req(k, text=PROMPT, system=None):
        content = [*[{"type": "image"} for _ in range(N_IMG)], {"type": "text", "text": text}]
        msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
               [{"role": "user", "content": content}]
        return {"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": frames(k)}}

    def p_yes(o):
        lp = o.outputs[0].logprobs[0]; py, pn = [], []
        for tid, l in lp.items():
            t = proc.tokenizer.decode([tid]).strip().lower()
            if t == "yes": py.append(l.logprob)
            elif t == "no": pn.append(l.logprob)
        ey = sum(np.exp(py)) if py else 0.0; en = sum(np.exp(pn)) if pn else 0.0
        return ey / (ey + en) if ey + en > 0 else 0.5

    # ── B: CUDA graph 重试（降显存到 0.82）──
    log("\n=== B: CUDA graph 重试 (gpu_memory_utilization=0.82, enforce_eager=False) ===")
    llm, eager = None, None
    for ge, gm in ((False, 0.82), (True, 0.90)):
        try:
            t0 = time.time()
            llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=gm, max_model_len=4096,
                      limit_mm_per_prompt={"image": N_IMG}, enforce_eager=ge,
                      enable_lora=True, max_lora_rank=16)
            eager = ge
            log(f"  引擎加载成功: enforce_eager={ge}, gpu_mem_util={gm}, 用时 {time.time()-t0:.0f}s")
            break
        except Exception as e:
            log(f"  enforce_eager={ge} @ {gm} 失败: {str(e)[:150]}")
            gc.collect()
    if llm is None:
        R["B_cuda_graph"] = {"status": "both configs failed"}
        return None
    R["B_cuda_graph"] = {"enforce_eager": eager, "status": "ok" if eager is False else "fell back to eager"}

    lora = LoRARequest("violence", 1, ADAPTER)
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)

    # 同口径延迟（与 exp10 的 200.3 ms/clip eager 基线对比）
    rr = [req(k) for k in K_te]
    t = time.time(); outs = llm.generate(rr, sp, lora_request=lora, use_tqdm=False)
    dt = (time.time() - t) / len(K_te) * 1e3
    p_v3 = np.array([p_yes(o) for o in outs])
    log(f"  饱和批延迟 {dt:.1f} ms/clip （exp10 eager 基线 200.3 ms/clip）")
    R["B_cuda_graph"]["ms_per_clip"] = round(dt, 1)
    R["B_cuda_graph"]["eager_baseline_ms"] = 200.3

    # ── 零样本在全部 1189 条上打分（不需训练，样本更大）──
    log("\n=== 零样本 V3 在全部 1189 片段上打分（无需训练，故无泄漏）===")
    t = time.time()
    outs_all = llm.generate([req(k) for k in K_all], sp, use_tqdm=False)
    p_zs_all = np.array([p_yes(o) for o in outs_all])
    a, f1, f2 = metrics(y_all, p_zs_all)
    log(f"  {len(K_all)} 条: AUC={a:.4f} F1={f1:.4f} F2={f2:.4f}  ({(time.time()-t)/len(K_all)*1e3:.0f} ms/clip)")
    R["zeroshot_full"] = {"n": len(K_all), "AUC": round(a, 4), "F1": round(f1, 4), "F2": round(f2, 4)}
    idx = {k: i for i, k in enumerate(K_all)}
    p_zs_te = np.array([p_zs_all[idx[k]] for k in K_te])

    # ── C: 类别均衡子样本重测约束解码 ──
    log("\n=== C: 约束解码判定质量（类别均衡 30 暴力 + 30 正常）===")
    rng = np.random.default_rng(0)
    pos = [k for k in K_te if iv(k) == 1]; neg = [k for k in K_te if iv(k) == 0]
    bal = list(rng.permutation(pos)[:30]) + list(rng.permutation(neg)[:30])
    y_bal = np.array([iv(k) for k in bal])
    log(f"  子样本 {len(bal)} 条, 暴力 {y_bal.mean():.0%}（上一轮是 0%，故那些数字作废）")

    SYSTEM = ("你是短视频平台的内容安全审核员，负责判定视频是否包含真实或影视暴力。"
              "违规类型仅限: violence(斗殴/袭击), weapon(持械伤人), gore(血腥伤残), riot(骚乱), "
              "none(不违规)。你只依据画面证据，不臆测画面外的内容。")
    JSON_INSTR = ("以上是同一视频均匀采样的 8 帧。只输出一个 JSON 对象，字段为："
                  "description(一两句画面描述)、is_violating(布尔)、"
                  "category(violence/weapon/gore/riot/none 之一)、reason(一句画面证据)。")
    SCHEMA = {"type": "object", "properties": {
        "description": {"type": "string"}, "is_violating": {"type": "boolean"},
        "category": {"type": "string", "enum": ["violence", "weapon", "gore", "riot", "none"]},
        "reason": {"type": "string"}}, "required": ["description", "is_violating", "category", "reason"]}
    from vllm.sampling_params import StructuredOutputsParams
    gsp = SamplingParams(temperature=0, max_tokens=220,
                         structured_outputs=StructuredOutputsParams(json=SCHEMA))
    jt = [o.outputs[0].text for o in
          llm.generate([req(k, JSON_INSTR, SYSTEM) for k in bal], gsp, use_tqdm=False)]
    ok, verd, cats = 0, [], []
    for t_ in jt:
        try:
            d = json.loads(t_); ok += 1
            verd.append(1 if d.get("is_violating") else 0); cats.append(d.get("category"))
        except Exception:
            verd.append(None); cats.append(None)
    valid = [(v, yb, k) for v, yb, k in zip(verd, y_bal, bal) if v is not None]
    acc = float(np.mean([v == yb for v, yb, _ in valid]))
    rec = float(np.mean([v == 1 for v, yb, _ in valid if yb == 1]))
    spec = float(np.mean([v == 0 for v, yb, _ in valid if yb == 0]))
    zidx = {k: i for i, k in enumerate(K_te)}
    dis = sum(1 for v, _, k in valid if v != int(p_zs_te[zidx[k]] >= 0.5))
    log(f"  JSON 可解析 {ok}/{len(bal)}={ok/len(bal):.0%}; category 合法 "
        f"{sum(1 for c in cats if c in SCHEMA['properties']['category']['enum'])}/{len(bal)}")
    log(f"  判定准确率 {acc:.3f}（召回 {rec:.3f} / 特异度 {spec:.3f}）")
    log(f"  判定 vs logits 不一致 {dis}/{len(valid)}={dis/max(len(valid),1):.1%}")
    R["C_balanced"] = {"n": len(bal), "pos_rate": float(y_bal.mean()), "json_ok": ok / len(bal),
                       "verdict_acc": round(acc, 4), "verdict_recall": round(rec, 4),
                       "verdict_specificity": round(spec, 4),
                       "disagree_with_logits": round(dis / max(len(valid), 1), 4)}
    del llm; gc.collect()
    return p_v3, p_zs_te, p_zs_all


# ═════════════════════════════ A3: 配对自助法 ═════════════════════════════
def experiment_A3(K_te, y_te, scores: dict, n_boot=2000):
    from sklearn.metrics import roc_auc_score, fbeta_score
    log(f"\n=== A3: 电影分组配对自助法（{n_boot} 次重采样，95% CI）===")
    movies = np.array([k.split("__")[0] for k in K_te])
    uniq = np.unique(movies)
    by_movie = {m: np.where(movies == m)[0] for m in uniq}
    rng = np.random.default_rng(0)
    names = list(scores)
    boot = {n: [] for n in names}
    diffs = {f"{a}-{b}": [] for i, a in enumerate(names) for b in names[i+1:]}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_movie[m] for m in pick])
        if len(np.unique(y_te[idx])) < 2:
            continue
        cur = {}
        for n in names:
            cur[n] = roc_auc_score(y_te[idx], scores[n][idx]); boot[n].append(cur[n])
        for i, a in enumerate(names):
            for b in names[i+1:]:
                diffs[f"{a}-{b}"].append(cur[a] - cur[b])
    log(f"{'模型':<10}{'AUC':>9}{'95% CI':>22}")
    for n in names:
        v = np.array(boot[n])
        log(f"{n:<10}{roc_auc_score(y_te, scores[n]):>9.4f}"
            f"{f'[{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]':>22}")
    log(f"\n{'两两 AUC 差':<14}{'均值':>9}{'95% CI':>22}  显著?")
    sig = {}
    for k, v in diffs.items():
        v = np.array(v); lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
        s = "是" if (lo > 0 or hi < 0) else "**否（差距在噪声内）**"
        sig[k] = {"mean": round(float(v.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
                  "significant": bool(lo > 0 or hi < 0)}
        log(f"{k:<14}{v.mean():>9.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}  {s}")
    R["A3_bootstrap"] = {"auc_ci": {n: [round(float(np.percentile(boot[n], 2.5)), 4),
                                        round(float(np.percentile(boot[n], 97.5)), 4)] for n in names},
                         "pairwise": sig}


def main():
    K_all, y_all, g_all, K_te, y_te, p_v1, p_v2, te_idx = experiment_A()
    out = experiment_BC(K_all, y_all, K_te, y_te)
    if out is None:
        json.dump(R, open(OUT_JSON, "w"), ensure_ascii=False, indent=1); return
    p_v3, p_zs_te, p_zs_all = out
    experiment_A3(K_te, y_te, {"V1": p_v1, "V2": p_v2, "V3": p_v3, "V3-zs": p_zs_te})
    np.savez(OUT_NPZ, keys_te=np.array(K_te), y_te=y_te, p_v1=p_v1, p_v2=p_v2,
             p_v3=p_v3, p_zs_te=p_zs_te, keys_all=np.array(K_all), y_all=y_all, p_zs_all=p_zs_all)
    json.dump(R, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    log(f"\n分数 → {OUT_NPZ}\n结果 → {OUT_JSON}\nEXP11_DONE")


if __name__ == "__main__":
    main()
