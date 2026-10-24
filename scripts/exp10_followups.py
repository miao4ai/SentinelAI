"""第十章后续实验 P0-P2：验证 prefill 瓶颈 / 约束解码 / 校准修正。

三个待回答的问题（见 docs/chapter10_review.md §5-6）：

P0  我们到底是 KV-cache 受限还是 prefill 算力受限？
    vLLM 日志里的 "最大并发 7.17x" 是 29360/4096 的规划数字（假设请求长到 max_model_len），
    而我们的请求只有 ~559 token。若真是 prefill 算力受限，**砍视觉 token 应该近乎线性降延迟**，
    而并发数远能超过 7.17。三个扫描：并发、帧数、CUDA graph。

P1  约束解码能否根治 60% 的格式漂移？
    §10.3 只是"建议"用 guided decoding，从没验证。这里用 JSON Schema 实测解析率。

P2  V3 的 F1 0.827 能靠重新校准救回多少？
    AUC 0.963 说明排序没问题、只是阈值歪（adapter 在 46% 暴力占比上校准，测试切分 37%）。
    按电影分组把测试集切成 calib/eval 两半，在 calib 上拟合温度缩放与等渗回归。
    **并且把所有分数存盘** —— 第十章没存，导致任何阈值分析都要重跑 VLM。

Run: ~/vllm-env/bin/python scripts/exp10_followups.py
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
from PIL import Image

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
ADAPTER = f"{DATA}/qwen_qlora_adapter"
I3D = f"{DATA}/data/i3d_rgb"
DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT = f"{DATA}/exp10_scores.npz"
RESULTS = os.path.expanduser("~/exp10_results.json")

PROMPT = ("You are given {n} frames evenly sampled from one video. Does the video depict "
          "physical violence (fighting, assault, weapons used against people, blood or injury)? "
          "Answer with exactly one word: yes or no.")
SYSTEM = ("你是短视频平台的内容安全审核员，负责判定视频是否包含真实或影视暴力。"
          "违规类型仅限: violence(斗殴/袭击), weapon(持械伤人), gore(血腥伤残), riot(骚乱), "
          "none(不违规)。你只依据画面证据，不臆测画面外的内容。")
COT = ("以上是同一视频均匀采样的 8 帧。请按三步输出：\n"
       "1) 画面描述：一两句话客观描述看到的内容；\n2) 违规判定：是 / 否；\n"
       "3) 违规类型：从 violence/weapon/gore/riot/none 中选一个，并给一句理由。")
SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "is_violating": {"type": "boolean"},
        "category": {"type": "string", "enum": ["violence", "weapon", "gore", "riot", "none"]},
        "reason": {"type": "string"},
    },
    "required": ["description", "is_violating", "category", "reason"],
}
ALLOWED = set(SCHEMA["properties"]["category"]["enum"])

R = {}   # 结果累加器，最后写 JSON


def is_violent(k): return 0 if "_label_A" in k else 1
def label_code(k): return k.split("_label_")[1].split("-")[0] if "_label_" in k else "?"
def log(*a): print(*a, flush=True)


# ─────────────────────────── 共同评测底座（与第十章同一切分） ───────────────────────────
def build_split():
    from sklearn.model_selection import GroupShuffleSplit
    files = sorted(glob.glob(f"{FRAMES}/*.npz"))
    fkeys = [os.path.splitext(os.path.basename(f))[0] for f in files]
    fy = np.array([is_violent(k) for k in fkeys])
    fg = np.array([k.split("__")[0] for k in fkeys])
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(files, fy, fg))
    return fkeys, set(fg[tr]), set(fg[te])


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


def run_v1_v2(K_tr, K_te, y_tr, y_te):
    """V1/V2 分数（放在 vLLM 之前跑，跑完必须释放显存）。"""
    import torch
    from torch import nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sentinelai.early_fusion import JointFusionTransformer

    vis = {}
    for d in DIRS:
        for f in glob.glob(f"{I3D}/{d}/*.npy"):
            a = np.load(f)
            a = a.mean(1) if a.ndim == 3 else a
            vis[os.path.basename(f)[:-4]] = a.astype(np.float32)
    aud_p, aud_s = load_npz_dir(f"{DATA}/audio_full"), load_npz_dir(f"{DATA}/audio_seq", "sequence")
    txt = load_npz_dir(f"{DATA}/text_features")

    def stack(keys):
        return (np.stack([vis[k].mean(0) for k in keys]), np.stack([aud_p[k] for k in keys]),
                np.stack([txt[k] for k in keys]),
                np.stack([resample(vis[k], 32) for k in keys]),
                np.stack([resample(aud_s[k], 16) for k in keys]))

    Vp_tr, Ap_tr, Tp_tr, Vs_tr, As_tr = stack(K_tr)
    Vp_te, Ap_te, Tp_te, Vs_te, As_te = stack(K_te)

    X_tr = np.concatenate([Vp_tr, Ap_tr, Tp_tr], 1)
    X_te = np.concatenate([Vp_te, Ap_te, Tp_te], 1)
    v1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_tr, y_tr)
    p_v1 = v1.predict_proba(X_te)[:, 1]

    dev = "cuda"
    torch.manual_seed(0)
    v2 = JointFusionTransformer({"visual": 2048, "audio": 768, "text": 768},
                                d_model=128, n_layers=2, n_categories=1).to(dev)
    opt = torch.optim.AdamW(v2.parameters(), 1e-3)
    lf = nn.BCEWithLogitsLoss()
    T = {"visual": torch.tensor(Vs_tr, device=dev), "audio": torch.tensor(As_tr, device=dev),
         "text": torch.tensor(Tp_tr[:, None, :], device=dev)}
    E = {"visual": torch.tensor(Vs_te, device=dev), "audio": torch.tensor(As_te, device=dev),
         "text": torch.tensor(Tp_te[:, None, :], device=dev)}
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)[:, None]
    for _ in range(200):
        v2.train(); opt.zero_grad()
        o = v2(T); lo = o[0] if isinstance(o, tuple) else o
        lf(lo, yt).backward(); opt.step()
    v2.eval()
    with torch.no_grad():
        o = v2(E); lo = o[0] if isinstance(o, tuple) else o
        p_v2 = torch.sigmoid(lo).squeeze(1).cpu().numpy()

    del T, E, yt, opt, v2, vis, aud_p, aud_s, txt
    import gc; gc.collect(); torch.cuda.empty_cache()
    return p_v1, p_v2


# ─────────────────────────────────── VLM 工具 ───────────────────────────────────
def make_scorer(llm, proc, n_img):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    prompt = PROMPT.format(n=n_img)

    def frames(key):
        arr = np.load(f"{FRAMES}/{key}.npz", allow_pickle=True)["frames"]
        step = max(len(arr) // n_img, 1)
        return [Image.fromarray(arr[i]) for i in range(0, len(arr), step)][:n_img]

    def req(key):
        msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in range(n_img)],
                                             {"type": "text", "text": prompt}]}]
        return {"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": frames(key)}}

    def p_yes(out):
        lp = out.outputs[0].logprobs[0]
        py, pn = [], []
        for tid, l in lp.items():
            t = proc.tokenizer.decode([tid]).strip().lower()
            if t == "yes": py.append(l.logprob)
            elif t == "no": pn.append(l.logprob)
        ey = sum(np.exp(py)) if py else 0.0
        en = sum(np.exp(pn)) if pn else 0.0
        return ey / (ey + en) if ey + en > 0 else 0.5

    def score(keys, lora=None):
        outs = llm.generate([req(k) for k in keys], sp, lora_request=lora, use_tqdm=False)
        return np.array([p_yes(o) for o in outs])

    return score, req, sp


def metrics(y, p, thr=0.5):
    from sklearn.metrics import (f1_score, fbeta_score, precision_score, recall_score,
                                 roc_auc_score)
    pred = (p >= thr).astype(int)
    return {"P": round(float(precision_score(y, pred, zero_division=0)), 4),
            "R": round(float(recall_score(y, pred)), 4),
            "F1": round(float(f1_score(y, pred)), 4),
            "F2": round(float(fbeta_score(y, pred, beta=2)), 4),
            "AUC": round(float(roc_auc_score(y, p)), 4)}


def main():
    fkeys, train_movies, test_movies = build_split()
    aud = {str(np.load(f, allow_pickle=True)["key"]) for f in glob.glob(f"{DATA}/audio_full/*.npz")}
    txt = {str(np.load(f, allow_pickle=True)["key"]) for f in glob.glob(f"{DATA}/text_features/*.npz")}
    seq = {str(np.load(f, allow_pickle=True)["key"]) for f in glob.glob(f"{DATA}/audio_seq/*.npz")}
    allk = [k for k in fkeys if k in aud and k in txt and k in seq]
    K_tr = [k for k in allk if k.split("__")[0] in train_movies]
    K_te = [k for k in allk if k.split("__")[0] in test_movies]
    y_tr = np.array([is_violent(k) for k in K_tr]); y_te = np.array([is_violent(k) for k in K_te])
    log(f"底座: 训练 {len(K_tr)} / 测试 {len(K_te)} (与第十章一致), 测试暴力占比 {y_te.mean():.0%}")
    R["n_train"], R["n_test"], R["test_pos_rate"] = len(K_tr), len(K_te), float(y_te.mean())

    log("\n=== V1/V2 分数（为 P2 存盘）===")
    p_v1, p_v2 = run_v1_v2(K_tr, K_te, y_tr, y_te)
    log(f"V1 {metrics(y_te, p_v1)}\nV2 {metrics(y_te, p_v2)}")
    R["V1"], R["V2"] = metrics(y_te, p_v1), metrics(y_te, p_v2)

    from transformers import AutoProcessor
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    proc = AutoProcessor.from_pretrained(MODEL)

    # ══════════════ 引擎 A：基线 (enforce_eager=True) —— P0 扫描 + P2 打分 + P1 ══════════════
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096,
              limit_mm_per_prompt={"image": 8}, enforce_eager=True,
              enable_lora=True, max_lora_rank=16)
    log(f"\n引擎A(eager) 加载 {time.time()-t0:.0f}s")
    lora = LoRARequest("violence", 1, ADAPTER)
    score8, req8, sp8 = make_scorer(llm, proc, 8)

    # ── P0-a 并发扫描：若 KV 真的卡在 7.17×，吞吐应在 N≈8 就压平 ──
    log("\n=== P0-a 并发扫描 (提交 N 条，看吞吐) ===")
    conc = []
    pool = K_te[:64]
    for n in (1, 4, 8, 16, 32, 64):
        ks = (pool * ((n // len(pool)) + 1))[:n]
        reqs = [req8(k) for k in ks]                       # 预处理不计时
        t = time.time()
        llm.generate(reqs, sp8, lora_request=lora, use_tqdm=False)
        dt = time.time() - t
        conc.append({"N": n, "total_s": round(dt, 2), "ms_per_clip": round(dt / n * 1e3, 1),
                     "QPS": round(n / dt, 2)})
        log(f"  N={n:>3}  {dt:6.2f}s  {dt/n*1e3:7.1f} ms/clip  {n/dt:6.2f} QPS")
    R["P0_concurrency"] = conc

    # ── P0-b 帧数扫描：prefill 算力受限的话，砍视觉 token 应近乎线性降延迟 ──
    log("\n=== P0-b 帧数扫描 (视觉 token 减半 → 延迟/质量) ===")
    frame_rows = []
    for n_img in (8, 4, 2):
        _, rq_fn, sp_n = make_scorer(llm, proc, n_img)
        rq = [rq_fn(k) for k in K_te]
        t = time.time(); outs = llm.generate(rq, sp_n, lora_request=lora, use_tqdm=False)
        dt = time.time() - t
        # 直接从 outs 复算概率
        ps = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]; py, pn = [], []
            for tid, l in lp.items():
                tk = proc.tokenizer.decode([tid]).strip().lower()
                if tk == "yes": py.append(l.logprob)
                elif tk == "no": pn.append(l.logprob)
            ey = sum(np.exp(py)) if py else 0.0; en = sum(np.exp(pn)) if pn else 0.0
            ps.append(ey / (ey + en) if ey + en > 0 else 0.5)
        ps = np.array(ps)
        m = metrics(y_te, ps)
        row = {"n_img": n_img, "visual_tokens": n_img * 64, "ms_per_clip": round(dt/len(K_te)*1e3, 1), **m}
        frame_rows.append(row)
        log(f"  {n_img} 帧 ({n_img*64} 视觉token): {dt/len(K_te)*1e3:6.1f} ms/clip  "
            f"AUC={m['AUC']:.4f} F1={m['F1']:.4f}")
        if n_img == 8:
            p_v3, outs8 = ps, outs
    R["P0_frames"] = frame_rows

    # ── P2 打分存盘：adapter on 已有，再补 adapter off ──
    log("\n=== P2 adapter off 打分 ===")
    p_zs = score8(K_te)
    log(f"V3(QLoRA) {metrics(y_te, p_v3)}\nV3(零样本) {metrics(y_te, p_zs)}")
    R["V3_qlora"], R["V3_zeroshot"] = metrics(y_te, p_v3), metrics(y_te, p_zs)
    np.savez(OUT, keys=np.array(K_te), y=y_te, codes=np.array([label_code(k) for k in K_te]),
             p_v1=p_v1, p_v2=p_v2, p_v3=p_v3, p_zs=p_zs)
    log(f"分数已存盘 → {OUT}（以后做阈值分析不必再跑 VLM）")

    # ── P1 约束解码 ──
    log("\n=== P1 约束解码 vs 自由生成（60 条）===")
    hk = K_te[:60]
    import re
    from vllm import SamplingParams as SP2

    def cot_req(k):
        arr = np.load(f"{FRAMES}/{k}.npz", allow_pickle=True)["frames"]
        fr = [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr)//8)][:8]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [*[{"type": "image"} for _ in fr],
                                             {"type": "text", "text": COT}]}]
        return {"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": fr}}

    free_txt = [o.outputs[0].text for o in
                llm.generate([cot_req(k) for k in hk], SP2(temperature=0, max_tokens=220), use_tqdm=False)]
    LAB = re.compile(r"违规判定[：:]\s*\**\s*(是|否)")
    INL = re.compile(r"(是|否)\s*[；;，,]\s*([a-zA-Z_]+)")
    n_lab = sum(bool(LAB.search(t)) for t in free_txt)
    n_inl = sum(1 for t in free_txt if not LAB.search(t) and INL.search(t))
    log(f"  自由生成: 三步标签式 {n_lab}/60={n_lab/60:.1%}, 单行压缩式 {n_inl}/60={n_inl/60:.1%}, "
        f"两者都不匹配 {60-n_lab-n_inl}/60={(60-n_lab-n_inl)/60:.1%}")

    # 构造约束解码的 SamplingParams（vLLM 版本间 API 不同，逐个尝试）
    guided_sp, api = None, None
    try:
        from vllm.sampling_params import StructuredOutputsParams
        guided_sp = SP2(temperature=0, max_tokens=220,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA)); api = "StructuredOutputsParams"
    except Exception:
        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided_sp = SP2(temperature=0, max_tokens=220,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA)); api = "GuidedDecodingParams"
        except Exception as e:
            log("  约束解码 API 不可用:", str(e)[:120])

    JSON_INSTR = ("以上是同一视频均匀采样的 8 帧。只输出一个 JSON 对象，字段为："
                  "description(一两句画面描述)、is_violating(布尔)、"
                  "category(violence/weapon/gore/riot/none 之一)、reason(一句画面证据)。")

    def json_req(k):
        arr = np.load(f"{FRAMES}/{k}.npz", allow_pickle=True)["frames"]
        fr = [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr)//8)][:8]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [*[{"type": "image"} for _ in fr],
                                             {"type": "text", "text": JSON_INSTR}]}]
        return {"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": fr}}

    if guided_sp is not None:
        log(f"  使用 {api}")
        jreq = [json_req(k) for k in hk]
        t = time.time()
        jtxt = [o.outputs[0].text for o in llm.generate(jreq, guided_sp, use_tqdm=False)]
        dtj = time.time() - t
        ok, enum_ok, verdicts = 0, 0, []
        for t_, k in zip(jtxt, hk):
            try:
                d = json.loads(t_)
                ok += 1
                if d.get("category") in ALLOWED: enum_ok += 1
                verdicts.append((k, bool(d.get("is_violating"))))
            except Exception:
                verdicts.append((k, None))
        log(f"  约束解码: JSON 可解析 {ok}/60={ok/60:.1%}, category 合法 {enum_ok}/60={enum_ok/60:.1%}, "
            f"{dtj/60*1e3:.0f} ms/clip")
        idx = {k: i for i, k in enumerate(K_te)}
        pairs = [(v, p_zs[idx[k]]) for k, v in verdicts if v is not None]
        dis = sum(1 for v, s in pairs if int(v) != int(s >= 0.5))
        acc = np.mean([int(v) == is_violent(k) for k, v in verdicts if v is not None])
        log(f"  判定 vs logits 不一致 {dis}/{len(pairs)}={dis/max(len(pairs),1):.1%}; "
            f"判定自身准确率 {acc:.3f}")
        R["P1"] = {"api": api, "free_labelled": n_lab/60, "free_inline": n_inl/60,
                   "free_unparseable": (60-n_lab-n_inl)/60, "guided_json_ok": ok/60,
                   "guided_enum_ok": enum_ok/60, "guided_ms_per_clip": round(dtj/60*1e3, 1),
                   "guided_disagree": dis/max(len(pairs), 1), "guided_verdict_acc": float(acc),
                   "guided_sample": jtxt[0][:400]}
        log(f"  样例: {jtxt[0][:200]}")

    del llm
    import gc, torch; gc.collect(); torch.cuda.empty_cache(); time.sleep(5)

    # ══════════════ 引擎 B：CUDA graph (enforce_eager=False) ══════════════
    log("\n=== P0-c CUDA graph (enforce_eager=False) ===")
    try:
        t0 = time.time()
        llm2 = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096,
                   limit_mm_per_prompt={"image": 8}, enforce_eager=False,
                   enable_lora=True, max_lora_rank=16)
        log(f"  引擎B 加载 {time.time()-t0:.0f}s (含 CUDA graph 捕获)")
        lora2 = LoRARequest("violence", 1, ADAPTER)
        sc2, rq2, sp2 = make_scorer(llm2, proc, 8)
        rr = [rq2(k) for k in K_te]
        t = time.time(); llm2.generate(rr, sp2, lora_request=lora2, use_tqdm=False)
        dt = time.time() - t
        log(f"  饱和批: {dt/len(K_te)*1e3:.1f} ms/clip (eager 基线见 P0-b 的 8 帧行)")
        R["P0_cudagraph_ms_per_clip"] = round(dt/len(K_te)*1e3, 1)
        del llm2; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log("  CUDA graph 失败:", str(e)[:200])
        R["P0_cudagraph_error"] = str(e)[:200]

    # ══════════════ P2 校准（纯 CPU，用存好的分数） ══════════════
    log("\n=== P2 校准修正 (按电影分组把测试集切成 calib/eval 两半) ===")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupShuffleSplit
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import f1_score, fbeta_score, roc_auc_score

    g = np.array([k.split("__")[0] for k in K_te])
    ci, ei = next(GroupShuffleSplit(1, test_size=0.5, random_state=1).split(K_te, y_te, g))
    log(f"  calib {len(ci)} / eval {len(ei)} 片段（电影不相交）")
    cal_rows = []
    for name, p in [("V3 QLoRA", p_v3), ("V3 零样本", p_zs)]:
        base = {"F1": f1_score(y_te[ei], (p[ei] >= .5).astype(int)),
                "F2": fbeta_score(y_te[ei], (p[ei] >= .5).astype(int), beta=2)}
        # (a) 在 calib 上搜最优阈值
        ths = np.unique(np.round(p[ci], 4))
        best_th = max(ths, key=lambda t: fbeta_score(y_te[ci], (p[ci] >= t).astype(int), beta=2))
        thr_m = {"F1": f1_score(y_te[ei], (p[ei] >= best_th).astype(int)),
                 "F2": fbeta_score(y_te[ei], (p[ei] >= best_th).astype(int), beta=2)}
        # (b) 温度缩放（在 logit 空间）
        lg = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))
        def nll(T):
            q = 1/(1+np.exp(-lg[ci]/max(T, 1e-3)))
            return -np.mean(y_te[ci]*np.log(q+1e-9) + (1-y_te[ci])*np.log(1-q+1e-9))
        Topt = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        q = 1/(1+np.exp(-lg/Topt))
        temp_m = {"F1": f1_score(y_te[ei], (q[ei] >= .5).astype(int)),
                  "F2": fbeta_score(y_te[ei], (q[ei] >= .5).astype(int), beta=2)}
        # (c) 等渗回归
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[ci], y_te[ci])
        r = iso.predict(p[ei])
        iso_m = {"F1": f1_score(y_te[ei], (r >= .5).astype(int)),
                 "F2": fbeta_score(y_te[ei], (r >= .5).astype(int), beta=2)}
        row = {"model": name, "AUC_eval": round(float(roc_auc_score(y_te[ei], p[ei])), 4),
               "base_F1": round(base["F1"], 4), "base_F2": round(base["F2"], 4),
               "best_thr": round(float(best_th), 4),
               "thr_F1": round(thr_m["F1"], 4), "thr_F2": round(thr_m["F2"], 4),
               "T": round(Topt, 3), "temp_F1": round(temp_m["F1"], 4), "temp_F2": round(temp_m["F2"], 4),
               "iso_F1": round(iso_m["F1"], 4), "iso_F2": round(iso_m["F2"], 4)}
        cal_rows.append(row)
        log(f"  {name}: AUC={row['AUC_eval']}  原始 F1={row['base_F1']}/F2={row['base_F2']}  "
            f"→ 调阈值(th={row['best_thr']}) F1={row['thr_F1']}/F2={row['thr_F2']}  "
            f"→ 温度(T={row['T']}) F1={row['temp_F1']}/F2={row['temp_F2']}  "
            f"→ 等渗 F1={row['iso_F1']}/F2={row['iso_F2']}")
    R["P2_calibration"] = cal_rows

    json.dump(R, open(RESULTS, "w"), ensure_ascii=False, indent=1)
    log(f"\n结果 JSON → {RESULTS}")
    log("EXP10_DONE")


if __name__ == "__main__":
    main()
