# %% [markdown]
# # 第九章 — 纯本地高性能推理引擎与审核流水线
#
# 两个问题：**9.1** 用 vLLM 承载微调后的 Qwen2.5-VL 权重，开启 PagedAttention 与
# Continuous Batching，最大化 GPU 利用率；**9.2** 混合架构路由分发——分级审核流水线
# （Cascade）：简单内容走 V1/V2 小模型（Triton 形态部署），小模型置信度落在
# **[0.4, 0.6]** 的边界 hard case 路由给 V3 Qwen-VL 深度推理。
#
# 规矩不变：**问题/原理/结论在 markdown，代码在 cell**，输出为 L4 24GB 上的真实运行结果。

# %% [markdown]
# ## 9.1 VLM 高并发加速：vLLM 承载 QLoRA 权重
#
# **为什么选 vLLM（而不是裸 transformers）**：
#
# 1. **PagedAttention**：把 KV cache 切成固定大小的 block（默认 16 token/块），像操作系统
#    的虚拟内存分页一样按需分配——消除了预分配 `max_len` 连续显存造成的碎片和浪费
#    （传统方式 60-80% 的 KV 显存是浪费的），同样的卡能塞下大得多的有效 batch。
# 2. **Continuous Batching**：调度粒度是"一步生成"而不是"一个 batch"——新请求随时插进
#    正在跑的 batch，先生成完的请求立刻退出让位。GPU 不再等最慢的样本，吞吐可以贴着
#    算力上限跑。审核流量恰好是"大量短判定"（我们只要 1 个 yes/no token），最吃这两项。
# 3. **LoRA 原生支持**：`enable_lora=True` 直接挂 8.2 产出的 40MB adapter（LLM-only 正是
#    vLLM 对 VL 模型 LoRA 的支持范围），不需要合并权重；一个底座可同时服务多个 adapter。
#
# **两种部署形态**：本 notebook 用**离线引擎**（`vllm.LLM`，进程内、便于测量）；生产上换
# **服务形态**即可，OpenAI 兼容接口：
#
# ```bash
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 4096 \
#   --limit-mm-per-prompt image=8 --gpu-memory-utilization 0.92 \
#   --enable-lora --max-lora-rank 16 \
#   --lora-modules violence=$HOME/documents/SentinelAI/data/xd-violence/qwen_qlora_adapter
# # 客户端: POST /v1/chat/completions, "model": "violence"（挂 adapter）或底座名（零样本）
# ```
#
# 下面的引擎参数注释标明了 PagedAttention / Continuous Batching 分别由什么控制。

# %%
import glob, json, os, time
import numpy as np
from PIL import Image

DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
ADAPTER = f"{DATA}/qwen_qlora_adapter"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMG = 8

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import vllm
print("vLLM", vllm.__version__)

llm = LLM(
    model=MODEL, dtype="bfloat16",
    gpu_memory_utilization=0.92,   # 权重之外的显存全部交给 PagedAttention 的 KV block 池
    max_model_len=4096,            # 8 帧×~64 视觉 token + 提示词, 4k 足够
    limit_mm_per_prompt={"image": N_IMG},
    enable_lora=True, max_lora_rank=16,
    enforce_eager=True,            # L4 上省掉 CUDA graph 的显存开销
)
lora = LoRARequest("violence", 1, ADAPTER)
print("engine up, adapter =", os.path.basename(ADAPTER))

# %%
# 打分函数：强制 yes/no, 取首 token 的 top-20 logprobs 归一化成 P(violent) —— 与 §8.2 同协议
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(MODEL)
PROMPT = ("You are given 8 frames evenly sampled from one video. Does the video depict "
          "physical violence (fighting, assault, weapons used against people, blood or injury)? "
          "Answer with exactly one word: yes or no.")
SP = SamplingParams(temperature=0, max_tokens=1, logprobs=20)

def safe(k): return k.replace("/", "_")
def is_violent(k): return 0 if "_label_A" in k else 1

def load_frames(key):
    arr = np.load(f"{FRAMES}/{safe(key)}.npz", allow_pickle=True)["frames"]
    return [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr) // N_IMG)][:N_IMG]

def make_req(key):
    msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in range(N_IMG)],
                                         {"type": "text", "text": PROMPT}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return {"prompt": txt, "multi_modal_data": {"image": load_frames(key)}}

def p_from_logprobs(out):
    lp = out.outputs[0].logprobs[0]                     # {token_id: Logprob} top-20
    py, pn = [], []
    for tid, l in lp.items():
        t = proc.tokenizer.decode([tid]).strip().lower()
        if t == "yes": py.append(l.logprob)
        elif t == "no": pn.append(l.logprob)
    ey = sum(np.exp(py)) if py else 0.0
    en = sum(np.exp(pn)) if pn else 0.0
    return ey / (ey + en) if ey + en > 0 else 0.5

def vlm_score(keys, lora_req=lora):
    outs = llm.generate([make_req(k) for k in keys], SP, lora_request=lora_req)
    return np.array([p_from_logprobs(o) for o in outs])

# %% [markdown]
# **Continuous Batching 的效果实测**：同一批片段，先一个一个提交（batch=1，模拟裸推理的
# 逐条串行），再整批一次性丢给引擎（调度器自动连续批处理）。

# %%
import random
random.seed(0)
pool = [os.path.basename(f)[:-4] for f in glob.glob(f"{FRAMES}/*.npz")]
demo = random.sample(pool, 24)

t0 = time.time()
for k in demo[:4]:
    vlm_score([k])                      # batch=1: 每条独占一次完整前向调度
t_seq = (time.time() - t0) / 4

t0 = time.time()
ps = vlm_score(demo)                    # 一次提交 24 条: continuous batching 接管
t_bat = (time.time() - t0) / len(demo)

print(f"batch=1 串行:        {t_seq*1000:7.0f} ms/clip")
print(f"24 条连续批处理:      {t_bat*1000:7.0f} ms/clip   加速 {t_seq/t_bat:.1f}x")
print(f"样本得分范围 [{ps.min():.2f}, {ps.max():.2f}], 暴力占比(阈值0.5): {(ps>=.5).mean():.0%}")

# %% [markdown]
# **Profiling：加速到底从哪来？** 两个测量：
#
# 1. **batch-size 扫描**——每 clip 延迟随并发数的摊薄曲线。单请求时前向以"细长" GEMM
#    为主，算术强度低、显存带宽是瓶颈，SM 大量空转；并发把同层的小 GEMM 合成大 GEMM，
#    算术强度上升，直到吃满算力（曲线变平的位置就是 compute-bound 拐点）。
# 2. **GPU 利用率时间线**——用 NVML 以 100ms 间隔采样 SM 利用率，分别覆盖 batch=1 串行段
#    与整批段。串行段的锯齿与空隙（请求间的 Python 调度、CPU 端图像预处理、未饱和的
#    前向）正是 continuous batching 抹掉的浪费；整批段应接近满载的平顶。
#
# 注意我们的负载只生成 **1 个 token**（yes/no），耗时几乎全在 **prefill**（8 帧视觉编码 +
# 提示词前向），所以此处收益主要来自 prefill 的批内并行与调度开销消除；若是长文本生成
# （如 7.3 的 CoT 解释），continuous batching 在 decode 阶段的"先完成先退出"会再叠加一层收益。

# %%
import threading
import matplotlib.pyplot as plt
import pynvml
pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)

class GpuTrace:
    """后台线程每 100ms 采一次 SM 利用率。"""
    def __init__(self): self.t, self.u, self._stop = [], [], False
    def _run(self, t0):
        while not self._stop:
            self.u.append(pynvml.nvmlDeviceGetUtilizationRates(_h).gpu)
            self.t.append(time.time() - t0); time.sleep(0.1)
    def __enter__(self):
        self._th = threading.Thread(target=self._run, args=(time.time(),), daemon=True)
        self._th.start(); return self
    def __exit__(self, *a): self._stop = True; self._th.join()

sizes = [1, 2, 4, 8, 16, 24]
lat = []
for bs in sizes:
    t0 = time.time(); vlm_score(demo[:bs]); lat.append((time.time() - t0) / bs)

with GpuTrace() as tr_seq:                       # 串行段: 4 条逐一提交
    for k in demo[:4]: vlm_score([k])
with GpuTrace() as tr_bat:                       # 整批段: 24 条一次提交
    vlm_score(demo)

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].plot(sizes, [l * 1000 for l in lat], "o-")
ax[0].set(xscale="log", xlabel="batch size", ylabel="ms / clip",
          title="per-clip latency vs batch size (amortization)")
ax[0].set_xticks(sizes); ax[0].set_xticklabels(sizes)
for t, u, lab in [(tr_seq.t, tr_seq.u, f"batch=1 serial (mean {np.mean(tr_seq.u):.0f}%)"),
                  (tr_bat.t, tr_bat.u, f"batched x24 (mean {np.mean(tr_bat.u):.0f}%)")]:
    ax[1].plot(t, u, label=lab)
ax[1].set(xlabel="sec", ylabel="GPU util %", ylim=(0, 105),
          title="SM utilization timeline"); ax[1].legend(loc="lower right")
plt.tight_layout(); plt.show()
print(f"平均 SM 利用率: 串行 {np.mean(tr_seq.u):.0f}% → 整批 {np.mean(tr_bat.u):.0f}%")
print(f"每 clip 延迟: batch=1 {lat[0]*1000:.0f} ms → batch={sizes[-1]} {lat[-1]*1000:.0f} ms "
      f"({lat[0]/lat[-1]:.1f}x)")

# %% [markdown]
# ## 9.2 混合架构路由分发：分级审核流水线 (Cascade)
#
# ```
#          全部流量 (100%)
#               │
#      ┌────────▼─────────┐   V1/V2 小模型: 3 模态特征 + ③concat-LR
#      │  低耗流 (Triton)  │   CPU 上亚毫秒级, 千级并发
#      └────────┬─────────┘
#               │ p_small
#     ┌─────────┼──────────────┐
#     │p<0.4    │0.4≤p≤0.6     │p>0.6
#     ▼         ▼              ▼
#   放行     ┌──────────┐    判违规
#            │ 高耗流    │    V3 Qwen-VL (vLLM), 深度推理
#            │ hard case │    只吃边界流量
#            └──────────┘
# ```
#
# **低耗流的部署形态（Triton Inference Server）**：小模型是 sklearn 的 LR——导出 ONNX 后
# 由 Triton 的 onnxruntime backend 承载，`dynamic_batching` 把零散请求在 500µs 窗口内
# 自动攒批。模型仓库布局与配置：
#
# ```
# model_repository/fusion_concat_lr/
# ├── config.pbtxt
# └── 1/model.onnx
# ```
# ```protobuf
# # config.pbtxt
# name: "fusion_concat_lr"
# backend: "onnxruntime"
# max_batch_size: 256
# input  [ { name: "features",      data_type: TYPE_FP32, dims: [ 3584 ] } ]
# output [ { name: "probabilities", data_type: TYPE_FP32, dims: [ 2 ] } ]
# dynamic_batching { max_queue_delay_microseconds: 500 }
# instance_group [ { count: 2, kind: KIND_CPU } ]
# ```
# ```bash
# docker run --rm -p8000:8000 -v $PWD/model_repository:/models \
#   nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
# ```
# 本 notebook 里我们训练同一个小模型、导出同一个 model.onnx，并用 onnxruntime
# 本地实测它的延迟（即 Triton backend 内实际执行的东西）。

# %%
# ── V1/V2 小模型: 电影分组 80/20, ③ feature-concat LR ──
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

def load_npz_dir(d):
    out = {}
    for f in glob.glob(f"{d}/*.npz"):
        z = np.load(f, allow_pickle=True); out[str(z["key"])] = z["embedding"].astype(np.float32)
    return out

I3D_DIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954", "test_videos"]
vis = {}
for d in I3D_DIRS:
    for f in glob.glob(f"{DATA}/data/i3d_rgb/{d}/*.npy"):
        a = np.load(f); a = a.mean(1) if a.ndim == 3 else a
        vis[os.path.basename(f)[:-4]] = a.astype(np.float32)
aud, txt = load_npz_dir(f"{DATA}/audio_full"), load_npz_dir(f"{DATA}/text_features")
keys = sorted(k for k in vis if k in aud and k in txt)
y = np.array([is_violent(k) for k in keys])
groups = np.array([k.split("__")[0] for k in keys])
X = np.concatenate([np.stack([vis[k].mean(0) for k in keys]),
                    np.stack([aud[k] for k in keys]),
                    np.stack([txt[k] for k in keys])], 1)
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, groups))
small = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
p_small = small.predict_proba(X[te])[:, 1]
yte = y[te]; keys_te = [keys[i] for i in te]
print(f"{len(keys)} 片段, 训练 {len(tr)} / 测试 {len(te)} (电影不相交)")
print(f"小模型单独: AUC={roc_auc_score(yte, p_small):.3f}  F1@0.5={f1_score(yte, (p_small>=.5)):.3f}")

# %%
# ── 置信度分布与 [0.4, 0.6] 路由带 ──
import matplotlib.pyplot as plt
LO, HI = 0.4, 0.6
band = (p_small >= LO) & (p_small <= HI)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(p_small[yte == 0], bins=50, alpha=0.6, label="normal")
ax.hist(p_small[yte == 1], bins=50, alpha=0.6, label="violent")
ax.axvspan(LO, HI, color="red", alpha=0.12, label=f"route to VLM ({band.mean():.1%})")
ax.set(xlabel="small-model confidence p", ylabel="#clips", title="cascade routing band")
ax.legend(); plt.tight_layout(); plt.show()
acc_band_small = ((p_small[band] >= .5) == yte[band]).mean() if band.any() else float("nan")
print(f"路由带内 {band.sum()} 条 ({band.mean():.1%} 流量), 小模型在带内的准确率仅 {acc_band_small:.2f}")

# %%
# ── 低耗流延迟: 导出 ONNX 并用 onnxruntime 实测 (即 Triton 内实际执行的模型) ──
from skl2onnx import to_onnx
import onnxruntime as ort
onx = to_onnx(small, X[:1].astype(np.float32),
              options={id(small.steps[-1][1]): {"zipmap": False}})
open(f"{DATA}/fusion_concat_lr.onnx", "wb").write(onx.SerializeToString())
sess = ort.InferenceSession(f"{DATA}/fusion_concat_lr.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
for bs in (1, 256):
    xb = X[te][:bs].astype(np.float32)
    sess.run(None, {iname: xb})                                   # warmup
    t0 = time.time(); n = 200
    for _ in range(n): sess.run(None, {iname: xb})
    dt = (time.time() - t0) / n
    print(f"onnxruntime CPU  batch={bs:>3}: {dt*1e3:6.2f} ms/批 = {dt/bs*1e6:7.1f} µs/clip")

# %%
# ── 高耗流: 路由带内的 hard case 交给 vLLM + QLoRA adapter 深度推理 ──
band_keys = [k for k, b in zip(keys_te, band) if b]
band_have = [k for k in band_keys if os.path.exists(f"{FRAMES}/{safe(k)}.npz")]
print(f"路由带 {len(band_keys)} 条, 其中 {len(band_have)} 条有 frames16 缓存, 送 VLM")
t0 = time.time()
p_vlm = vlm_score(band_have)
t_deep = (time.time() - t0) / max(len(band_have), 1)
print(f"VLM 深度推理: {t_deep*1000:.0f} ms/clip (continuous batching)")

# 级联决策: 带外 = 小模型原判; 带内有帧 = VLM 判; 带内无帧 = 退回小模型
p_cas = p_small.copy()
vmap_band = dict(zip(band_have, p_vlm))
for i, k in enumerate(keys_te):
    if k in vmap_band: p_cas[i] = vmap_band[k]

import pandas as pd
rows = {}
for name, p in [("V1/V2 小模型单独", p_small), ("级联 (band→VLM)", p_cas)]:
    pred = (p >= .5).astype(int)
    rows[name] = {"AUC": roc_auc_score(yte, p), "F1@0.5": f1_score(yte, pred),
                  "错误数": int((pred != yte).sum()),
                  "VLM 调用占比": f"{len(band_have)/len(yte):.1%}" if "级联" in name else "0%"}
tbl = pd.DataFrame(rows).T
band_idx = [i for i, k in enumerate(keys_te) if k in vmap_band]
fixed = sum(((p_small[i] >= .5) != yte[i]) and ((p_cas[i] >= .5) == yte[i]) for i in band_idx)
broke = sum(((p_small[i] >= .5) == yte[i]) and ((p_cas[i] >= .5) != yte[i]) for i in band_idx)
print(tbl.round(3))
print(f"\n带内 {len(band_idx)} 条经 VLM 复审: 修正 {fixed} 个错判, 引入 {broke} 个新错")

# %%
# ── 成本对账: 1000 条流量的推理开销 ──
n_route = band.mean()
cost_small = 1000 * (dt / 256) * 1000            # 全量过小模型 (batch=256 摊薄), ms
cost_vlm_all = 1000 * t_bat * 1000               # 假设全量过 VLM, ms
cost_cas = cost_small + 1000 * n_route * t_deep * 1000
print(f"每 1000 条流量的计算时间:")
print(f"  纯小模型:   {cost_small/1000:8.2f} s   (质量下限)")
print(f"  纯 VLM:     {cost_vlm_all/1000:8.2f} s   (质量上限, 成本 {cost_vlm_all/cost_small:,.0f}x)")
print(f"  级联:       {cost_cas/1000:8.2f} s   (VLM 只吃 {n_route:.1%} 流量, "
      f"成本是纯 VLM 的 {cost_cas/cost_vlm_all:.1%})")
