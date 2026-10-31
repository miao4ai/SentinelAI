# 第九章术语笔记 — 推理引擎与服务化

> 配套 `inference_pipeline.ipynb`。本文是**术语 + 知识点清单**，按"必须掌握 / 知道即可"分层，
> 每个概念都挂上我们自己的实测数字。标 ⭐ 的是面试和实战里真正会被追问的。

---

## 0. 先建立主线：prefill vs decode ⭐⭐⭐

**这是整章的地基。不理解这条，后面所有术语都会记成孤立的名词。**

大模型推理分两个性质完全不同的阶段：

| | **Prefill（预填充）** | **Decode（逐 token 生成）** |
|---|---|---|
| 做什么 | 把整个提示词**一次性**过一遍模型，算出所有位置的 K/V | 每次生成 **1 个** token，复用之前的 KV |
| 并行度 | 高——N 个 token 同时算，是**大矩阵乘** | 低——每步只算 1 个 token，是**细长矩阵乘** |
| 瓶颈 | **算力（compute-bound）** | **显存带宽（memory-bound）**：每步都要把全部权重搬一遍 |
| 批处理收益 | 小（本来就在满算力跑） | **巨大**（多条请求共享同一次权重搬运）|
| 量化（W4A16）收益 | 可能**变慢**（反量化开销加在计算瓶颈上）| **显著加速**（权重体积小→搬得快）|

### 为什么这对我们特别重要

**网上 99% 的 vLLM 教程讲的是聊天场景**：提示词短、生成几百上千 token → **decode 主导**。
而我们的审核打分是：

```
输入 559 token（512 视觉 + 47 文本）→ 只生成 1 个 token（yes/no）
```

**100% 是 prefill，几乎没有 decode。** 所以教程里的所有结论对我们都要打问号：

| 教程说 | 我们实测 |
|---|---|
| continuous batching 带来 10-20× 吞吐 | 我们只有 ~1.5×（见 §1.4）|
| KV cache 是瓶颈，要扩大 KV 池 | **KV 根本不是瓶颈**，池子能装 ≈52 条而我们只用 ~8（见 §1.2）|
| 量化到 4-bit 提升吞吐 | **对我们大概率无效甚至更慢** |

**实测铁证**（exp10 帧数扫描）：

```
视觉 token 512 → 256 → 128
延迟     200.3 → 114.1 → 71.8 ms/clip
拟合：latency = 28.7 ms + 0.3349 ms × visual_tokens     R² = 0.99998
```

延迟与**要算多少 token** 严格成正比 = 计算瓶颈的教科书签名。
**背下这个结论：我们的系统是 prefill-compute-bound。**

---

## 1. 推理引擎：vLLM

### 1.1 vLLM 是什么 ⭐

一个**专为 LLM 推理优化的服务引擎**，替代直接用 HuggingFace `model.generate()`。
两种用法：

```python
# 离线引擎（我们 notebook 用的，进程内，便于测量）
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", ...)

# 服务形态（生产用，OpenAI 兼容接口）
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-lora --lora-modules violence=<adapter路径>
# 客户端 POST /v1/chat/completions
```

它解决的核心问题是**显存利用率**和**调度效率**，不是算得更快（单条请求的 FLOPs 没变）。

### 1.2 KV cache ⭐⭐

**是什么**：Transformer 生成第 n 个 token 时，注意力要用到前 n−1 个 token 的 K 和 V 向量。
不缓存就要每步重算全部历史 → O(n²) 爆炸。缓存下来就是 KV cache。

**大小怎么算**：

```
每 token 每层 KV 字节数 = 2(K和V) × kv_heads × head_dim × 2(bf16)
Qwen2.5-7B: GQA 4 个 kv_head × 128 = 512 维
  → 2 × 512 × 2 = 2 KB/token/层，× 28 层 ≈ 57 KB/token
```

**我们的实测**（vLLM 自报）：

```
权重 15.67 GiB + 峰值激活 1.95 GiB  →  KV 池只剩 1.57 GiB = 29,360 tokens
```

⚠️ **这里有个坑我踩过**：vLLM 日志会打印
`"Maximum concurrency for 4,096 tokens per request: 7.17x"`。
这**只是 `29,360 ÷ 4,096`**，是假设"每个请求都长到 `max_model_len`"的**最坏情况规划数字**。
我们的请求只有 559 token → 实际能装 `29,360 ÷ 560 ≈ 52` 条。
**我当时把它误读成瓶颈，写进了文档，后来被实验证伪。**

### 1.3 PagedAttention ⭐⭐⭐

**要解决的问题**：传统实现给每个请求**预分配** `max_seq_len` 的**连续**显存。
但实际用多少事先不知道 → 内部碎片（预留 4096 用了 560，浪费 86%）+ 外部碎片。
业界测算浪费 **60–80%**。

**解法**：抄操作系统的**虚拟内存分页**。
把 KV cache 切成固定大小 **block（默认 16 token）**，用 **block table** 把逻辑位置映射到物理块，
**按需分配、不要求连续**。

**"长度未知怎么预留？"——根本不预留** ⭐（这个问题很常被问）：

```
Prefill：长度已知。559 token → ceil(559/16) = 35 块，精确分配
Decode：每满 16 个 token 才申请 1 块
   token 560 → 第35块写满，不分配
   token 561 → 申请第36块
```

浪费上限被锁死在**每序列最后一块的 ≤15 token**（≈2.6%），对比预分配的 86%。

**池子真用光了怎么办**：
- **准入控制**：调度器检查剩余块数才放新请求进来，留 watermark 防颠簸；
- **抢占（preemption）**：踢掉一个正在跑的请求（通常 LIFO），两种方式——
  **swap**（KV 换到 CPU 内存）或 **recompute**（丢掉 KV，之后重新 prefill，短序列更划算）。
  日志里出现 `Preemption` 警告就是这在发生。

**额外收益：block 可跨请求共享**（copy-on-write）→ 这就是 **prefix caching**（见 §5.1 的坑）。

### 1.4 Continuous Batching（迭代级调度）⭐⭐

**静态批处理的问题**：凑齐 N 条一起跑、一起返回。
**整批耗时由最长的那条决定**——若一条生成 200 token、其余 5 个，那 195 步里大部分位置在空转；
新请求还必须等整批做完（**队头阻塞 head-of-line blocking**）。

**解法**：调度粒度从"一批"降到"**一步**"。每生成一步就重新调度：完成的立刻退出释放位置，
排队的立刻插进来。

**但对我们收益有限** ⭐（这是真正要理解的点）：
我们 `max_tokens=1`，所有"序列"长度完全一样——**根本没有长度方差可利用**。
我们测到的加速来自另外两处：
1. **prefill 的批内并行**（小 GEMM 合成大 GEMM）；
2. **调度开销摊薄**（省掉 N 次 Python 调用和 kernel launch）。

实测（exp10 并发扫描，预处理不计时）：

| 提交条数 | ms/clip | QPS |
|---:|---:|---:|
| 4 | 162 | 6.2 |
| 8 | 102 | 9.8 |
| 32 | 96 | 10.4 |
| 64 | 97 | **10.3**（压平）|

**从 N=8 起就压平在 ~10 QPS**——GPU 已被单请求的 prefill 打满，再加并发也榨不出吞吐。
聊天场景（decode 主导）能做到 10-20× 的批处理收益，我们只有约 1.5×。

### 1.5 引擎参数速查 ⭐

| 参数 | 含义 | 我们的值 / 教训 |
|---|---|---|
| `gpu_memory_utilization` | 允许引擎占整卡显存的比例，**权重之外全给 KV 池** | 0.90。开 CUDA graph 时要降到 ~0.82，否则初始化失败 |
| `max_model_len` | 单请求最大 token 数 | 4096。**它只影响准入规划，不影响实际分配** |
| `enforce_eager` | 关闭 CUDA graph | 我们设 True 省显存；代价是损失 kernel launch 优化 |
| `limit_mm_per_prompt` | 每个请求最多几张图 | `{"image": 8}` |
| `enable_lora` / `max_lora_rank` | 开启 LoRA 热挂载 | True / 16 |
| `max_num_seqs` | 最大并发序列数 | 没调过，默认即可 |

**CUDA graph 是什么**：把一串固定的 GPU kernel 调用录制成一张"图"，之后一次提交整张图，
省掉每个 kernel 的 CPU 端启动开销。对 **decode**（每步都是同样的小 kernel 序列）收益大，
对 prefill 收益小。代价是**捕获时要额外显存**。

### 1.6 LoRA 服务 ⭐

```python
llm = LLM(..., enable_lora=True, max_lora_rank=16)
lora = LoRARequest("violence", 1, "/path/to/adapter")
llm.generate(prompts, params, lora_request=lora)    # 挂 adapter
llm.generate(prompts, params)                        # 不挂 = 零样本底座
```

**关键价值**：一个 16GB 底座可以**按请求粒度**挂不同的 40MB adapter
（暴力检测一个、色情检测一个），**不用部署多个模型**。
我们的"打分模式 / 解释模式"两模式部署就是靠这个开关。

### 1.7 首 token logits 打分（不是生成）⭐⭐

**容易被忽略但很重要**的一招：我们**不让模型生成文本**，只读第一个位置的 logits：

```python
SamplingParams(temperature=0, max_tokens=1, logprobs=20)
# 取 top-20 logprobs 里 yes/no 系列 token 的概率，归一化
P(violent) = P(yes) / (P(yes) + P(no))
```

**好处**：输出是**连续可校准的概率**而非离散字符串 → 能算 AUC、能调阈值、能做级联路由。
而且训练目标（答案 token 的交叉熵）和推理读数**在同一个位置**，完全对齐。

---

## 2. 服务框架：Triton Inference Server

### 2.1 ⚠️ 先澄清同名陷阱 ⭐⭐

| | **NVIDIA Triton Inference Server** | **OpenAI Triton** |
|---|---|---|
| 是什么 | **模型服务框架**（原 TensorRT Inference Server）| **GPU kernel 编程语言**（`@triton.jit` 写 Python 生成 CUDA）|
| 我们用了吗 | **用了**，部署级联第一级 | 没直接写，但 PyTorch/vLLM 底层大量使用 |

面试被问"你用的 Triton"一定要先确认是哪个。

### 2.2 目录即契约 ⭐

Triton **不用写服务代码**，只摆目录：

```
model_repository/fusion_concat_lr/
├── config.pbtxt      # 声明 backend、输入输出张量、批处理策略、实例数
└── 1/model.onnx      # 版本目录；放 2/ 即灰度新版
```

```protobuf
name: "fusion_concat_lr"
backend: "onnxruntime"              # 也可 pytorch / tensorrt / python / vllm
max_batch_size: 256
dynamic_batching { max_queue_delay_microseconds: 500
                   preferred_batch_size: [64, 256] }
instance_group [ { count: 2, kind: KIND_CPU } ]
```

端口：**8000 HTTP / 8001 gRPC（生产用这个，更快）/ 8002 Prometheus 指标**。

### 2.3 Dynamic Batching ⭐

零散的单条请求在一个**微秒级窗口**内自动攒成批再送进模型。
`max_queue_delay_microseconds: 500` 表示最多等 500 µs。

**这是延迟换吞吐的旋钮**：对我们的 LR（单条 63 µs）来说，等 500 µs 几乎白赚吞吐。

实测效果：

| 并发 | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| req/s | 934 | 2172 | 2430 | **2463** |

并发 4 起 dynamic batching + 双实例生效，~16 并发吃满平台。

### 2.4 ONNX / opset ⭐

**ONNX** 是跨框架的模型交换格式；**onnxruntime** 是执行它的运行时（Triton 的一种 backend）。

**opset** 是 ONNX 的算子集版本号。⚠️ **我们踩的坑**：
skl2onnx 默认导出 **opset 22**，而 `tritonserver:24.08-py3` 镜像里的 onnxruntime 只支持到 **21**
→ 模型加载失败、服务起不来。修复：`to_onnx(..., target_opset=21)`。

**教训：导出方与部署镜像的 opset 支持范围必须对表，别用默认值。**

### 2.5 知道即可（我们没用到）

- **ensemble 调度器**：把「预处理 → 模型A → 模型B」串成**服务端的一张图**，客户端一次调用。
  **这才是 Triton 在我们架构里真正该做的事**——把 I3D/AST/文本嵌入三个编码器串起来，
  而不是服务那个 0.09 ms 的 LR 头。我们没搭。
- **`perf_analyzer`**：官方压测工具，自动扫并发出延迟-吞吐曲线。我们用的是自己写的多线程客户端。
- **模型热更新**：`--model-control-mode=explicit` + load/unload API，不重启换模型。
- **版本灰度**：`version_policy` 控制 1/ 和 2/ 并存或只服务最新。

---

## 3. 性能指标术语 ⭐⭐

| 术语 | 含义 | 坑 |
|---|---|---|
| **Latency** | 单条请求耗时 | **必须说明是端到端还是只算模型前向** |
| **Throughput / QPS** | 单位时间处理条数 | **必须注明 batch 与并发**，否则数字无意义 |
| **p50 / p95 / p99** | 延迟分位数 | 审核是异步任务，**队列积压看尾部，p99 比 p50 重要** |
| **饱和吞吐** | 并发足够大时的吞吐上限 | 和"单条延迟的倒数"完全不是一回事 |
| **冷启动 / 预热** | 首批请求含 kernel 编译、缓存预热 | 我们实测**冷热差 10 倍**（1288 vs 122 ms）——**压测必须先热身** |

**Little's 定律**（值得记）：`并发数 = 吞吐 × 单条延迟`。
我们：10.3 QPS × 0.096 s ≈ **1**——意味着 GPU 实际上同时只在处理"约一条请求的工作量"，
**再加并发也没有空闲算力可用**。这是 compute-bound 的又一个佐证。

**端到端的正确定义** ⭐：**从视频输入到 JSON 吐出**，含解码、特征提取/抽帧、模型前向、后处理。
只报模型前向是耍流氓——第十章正是因为补上这一段，才发现 V1/V2 端到端要 3.7 秒
（模型本身只占 0.09~2.4 ms），比 VLM 的 188 ms 慢 19 倍。

---

## 4. 架构术语：级联（Cascade）⭐

```
          全部流量
             │
   ┌─────────▼─────────┐  第一级「低耗流」：小模型（Triton）
   │  p = 小模型置信度   │  63 µs/clip
   └─────────┬─────────┘
   ┌─────────┼─────────┐
   │p<0.4    │0.4≤p≤0.6│p>0.6
   ▼         ▼         ▼
 放行    「高耗流」    判违规
        VLM 深度推理
        （vLLM）
```

**核心假设**：小模型对绝大多数样本很有把握，只有**置信度落在混淆带**的才值得动用大模型。
**验证带选得对不对的指标**：小模型**在带内的准确率**——我们实测 **0.41**（比抛硬币还差），
说明这个带确实框住了它拿不准的样本。

⚠️ **但这个架构在我们这里被第十章推翻了**：级联的前提是"第一级便宜"，
而那个 63 µs 只是 LR 头，**不含特征提取**。算上解码+I3D+AST+ASR 后第一级要 3.7 秒，
**全量过小模型比全量送 VLM 还贵 19 倍**。详见 `chapter10_review.md`。

---

## 5. 测量陷阱（我们真踩过的）⭐⭐⭐

**面试里讲这些比讲成功数字更有说服力。**

### 5.1 prefix caching 污染了 ch9 的加速比（待核实）

第九章报"batch=1 → batch=24 加速 4.2×"（122 → 29 ms/clip），
但第十章用 **295 条互不相同**的片段测出来是 **200 ms/clip**。差了近 7 倍。

**最可能的解释**：ch9 的扫描是 `demo[:1], demo[:2], ..., demo[:24]`——
**后面的批次包含前面所有片段**，而 vLLM v1 默认开启 **automatic prefix caching**，
相同前缀（我们的提示词完全相同，甚至相同图片）会命中缓存。
**那 4.2× 很可能测的是缓存命中率，不是批处理收益。**

> **通用教训：压测必须用互不相同的输入，或显式关掉 prefix caching，
> 否则你测的是自己的缓存。** （这条还没验证，列在待办里。）

### 5.2 把 CPU 预处理算进了 GPU 利用率窗口

ch9 的 NVML 采样显示串行 62% → 整批 **22%**，方向反了。
原因是计时区间里包含了加载 npz、PIL 转换、tokenize 这些 **CPU 工作**，
整批阶段 GPU 在等 CPU，自然采到低利用率。**延迟数字可信，利用率数字作废。**

### 5.3 vLLM v1 的引擎在独立子进程

父进程的 `torch.cuda.memory_allocated()` 读到 **0.0 GB**——
必须用 `nvidia-smi` 才能测到 VLM 的真实占用（18.7 GB）。

### 5.4 误读日志里的"最大并发 7.17×"

见 §1.2。那是 `29,360 ÷ 4,096` 的最坏情况规划值，不是实际瓶颈。

---

## 6. 自测题（能答上来就算掌握了）

1. 为什么我们的 continuous batching 收益只有 ~1.5×，而聊天场景能有 10-20×？
2. PagedAttention 在长度未知的情况下怎么决定分配多少块？池子用光了会怎样？
3. 为什么 W4A16 量化对我们的打分负载大概率无效甚至更慢？
4. `latency = 28.7 + 0.3349 × visual_tokens`，R²=0.99998——这个式子证明了什么？
5. vLLM 日志里的 "Maximum concurrency 7.17x" 是什么意思？是瓶颈吗？
6. Triton 的 `max_queue_delay_microseconds` 调大调小分别换来什么？
7. 为什么我们打分时读 logits 而不是让模型生成 "yes"/"no" 字符串？
8. 一个同事报告"我们的模型 50 ms 延迟"，你要追问哪三件事？
   <br>（提示：端到端还是模型前向？batch/并发多少？热身了吗？）

---

## 附：待办（本章遗留）

| 优先级 | 任务 |
|---|---|
| 高 | **验证 §5.1 的 prefix caching 假设**——用互不相同的片段重测批处理加速比 |
| 中 | CUDA graph（`gpu_memory_utilization` 降到 0.82）——exp11 正在跑 |
| 中 | Triton **ensemble** 串联三个编码器（Triton 真正该做的事）|
| 低 | `perf_analyzer` 规范压测、prefix caching 显式开关对照、HF `generate` 基线做真 A/B |
