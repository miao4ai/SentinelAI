# %% [markdown]
# # LLM 模型实验 — Qwen2.5-VL 本地部署 · 多帧输入 · 审核 Prompt · 微调数据 · PEFT
#
# 覆盖第 7/8 章的五个问题：**7.1** VLM 本地部署选型（4-bit 量化加载）、**7.2** 多帧
# 输入格式（`<|vision_start|>…<|vision_end|>`）、**7.3** 审核专属 Prompt（人设 + CoT）、
# **8.1** 微调数据集构造（UCF-Crime + hard examples → VQA JSONL）、**8.2** PEFT
# （QLoRA 指令微调 + 权重合并）。
#
# 约定：**问题与解释写在 markdown，代码放 cell**，输出是 GPU 机（L4 24GB）上的真实运行结果。
# 前置：`data/xd-violence/frames16/` 已抽帧；模型权重已在 HF 缓存（跑过 V3 实验即有）。

# %% [markdown]
# ## 7.1 VLM 本地部署选型
#
# **问题**：选择 Qwen2-VL-7B-Instruct 作为核心底座，在本地显卡上进行 4-bit/8-bit 量化加载。
# 如何实现？为何挑选这个模型？好处在哪里？
#
# **为何选它（我们实际用它的直接升级版 Qwen2.5-VL-7B-Instruct，接口/模板完全同构）**：
#
# 1. **开源可商用**：Apache-2.0 权重，可本地部署、可微调、可合并再分发——审核场景数据
#    敏感，闭源 API（GPT-4o/Gemini）把帧传出去往往直接不合规。
# 2. **7B 是本地显卡的甜点位**：bf16 全量 ~16GB，**4-bit NF4 量化后 ~6GB**，单张
#    L4/3090/4090（24GB）加载后还有充足余量放 8 帧的视觉 token 和 KV cache；
#    2B 太弱、72B 单卡放不下。
# 3. **原生视频/多图支持**：视觉编码器 + M-RoPE 位置编码原生支持多帧序列（见 7.2），
#    动态分辨率（`min_pixels`/`max_pixels`）可以精确控制每帧的 token 预算。
# 4. **判定可以拿到概率**：Instruct 模型可被强制回答 yes/no，取**首 token 的
#    yes/no logits 归一化即 P(violent)**——不用采样、可校准、可算 AUC。
# 5. **我们自己的实测背书**（experiments.md §4.9/§4.10）：零样本 XD 留出集 AUC 0.900，
#    QLoRA 微调后 0.990；跨域到 UCF-Crime 零样本 0.879 ——特征迁移全军覆没的场景里
#    VLM 的世界知识是目前唯一能跨域的路线。
#
# **量化加载的实现**：`bitsandbytes` 的 `BitsAndBytesConfig` —— 4-bit 用 NF4 量化 +
# bf16 计算；8-bit 只需换成 `load_in_8bit=True`（显存 ~9GB，精度略高、速度略慢）。
#
# **为什么 4-bit 后是 ~6GB 而不是 16/4=4GB？** 三个原因：①(大头) bitsandbytes 只量化
# `nn.Linear`——**词嵌入和 lm_head 不量化**，Qwen 词表 15.2 万，这两块 ~1.1B 参数以 bf16
# 保留就是 ~2.2GB（RMSNorm/bias 也留高精度）；②量化元数据：NF4 每 64 权重存一个缩放常数，
# double_quant 后仍 ~0.13 bit/权重 ≈ 0.1GB；③底数其实是 8.3B 总参数（7.6B LLM + 0.67B
# 视觉塔）= bf16 ~16.6GB。对账：LLM 线性 NF4 ~3.3 + 常数 0.1 + 嵌入/lm_head bf16 ~2.2 +
# 视觉塔 NF4 ~0.3 ≈ **5.9GB**，正好等于实测——有效压缩率 ~2.8×，不是 4×。
# （注：下面 cell 打印的"参数量 4.69B / bf16 约 9GB"偏小，是因为 bnb 把 4-bit 权重打包进
# int8 张量后 `numel()` 报的是打包后的一半；真实总参数 8.3B。）
#
# **追问：嵌入 + lm_head 这 ~2.2GB 能压吗？** 能，分四档：① int8 行量化嵌入/lm_head
# （2.2→1.1GB，查表本是访存瓶颈、几乎无感，但 lm_head 的 logits 会被轻微扰动——它是
# 全模型对量化最敏感的一层，误差不经任何后续层稀释）；② AWQ/GPTQ 整模型 4-bit（社区有
# 现成 Qwen2.5-VL-AWQ，但它们通常也把嵌入/lm_head 留在 fp16，压的还是线性层）；③ 权重
# 绑定 tied embedding 直接砍半且无损——可惜 Qwen2.5 只有 0.5B/1.5B/3B 绑了，7B 没绑，
# 事后绑不了；④ 最激进：打分模式只读 yes/no 两个 logit，把 152k×3584 的 lm_head 换成
# 2×3584 的**分类头**（1.1GB→14KB），与 §4.10 两模式部署天然契合——打分用分类头，
# 解释模式才需要完整 lm_head。L4 24GB 上这 2.2GB 不是瓶颈，维持默认即可；这套压缩
# 在下沉到 8GB 级显卡时才是关键路径。

# %%
import glob, json, os, textwrap
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA = os.path.expanduser("~/documents/SentinelAI/data/xd-violence")
FRAMES = f"{DATA}/frames16"
N_IMG = 8

bnb = BitsAndBytesConfig(                       # 8-bit 替代: BitsAndBytesConfig(load_in_8bit=True)
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                  # NF4: 对正态分布权重信息最优的 4-bit 格点
    bnb_4bit_compute_dtype=torch.bfloat16,      # 反量化后用 bf16 做矩阵乘
    bnb_4bit_use_double_quant=True,             # 二次量化量化常数, 再省 ~0.4GB
)
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto").eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params/1e9:.2f}B   显存占用: {torch.cuda.memory_allocated()/1e9:.1f} GB "
      f"(bf16 全量约 {n_params*2/1e9:.0f} GB → 4-bit 压到 ~1/4)")

# %% [markdown]
# ## 7.2 多帧输入支持
#
# **问题**：适配 Qwen-VL 的多帧图像/视频拼接输入格式（`<|vision_start|>…<|vision_end|>`）。
# 如何实现？给出一个拼接结果。
#
# **实现**：不需要手拼特殊 token——把 N 帧作为 `content` 里的 N 个 `{"type":"image"}`
# 条目传给 `processor.apply_chat_template`，模板自动为**每帧**生成一段
# `<|vision_start|><|image_pad|><|vision_end|>`；随后 `processor(...)` 把每个
# `<|image_pad|>` 占位符**展开**成该帧实际的视觉 patch token（数量由分辨率预算决定，
# 28×28 一个 patch，2×2 patch 合并为 1 个 token）。下面用一个真实 XD 片段演示。

# %%
def safe(k): return k.replace("/", "_")

def load_frames(key, n=N_IMG):
    arr = np.load(f"{FRAMES}/{safe(key)}.npz", allow_pickle=True)["frames"]   # (16,224,224,3)
    return [Image.fromarray(arr[i]) for i in range(0, len(arr), len(arr) // n)][:n]

DEMO_KEY = "Bullet.in.the.Head.1990__#00-04-41_00-06-12_label_B1-0-0"   # 喋血街头 斗殴(真实: 暴力)
frames = load_frames(DEMO_KEY)

messages = [{"role": "user",
             "content": [*[{"type": "image"} for _ in frames],
                         {"type": "text", "text": f"这是同一视频均匀采样的 {len(frames)} 帧。"}]}]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== 模板拼接结果(截取) ===")
print(textwrap.shorten(text, 500, placeholder=" …"))

inputs = proc(text=[text], images=frames, return_tensors="pt")
img_tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
n_img_tok = int((inputs.input_ids == img_tok).sum())
print(f"\n展开后总 token: {inputs.input_ids.shape[1]}  其中视觉 token: {n_img_tok} "
      f"(每帧 ~{n_img_tok//len(frames)}), 文本 token: {inputs.input_ids.shape[1]-n_img_tok}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(frames), figsize=(2.0 * len(frames), 2.2))
for a, f in zip(ax, frames): a.imshow(f); a.axis("off")
fig.suptitle(f"输入的 {len(frames)} 帧: {DEMO_KEY}", fontsize=9); plt.show()

# %% [markdown]
# ## 7.3 审核专属 Prompt Engineering 设计
#
# **问题**：设计具备 System Prompt 角色定义的审核员人设；要求模型进行 Chain-of-Thought
# 推理——先描述画面内容，再判断是否违规，最后输出违规类型。给出案例、描述和判定结果。
#
# **实现**：system 角色给"内容安全审核员"人设 + 违规类型清单；user 侧强制三步 CoT 输出
# 格式。下面在两个 **V1/V2 融合模型判错过的样本**上跑真实推理：
# 喋血街头 斗殴（⑤late 漏报 FN）和 Fury 坦克行军（⑤late 误报 FP）——看 VLM 的 CoT 怎么判。

# %%
SYSTEM = ("你是短视频平台的内容安全审核员，负责判定视频是否包含真实或影视暴力。"
          "违规类型仅限: violence(斗殴/袭击), weapon(持械伤人), gore(血腥伤残), riot(骚乱), "
          "none(不违规)。你只依据画面证据，不臆测画面外的内容。")
COT = ("以上是同一视频均匀采样的 8 帧。请按三步输出：\n"
       "1) 画面描述：一两句话客观描述看到的内容；\n"
       "2) 违规判定：是 / 否；\n"
       "3) 违规类型：从 violence/weapon/gore/riot/none 中选一个，并给一句理由。")

def moderate(key):
    fr = load_frames(key)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [*[{"type": "image"} for _ in fr],
                                         {"type": "text", "text": COT}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=fr, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=220, do_sample=False)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
    fig, ax = plt.subplots(1, len(fr), figsize=(2.0 * len(fr), 2.2))
    for a, f in zip(ax, fr): a.imshow(f); a.axis("off")
    truth = "正常" if "_label_A" in key else "暴力"
    fig.suptitle(f"{key}  (真实标签: {truth})", fontsize=9); plt.show()
    print(ans, "\n" + "─" * 80)

moderate("Bullet.in.the.Head.1990__#00-04-41_00-06-12_label_B1-0-0")   # 融合模型漏报的斗殴
moderate("Fury.2014__#01-24-39_01-26-24_label_A")            # 融合模型误报的战争片行军戏

# %% [markdown]
# ## 8.1 微调数据集构造
#
# **问题**：将前期的 UCF-Crime 数据和 V1/V2 跑错的 Hard Examples，清洗并转化为 Qwen
# 接受的 VQA 对话 JSON/JSONL 格式；强制模型以结构化 JSON 返回判定结果
# （`{"is_violating": true, "category": "violence", "reason": "xxx"}`）。
#
# **实现**：两路数据源统一进同一个 schema——
#
# 1. **UCF-Crime**：文件名自带类别（`Fighting033`、`Assault011`、`Normal_Videos_x`…），
#    直接映射到审核类别（Fighting/Assault→violence, Shooting→weapon, Explosion→riot…）。
# 2. **Hard Examples**：`error_analysis.ipynb` 里 ⑤late(V1)/融合(V2) 判错的真实片段
#    （下面 16 个 key 即该 notebook §9 的 8 FP + 8 FN），XD 的 `label_A/B1/B2/B4/B6/G`
#    编码映射到类别。错例比随机样本对微调更有价值——这正是老模型的决策边界。
#
# 每条样本 = Qwen 对话格式（system 审核员人设 + user 8 帧图像与指令 + assistant 结构化
# JSON 答案）。训练时 loss 只加在 assistant 段（见 8.2）。

# %%
# --- 类别映射：两套原始标注 → 统一审核类别 ---
UCF_CAT = {"Fighting": "violence", "Assault": "violence", "Abuse": "violence",
           "Shooting": "weapon", "Robbery": "weapon", "Explosion": "riot",
           "Arson": "riot", "Normal": "none"}
XD_CAT = {"B1": "violence", "B2": "weapon", "B4": "riot", "B6": "gore",
          "G": "weapon", "A": "none"}          # XD 标签码: B1斗殴 B2枪击 B4骚乱 B6血腥 G爆炸/枪战

def xd_category(key):
    tag = key.split("_label_")[1].split("-")[0]
    return XD_CAT.get(tag, "violence")

# V1/V2 判错的 hard examples(来自 error_analysis.ipynb §9, 前8=FP误报 后8=FN漏报)
HARD = ["v=W0oYXpd_WIU__#1_label_A", "v=wDPzFSdbqnU__#1_label_A",
        "v=vFPQ_NiDBIU__#00-06-00_00-12-00_label_A", "v=UQ9jjX_4ywU__#1_label_A",
        "v=X59__bw-seg__#1_label_A", "Fury.2014__#01-24-39_01-26-24_label_A",
        "v=fkps18H3SXY__#01-30-00_01-36-00_label_A", "Fury.2014__#00-01-50_00-02-58_label_A",
        "v=idBwMN0E9Qg__#00-18-44_00-24-16_label_B4-0-0", "v=rldjZyZwUEM__#1_label_B1-0-0",
        "v=X1LwYJ8pDhc__#1_label_G-0-0", "v=qQCTvJq83_s__#1_label_B1-0-0",
        "The.Bourne.Legacy.2012__#01-03-59_01-04-59_label_B2-0-0",
        "IP.Man.2.2010__#00-52-23_00-53-29_label_B1-0-0",
        "Bullet.in.the.Head.1990__#00-04-41_00-06-12_label_B1-0-0",
        "v=oLFGaFmYoRw__#1_label_B6-0-0"]

USER_INSTR = ("以上是同一视频均匀采样的 8 帧。判断该视频是否违规，只输出 JSON："
              '{"is_violating": true|false, "category": "violence|weapon|gore|riot|none", '
              '"reason": "一句话画面证据"}')

def to_vqa(video, category, reason, frame_dir):
    """一条视频 → 一条 Qwen VQA 对话样本(JSONL 的一行)。"""
    ans = {"is_violating": category != "none", "category": category, "reason": reason}
    return {"conversations": [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [*[{"type": "image", "image": f"{frame_dir}/{safe(video)}_{i:02d}.jpg"}
                                       for i in range(N_IMG)],
                                     {"type": "text", "text": USER_INSTR}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(ans, ensure_ascii=False)}]}]}

# UCF-Crime: 文件名 → 类别(reason 留待人工/VLM 预标注后清洗)
ucf_rows = [to_vqa(v, UCF_CAT[v.rstrip("0123456789").split("_")[0].rstrip("0123456789")],
                   r, "data/ucf-crime/frames")
            for v, r in [("Fighting033_x264", "多人互相挥拳踢打，有人被打倒在地"),
                         ("Shooting008_x264", "持枪者向人开枪，目标应声倒地"),
                         ("Normal_Videos_308_x264", "商场内正常行走购物，无冲突")]]
# XD hard examples: label 码 → 类别
xd_rows = [to_vqa(k, xd_category(k), "(hard example: V1/V2 融合模型判错)", "data/xd-violence/frames8")
           for k in HARD if os.path.exists(f"{FRAMES}/{safe(k)}.npz")]

out_path = f"{DATA}/sft_moderation.jsonl"
with open(out_path, "w") as f:
    for r in ucf_rows + xd_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"写出 {len(ucf_rows) + len(xd_rows)} 条 → {out_path}\n")
print("=== 3 条样本(assistant 目标即结构化 JSON) ===")
for r in [ucf_rows[0], ucf_rows[2], xd_rows[-1]]:
    u = r["conversations"][1]["content"]
    print(f"user: [{len(u)-1} 帧图像] + 指令 | assistant → {r['conversations'][2]['content'][0]['text']}")

# %% [markdown]
# ## 8.2 参数高效微调 (PEFT)
#
# **问题**：纯本地用 LLaMA-Factory 或手写 LoRA/QLoRA 脚本对 Qwen2-VL 的 Attention 层做
# 指令微调 (SFT)；模型权重合并与本地保存。
#
# **我们选手写 QLoRA**（完整脚本: `scripts/finetune_qwen_qlora.py`，已在本仓库跑通两次）：
# LLaMA-Factory 适合快速起步，但手写脚本对**损失掩码**（loss 只加在 assistant 答案 token
# 上）、**冻结视觉塔**、**成对评测**（同一留出集 adapter on/off 各测一遍）这些关键细节
# 完全可控。配方：
#
# - **底座 4-bit NF4** 冻结（7.1 的加载方式）+ `prepare_model_for_kbit_training`；
# - **LoRA r=16, α=32**，只挂在 **LLM 的 attention (q/k/v/o_proj) + MLP (gate/up/down_proj)**
#   上，视觉塔不动——可训练参数 **~0.57%**，L4 24GB 单卡即可 SFT；
# - 实测（experiments.md §4.10）：1 epoch 后留出集 **AUC 0.900 → 0.990，F1 0.301 → 0.948**。
#
# **权重合并**：4-bit 底座**不能直接合并**（权重已量化）。正确姿势：以 bf16 重新加载底座
# → `PeftModel.from_pretrained` 挂 adapter → `merge_and_unload()` 把 ΔW=BA 加回原权重
# → `save_pretrained` 存成独立模型目录。合并后推理不再依赖 peft，也可再次量化部署。
# 日常部署其实**不必合并**：40MB adapter 挂/摘即可在"打分模式"(on)与"解释模式"(off,
# 零样本解释能力更完整)间切换（§4.10 的两模式部署）。
#
# **配方逐项拆解（每个术语具体在做什么）**
#
# **① LoRA 到底改了什么**。每个被选中的线性层原来是 `h = W₀x`（W₀ 冻结不动），LoRA 在旁边
# 并联一条低秩旁路：
#
# ```
# h = W₀x + (α/r) · B·A·x        A: r×d_in（随机初始化）  B: d_out×r（初始化为 0）
# ```
#
# r=16 就是这条旁路的"秩"——它假设**任务适配所需的权重变化 ΔW 集中在一个 16 维的低秩子
# 空间里**，不需要动完整的 3584×3584。B 初始化为 0 保证训练开始时旁路输出为零、模型行为
# 与底座完全一致，再从零学起。α=32 是缩放系数，实际乘 α/r=2——它把"旁路信号相对底座多
# 响"从学习率里解耦出来（换 r 不用重新调 lr）。算一笔账：7 个目标矩阵每层旁路参数
# 16×(d_in+d_out) 加总 ≈1.44M，×28 层 ≈ **40M 参数 = 0.57%**，存成 fp32 就是那个 190MB
# 的 adapter 文件。
#
# **② 为什么挂在 q/k/v/o + gate/up/down 上**。q/k/v/o_proj 是注意力的四个投影（决定"每个
# token 看哪里、取什么"），gate/up/down_proj 是 MLP 的三个矩阵（决定"取到的信息怎么加工"）。
# QLoRA 论文的消融结论是：**同等参数预算下，覆盖全部线性层 > 只挂注意力层**——所以 7 个
# 全挂。没挂的是：词嵌入/lm_head（1.1B，挂了参数量暴涨且任务不需要改词表语义）和视觉塔。
#
# **③ 为什么冻结视觉塔**。三个理由：(a) 3.3k 样本对 0.67B 视觉参数是杯水车薪，硬训会把
# 预训练学到的通用视觉能力冲掉（灾难性遗忘）；(b) 7.3 的 CoT 显示零样本就能准确**描述**
# 画面——感知没毛病，差的是**判定口径**，而口径长在 LLM 层；(c) 视觉塔不训就不用存它的
# 梯度和优化器状态，省显存。
#
# **④ 4-bit 底座怎么还能训**。W₀ 以 NF4 冻结，前向时每个 64 权重块**现场反量化成 bf16**
# 参与矩阵乘；反向时梯度**穿过** W₀ 流到 LoRA 的 A、B 上（链式法则需要用 W₀ 的值，但不
# 需要给 W₀ 算梯度）。显存账：AdamW 给每个可训练参数存 2 份状态，全量微调 7.6B 要
# ~60GB 优化器状态，LoRA 只要 40M×8B ≈ **0.3GB**——这就是 24GB 单卡能 SFT 的全部秘密。
# `prepare_model_for_kbit_training` 是配套的三件小事：把 LayerNorm 等敏感层留在 fp32
# 稳定训练、打开梯度检查点（用重算换显存）、让输入嵌入可传梯度（否则梯度到不了 LoRA）。
#
# **⑤ 损失掩码在掩什么**。训练样本是一整条对话拼成的 token 序列，但只有答案段算 loss：
#
# ```
# 序列:  [system 审核员人设][user: 8帧×64视觉token + 提问][assistant: "yes"]
# 标签:  [     -100        ][          -100              ][     ✓ 算 loss   ]
# ```
#
# -100 是 PyTorch 交叉熵的 ignore_index。不掩的话，模型会花大部分梯度去学"复述提示词和
# 视觉 token"——那是它本来就会的东西，学了白学还稀释答案信号。掩码后每条样本的全部梯度
# 都集中在那 1 个 yes/no token 上。
#
# **⑥ 成对评测在控制什么**。同 400 个片段、同帧、同 prompt、同解码参数，唯一差别是
# adapter 挂/摘——两组分数的差异**全部**归因于微调本身，抽样方差被配对消掉（类比配对
# t 检验之于独立样本 t 检验）。若各抽一批不同片段分别测，±0.02 级别的 AUC 差根本分不清
# 是微调效果还是抽样噪声。

# %%
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

ADAPTER = f"{DATA}/qwen_qlora_adapter"
if os.path.isdir(ADAPTER):
    peft_model = PeftModel.from_pretrained(model, ADAPTER)     # 挂上我们已训好的 adapter
    src = f"已训练 adapter: {ADAPTER}"
else:                                                          # 无 adapter 时演示从零挂 LoRA
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    peft_model = get_peft_model(model, lcfg)
    src = "新挂载的 LoRA(未训练)"
tr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
tot = sum(p.numel() for p in peft_model.parameters())
print(f"{src}\n可训练参数: {tr/1e6:.1f}M / {tot/1e9:.2f}B = {tr/tot:.2%}")
for f in sorted(glob.glob(f"{ADAPTER}/*")):
    print(f"  {os.path.basename(f):<28}{os.path.getsize(f)/1e6:8.1f} MB")

# %%
# adapter 的效果：同一片段成对打分 P(violent) —— on(微调) vs off(零样本)
YES_NO_PROMPT = ("You are given 8 frames evenly sampled from one video. Does the video depict "
                 "physical violence (fighting, assault, weapons used against people, blood or "
                 "injury)? Answer with exactly one word: yes or no.")

def first_ids(words):
    ids = set()
    for w in words:
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()):
            t = proc.tokenizer.encode(v, add_special_tokens=False)
            if t: ids.add(t[0])
    return sorted(ids)
YES, NO = first_ids(["yes"]), first_ids(["no"])

def p_violent(m, key):
    fr = load_frames(key)
    msgs = [{"role": "user", "content": [*[{"type": "image"} for _ in fr],
                                         {"type": "text", "text": YES_NO_PROMPT}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=fr, return_tensors="pt").to(m.device)
    with torch.no_grad():
        lg = m(**inp).logits[0, -1]
    py, pn = lg[YES].logsumexp(0), lg[NO].logsumexp(0)
    return float(torch.sigmoid(py - pn))

if isinstance(peft_model, PeftModel) and os.path.isdir(ADAPTER):
    for key in ["Bullet.in.the.Head.1990__#00-04-41_00-06-12_label_B1-0-0",
                "Fury.2014__#01-24-39_01-26-24_label_A"]:
        pon = p_violent(peft_model, key)
        with peft_model.disable_adapter():
            poff = p_violent(peft_model, key)
        truth = "正常" if "_label_A" in key else "暴力"
        print(f"{key[:52]:<54} 真实={truth}  P(violent): 零样本 {poff:.2f} → QLoRA {pon:.2f}")

# %%
# 权重合并与本地保存(需 bf16 重载底座, ~16GB 显存 + ~16GB 磁盘, 按需打开)
RUN_MERGE = False
if RUN_MERGE:
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                       device_map="auto")
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    merged.save_pretrained(f"{DATA}/qwen2.5-vl-7b-moderation")   # 独立完整模型, 不再依赖 peft
    proc.save_pretrained(f"{DATA}/qwen2.5-vl-7b-moderation")
else:
    print("RUN_MERGE=False — 合并代码见上, 部署走 40MB adapter 挂/摘的两模式方案即可")

# %% [markdown]
# ## 8.3 PEFT 完成后：各项指标提高了多少
#
# **问题**：QLoRA 微调完成后，相对零样本底座，各项 metric 的提升有多少？
#
# **实验过程**（`scripts/finetune_qwen_qlora.py`，完整跑过两次结论一致）：
#
# 1. **数据**：frames16 全部片段按**电影分组** 80/20 切分（同一部电影不跨训练/测试，防泄漏），
#    训练侧构成 ~3.3k 条 "8 帧 + yes/no 提问 → 答案" 指令对；
# 2. **训练**：8.2 的配方——4-bit NF4 冻结底座 + r=16 LoRA（attention+MLP），**loss 只加在
#    答案 token 上**，1 epoch（L4 单卡约 3 小时，末期 mean_loss≈0.10）；
# 3. **评测**：从**留出电影**抽 400 个片段，每个片段打分**两次**——adapter ON（微调）与
#    adapter OFF（零样本）——除 LoRA 权重外完全相同，是严格成对的对照；
#    每片段分数 = 首 token 的 P(yes)。分数已缓存（`qwen_qlora_scores.npz`），
#    下面直接从缓存计算指标，无需重新训练/推理。

# %%
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

z = np.load(f"{DATA}/qwen_qlora_scores.npz", allow_pickle=True)
yv, s_zs, s_ft = z["y"], z["s_zs"], z["s_ft"]
print(f"成对评测集: {len(yv)} 片段, {yv.mean():.0%} 暴力")

def metrics(s):
    pred = (s >= 0.5).astype(int)
    o = np.argsort(-s); k = max(1, len(yv) // 10)
    return {"AUC": roc_auc_score(yv, s), "AP": average_precision_score(yv, s),
            "F1@0.5": f1_score(yv, pred),
            "Precision@0.5": precision_score(yv, pred, zero_division=0),
            "Recall@0.5": recall_score(yv, pred), "Accuracy@0.5": accuracy_score(yv, pred),
            "p@top10%": yv[o[:k]].mean()}

tbl = pd.DataFrame({"零样本 (adapter off)": metrics(s_zs), "QLoRA (adapter on)": metrics(s_ft)})
tbl["Δ 提升"] = tbl["QLoRA (adapter on)"] - tbl["零样本 (adapter off)"]
tbl.round(3)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for tag, s in [("zero-shot", s_zs), ("QLoRA", s_ft)]:
    fpr, tpr, _ = roc_curve(yv, s)
    ax[0].plot(fpr, tpr, label=f"{tag}  AUC={roc_auc_score(yv, s):.3f}")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.5)
ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[0].legend()
for a, (tag, s) in zip(ax[1:], [("zero-shot P(yes)", s_zs), ("QLoRA P(yes)", s_ft)]):
    a.hist(s[yv == 0], bins=30, alpha=0.6, label="normal")
    a.hist(s[yv == 1], bins=30, alpha=0.6, label="violent")
    a.axvline(0.5, color="k", ls="--", lw=0.8); a.set(title=tag, xlabel="P(yes)"); a.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# **结果与结论**：
#
# | 指标 | 零样本 | QLoRA | Δ |
# |---|---|---|---|
# | AUC | 0.900 | **0.990** | +0.090 |
# | F1@0.5 | 0.301 | **0.948** | **+0.647** |
# | p@top10% | 1.000 | 1.000 | 0 |
#
# 1. **排序质量**（AUC/AP）：0.900 → 0.990——零样本本来就排得不错，微调把中段的错排
#    几乎清零，追平了 ① early 融合在同规模数据上的水平。
# 2. **阈值指标暴涨的真正原因是校准**：F1@0.5 从 0.301 → 0.948（+0.65）。看右侧直方图：
#    零样本的 P(yes) 整体压得很低（模型嘴上保守，"yes" 说得少），0.5 阈值下 recall 极低；
#    微调后两类分布被拉开并对齐到 0.5 两侧。也就是说**零样本欠的主要是校准，微调一并
#    修好了校准 + 排序**——这与 §4.8 "阈值校准能救回大半" 的结论互相印证。
# 3. **头部精度不变**：p@top10% 两者都是 1.000——最显眼的暴力零样本就能抓住，
#    PEFT 的增益集中在**中段样本与运营可用的固定阈值**上。
# 4. **性价比**：40MB adapter、0.57% 可训练参数、单卡 1 epoch，换来 F1 +0.65；
#    且 8.2 的成对打分显示 hard example 上仍有残余误差——微调不是魔法，
#    是把决策边界推向任务分布。
