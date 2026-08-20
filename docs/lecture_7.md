# Lecture 7 总结 — CLIP / 对比学习 / Triplet Loss，与各模态用什么模型

承接 Lecture 6（多模态融合）。这一讲两条线：① CLIP 和对比学习的原理，怎么引申到 triplet loss；② 每个模态具体用什么模型来编码。

---

## 1. CLIP 与对比学习 (Contrastive Learning)

### 1.1 CLIP 结构

CLIP = **两个独立编码器 + 一个共享空间**：

```
图像 ─▶ [Image Encoder (ViT/ResNet)]  ─▶ 投影 ─▶ 归一化 ─▶ image emb ┐
文本 ─▶ [Text  Encoder (Transformer)] ─▶ 投影 ─▶ 归一化 ─▶ text  emb ┘
                                                        ↘ 比余弦相似度 ↙
```

两个编码器**各自独立**（不共享权重），但被训练成：**配对的图文，embedding 靠近；不配对的，远离**。这就是「coordinated representation」——用一个学出来的对齐把两个模态放进同一空间（见 [`fusion.md`](fusion.md) 第②层）。

### 1.2 对比学习的核心思想

一句话：**把正样本（该靠近的）拉近，把负样本（该远离的）推开**。

要素：
- **anchor（锚）**：当前样本
- **positive（正）**：和 anchor 匹配的（同一图的正确描述）
- **negative（负）**：不匹配的

关键在**负样本**：模型是靠"把正样本从一堆负样本里挑出来"来学习的，所以**负样本越多、越难，学得越好**。

### 1.3 CLIP 的损失：对称 InfoNCE

一个 batch 有 N 对 (图, 文)。算 **N×N 余弦相似度矩阵**（除以温度 τ）：对角线是配对（要高），非对角是不配对（要低）。

```
对每张图：在 N 个文本上做 softmax，正确的那个文本当标签 → 交叉熵   (image → text)
对每个文本：在 N 张图上做 softmax，正确的那张图当标签  → 交叉熵   (text → image)
Loss = (L_i2t + L_t2i) / 2
```

- 这就是 **InfoNCE**（softmax over 多个负样本），CLIP 用 **batch 内其它样本当负样本**（in-batch negatives，免费的大量负样本）
- **温度 τ**（代码里 `logit_scale`）：缩放相似度、控制分布锐度，可学习
- 结果：一个对齐的共享空间 → 支持**零样本**（拿文本 prompt 当分类器，正是我们 `clip_screener.py` 的做法）

### 1.4 引申：对比损失家族 → Triplet Loss

对比学习的损失有一条演化线，**负样本个数**是主线：

| 损失 | 一次看几个负 | 形式 | 代表 |
|---|---|---|---|
| **Contrastive pair loss** | 1 对（正或负） | 正样本拉近；负样本推到 margin 外 | Hadsell 2006 |
| **Triplet loss** | 1 正 + 1 负 | 正要比负近至少一个 margin | FaceNet |
| **InfoNCE / N-pair** | 1 正 + **N 个负** | softmax 从 N 个负里挑正 | SimCLR / **CLIP** |

**Triplet Loss** —— (anchor a, positive p, negative n)：

```
L = max( 0,  d(a,p) − d(a,n) + margin )
```

- 直觉：**正样本到锚的距离，要比负样本近至少一个 margin**；已经满足就不罚（max 里的 0）
- `d` 通常是欧氏距离或 1−余弦
- **难点：负样本挖掘 (negative mining)**——随机负样本太容易、没梯度；要挑 **hard/semi-hard negative**（那些"离得太近的负样本"）才学得动

**和 CLIP 的关系**：
- Triplet = "**1 个负 + margin**" 的对比学习
- CLIP/InfoNCE = "**N 个负 + softmax + 温度**" 的对比学习——把 margin 换成了在很多负样本上的 softmax 分类
- 所以 CLIP 可以看成 triplet 的"多负样本、软化"版；triplet 是对比学习最直观的入门形式

> 记忆：**对比学习 = 拉正推负**；triplet 用 1 个负 + margin，InfoNCE/CLIP 用一堆负 + softmax。

---

## 2. 各模态用什么模型编码

**一个关键区分（你让我纠正的）**：**Whisper 是 ASR（语音→文字），不是 language model**。它属于「语音」这条线，把音频里的话转成文字；真正做「文本/语言语义」分析的是 RoBERTa/DeBERTa/XLM-R 这类 NLP 模型。别把「语音识别」和「语言理解」混为一谈。

### 2.1 模态 → 模型对照

| 模态 | 任务 | 常用模型 | 我们项目里 |
|---|---|---|---|
| **图像 / 帧** | 帧级表征、图文对齐 | ResNet / EfficientNet / ViT；**CLIP**（图文） | `visual_expert.py` / `clip_screener.py` |
| **视频（时序）** | 帧间动作 | I3D / **VideoMAE** / TimeSformer | 计划中（cross-attn 的 K/V） |
| **音频（非语音事件）** | 枪声/尖叫/爆炸 | **AST** / VGGish / **BEATs (Microsoft)** | `audio_expert.py`（AST；BEATs 作升级项） |
| **语音 (ASR)** | 语音 → 文字 | **Whisper (OpenAI)** / faster-whisper | 计划中（spec 用 faster-whisper） |
| **文本 / 语言** | 违规语义、仇恨言论 | RoBERTa / DeBERTa / **XLM-R** | `text_expert.py`（XLM-R 毒性） |

### 2.2 逐个说明

**音频 → BEATs**（你说的对）
- **BEATs** = 微软的音频表征模型（unilm 家族），用**声学 tokenizer + 掩码预训练**（类比 BERT 之于文本）。在 AudioSet 上是 SOTA，比 AST 更强。
- 用途：把音频（mel 频谱）编码成特征，识别枪声/尖叫/爆炸等事件。
- 我们现在用 **AST**（HuggingFace 一行加载、够用），BEATs 作为"想刷精度"的升级项（集成稍麻烦，权重走 unilm 仓库）。

**语音 → Whisper（这是 ASR，不是 LM）**
- **Whisper** = OpenAI 的**语音识别**模型，输入音频、输出**文字转写**（含时间戳）。
- 它是「音频 → 文本」的桥：Whisper 转出文字后，**再交给文本模型（XLM-R）做语义判定**。
- `faster-whisper` 是它的高效实现（我们 spec 里用的）。
- ⚠️ Whisper **不做**"这句话是否违规"的语义判断——那是下游 NLP 模型的活。

**文本 / 语言 → XLM-R / RoBERTa / DeBERTa**
- 真正的"语言模型"这一栏。输入文字（来自 ASR 或 OCR 或评论/元数据），判违规语义。
- 我们用多语言 **XLM-R** 毒性分类器（中英零样本可用）。

**图像 / 帧 → CLIP（呼应第 1 节）**
- CLIP 就是上面对比学习训出来的图文对齐模型，我们拿它做**零样本审核**（`clip_screener.py`）。

### 2.3 一条链串起来

```
视频 ┬─ 帧    → ResNet / CLIP        → 视觉违规
     ├─ 音频  → BEATs/AST            → 声音事件（枪声/尖叫）
     └─ 语音  → Whisper (ASR) → 文字 → XLM-R → 语言违规
```
注意最后一条：**Whisper 只负责转文字，语义判断交给 XLM-R**。

---

## 一句话总结

1. **对比学习 = 拉正推负**；CLIP 用对称 InfoNCE（一堆 in-batch 负样本 + 温度）对齐图文；**triplet loss** 是它的"1 负 + margin"入门版，关键在难负样本挖掘。
2. 各模态各有编码器：图像 ResNet/CLIP、音频 **BEATs**/AST、语音 **Whisper(ASR)**、文本 **XLM-R**。**Whisper 是语音识别不是语言模型**——转完字还要交给 NLP 模型判语义。
