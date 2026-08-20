# Lecture 6 总结 — 多模态内容审核：从单模态到融合

把 CV / NLP / Audio 三个模态各自"看懂"内容、再融合成一个审核判定，背后的原理与工程选择。对应代码：`sentinelai/`。

---

## 1. ResNet 的残差结构，以及 Transformer 为何也用残差

### 残差连接是什么

普通深网络堆得越深，训练误差反而**上升**（不是过拟合，是"退化 degradation"）——梯度在多层反传中衰减/爆炸，深层学不动。

ResNet 的解法：不让每层直接学目标映射 `H(x)`，而是学**残差** `F(x) = H(x) − x`，输出 `F(x) + x`：

```
        x ─────────────┐  (identity 捷径 / skip connection)
        │              │
   [Conv-BN-ReLU-Conv-BN]
        │              │
        └──► F(x) ──► (+) ──► ReLU ──►
```

### 好处

1. **梯度高速公路**：反传时梯度能沿 `+x` 那条捷径**直接**流回浅层，缓解梯度消失 → 能训练 50/101/152 层的深网络。
2. **学"扰动"比学"整体"容易**：如果某层最优就是恒等映射，让 `F(x)→0` 比让一堆非线性层拟合出恒等映射容易得多。优化 landscape 更平滑。
3. **恒等兜底**：加层最差也是恒等，不会让效果变差。

### Transformer 也用残差

Transformer 每个子层（自注意力、FFN）都包一层残差 + LayerNorm：

```
x = x + Sublayer(LayerNorm(x))
```

同样的道理：让梯度在几十层（BERT 12/24 层、大模型上百层）里稳定流动。**我们的 cross-attention 模块**（`sentinelai/cross_attention.py`）也是这个套路——attention 和 FFN 各带一个残差 + LayerNorm。

> 一句话：残差是"深度可训练"的关键发明，ResNet 用它做深 CNN，Transformer 用它做深注意力网络。

---

## 2. 取帧：工具、采样策略、为什么 embedding 要 pooling

### 用什么工具

**ffmpeg** 抽帧（`sentinelai/video.py` 还做了 NVDEC 硬件解码探测，GPU 上更快）：

```bash
ffmpeg -i video.mp4 -vf fps=1 output/%06d.jpg   # 每秒抽 1 帧
```

### 取帧策略

| 策略 | 做法 | 取舍 |
|---|---|---|
| **Fixed FPS**（我们用的） | 均匀每秒 N 帧 | 简单可预测，但可能漏短事件/取冗余帧 |
| **Keyframe (I-frame)** | 只取关键帧 | 内容感知、少冗余，但间隔不均 |
| **Uniform-N** | 全片均匀取固定 N 帧 | 帧数固定，适合 batch |
| **Scene-change** | 场景切换处取帧 | 抓转场，实现复杂 |

**核心权衡**：帧多 → 算力大、召回高（不漏短暂违规）；帧少 → 快但可能漏。审核偏向"宁多勿漏"，但要平衡成本。

### 为什么 embedding 要 pooling

两层 pooling，目的不同：

1. **空间 pooling（CNN 内部，Global Average Pooling）**
   ResNet 最后卷积输出是 `2048 × 7 × 7` 的特征图。GAP 把 `7×7` 空间维**平均**掉 → 每帧一个 **2048 维**向量。
   - 得到**定长**向量（不管输入分辨率）
   - **平移不变性**（物体在画面哪个位置都行）
   - 比 `flatten + 全连接` **参数少得多**、不易过拟合

2. **时序 pooling（跨帧 → 视频级）**
   我们 `extract_features` 保留每帧 `(N, 2048)` 不池化；到**分数层**才在 `fusion/signals.py` 用 **Max** 汇聚。
   - **为什么用 Max 不用 Mean**：10 分钟视频里 2 秒暴力，max 顶到 ~1.0；mean 会被正常帧稀释到接近 0（漏检）。

> 设计哲学：**空间取平均**（标准 CNN 语义），**时序取最大**（不漏短暂事件）。

---

## 3. Sentence-BERT（从 embedding 角度）+ NLP 可用的数据源

### 如何理解 Sentence-BERT

**普通 BERT 的问题**：它给每个 token 出 embedding，`[CLS]` 向量**并不适合直接比句子相似度**；要比两句话得把两句一起喂进去（检索 N 句要 O(N²) 次前向，太慢）。

**Sentence-BERT (SBERT)**：
- 用 **孪生网络 (siamese)**，把一句话的 token embedding **pooling（通常 mean）** 成一个**定长句向量**
- 用相似度目标（siamese / triplet loss）训练，使得**语义相近的句子 → 向量也相近**
- 于是可以直接用**余弦相似度**比较句子，支持**检索、聚类、kNN**

**从 embedding 角度**：SBERT = "一句话 → 语义空间里的一个点"。这和 CLIP 把图像映射到向量、ResNet 把帧映射到向量，是**同一个思想**——万物先变成可比较的向量。

> 我们的文本专家（`sentinelai/text_expert.py`）用的是 XLM-R 毒性**分类器**（多标签），不是纯 SBERT；但"文本 → 向量 → 判定"的骨架一致。SBERT 式句向量可以作为特征喂进融合。

### NLP 可用的数据源（做审核时）

文本不只有语音转写，视频里能挖的文本信号很多：

| 来源 | 说明 | 可靠性 |
|---|---|---|
| **ASR 转写** | 语音 → 文本（faster-whisper），主力 | 中（受口音/噪声影响） |
| **OCR** | 画面里烧录的字幕、路牌、meme、弹幕、水印文字 | 中（暗藏违规文字常在这） |
| **用户评论** | 社区反馈信号 | 低（噪声大但量大） |
| **元数据 (metadata)** | 标题、简介、标签/话题、分类 | 中（上传者自述） |
| **上传者/频道信息** | 历史违规记录、账号画像 | 高（先验） |

**要点**：审核不该只看 ASR。很多违规内容故意**不说出来**、只在**画面文字 (OCR)** 或**标签**里露馅——多源文本一起分析才全。

---

## 4. 音频：提取工具、看什么数据、特征

### 用什么工具提取

**ffmpeg** 把音轨抽出来、转成 **16kHz 单声道 PCM**（`sentinelai/audio_expert.py` 直接 ffmpeg 输出 `s16le` 读进 numpy，不落临时文件）：

```bash
ffmpeg -i video.mp4 -f s16le -ac 1 -ar 16000 -
```

### 音频主要看哪些数据

1. **非语音声学事件**：枪声、爆炸、尖叫、玻璃碎、警笛——**直接的暴力/危险信号**（AudioSet 有这些类，AST 零样本就能识别）
2. **语音** → 走 ASR → 文本 → 交给 NLP 专家
3. **音乐/环境音** → 上下文（判断氛围）

### 数据特征

- 音频是**一维时间序列**、采样率高（16k）、原始波形又长又不好直接用
- 标准做法是转成 **梅尔频谱图 (log-mel spectrogram)**：
  ```
  波形 → STFT → 梅尔滤波器组(mel filterbank) → 取 log → 时间×频率 的"图像"
  ```
- **为什么用 mel**：梅尔刻度贴合**人耳感知**（低频分辨细、高频粗）；把音频变成"图片"，就能用 CNN/Transformer（AST）来处理
- **两个特征性质影响设计**：
  - **多标签**：一段音里可同时有"说话+枪声" → 用 **sigmoid**（每类独立），不是 softmax
  - **时间局部性**：事件只在某几秒 → 要**分窗**（我们按 10s 窗口逐段打分），才能定位"第几秒响的"

---

## 5. 单模态如何得 0/1 + 多模态融合策略

### 单个模态如何得出 0/1

统一的一条链：

```
原始输入 → backbone → embedding → head(Linear) → logits → 激活 → 概率 → 阈值 → 0/1
```

- **激活**：多标签用 **sigmoid**（每类独立概率，一个片段可同时违规多类）；单标签才用 softmax
- **阈值**：概率 ≥ 0.5（可调）→ 判 1，否则 0
- **时序汇聚**：逐帧/逐窗的概率先 pool（我们用 max）成视频级，再阈值

例：视觉专家 `帧 → ResNet → 2048维 → Linear head → logits → sigmoid → violence 概率 → ≥0.5 → 1`。

### 多模态融合策略

**按位置（深度）分** —— 越早融合信息越全但越难训，越晚越可解释但可能丢信息：

融合有**两个维度**：**深度**（input→embedding→feature→decision→vote）和**表征方式**（joint 拼成一个共享向量 vs coordinated 各自编码再**学习对齐**，如 CLIP）。**五层**：

| # | 层 | 怎么融合 | 我们的实现 |
|---|---|---|---|
| ① | **Input / 数据层** | 编码器**之前**，原始信号混一起，一个模型同时感知+融合 | ✅ `early-fusion` |
| ② | **Embedding model-level (CLIP)** | 各自编码 → **学习一个模型对齐**到共享空间，比相似度 | ✅ `clip_screener.py` |
| ③ | **Feature 层** | 各自编码 → **拼接特征** → MLP / cross-attention | ✅ `embedding-mlp` / `cross_attention.py` |
| ④ | **Decision 层** | 各模态出分数 → 分类器合并 | ✅ `decision-tree` |
| ⑤ | **Late / Vote** | 各模态出 0/1 → 加权投票 | ✅ `mean-voting` |

> **第②层 vs 第③层**（关键）：② CLIP 用**学习到的对齐模型**把 embedding 对齐到共享空间（"model-level"），③ 只是**机械拼接**特征再喂 MLP。同样的输入（embedding），机制不同。
> **拼接已编码特征 ≠ early fusion**——那是 feature 层（③）；真 early fusion（①）在编码器之前。

**按方法分**：
- **拼接 + MLP**：可学跨模态交互，但要训练
- **加权投票（软/硬）**：免训练、可解释，权重靠先验（我们 `text>visual>audio`，因为音效易伪造）
- **决策树 / ensemble**：低维决策向量上可解释
- **Cross-Attention 深度融合**：音频/文本作 Q，视频帧作 K/V，"听到尖叫时关注哪几帧"
- **启发式熔断**：命中违禁词直接判违规、跳过融合（快车道）

**实验结论**（`docs/fusion.md` §5，合成数据）：**越早融合越好**（early-fusion F1 1.00 > embedding-mlp 0.988 > decision-tree 0.944）；投票基线 AUC 高但 F1 差（排序好、阈值差）。

---

## 一图串起来

```
视频 ──┬─ 抽帧(ffmpeg,1fps) ─ ResNet ─(GAP)→ 帧embedding ─ head → 视觉 0/1
       ├─ 抽音(ffmpeg,16k) ─ mel谱 ─ AST ─→ 声音事件      → 音频 0/1
       └─ ASR/OCR/评论/meta ─ 文本模型 ─→ 违规语义         → 文本 0/1
                                   │
        ①input / ②CLIP(embedding model-level) / ③feature-concat / ④decision / ⑤vote 任选
                                   ↓
                          最终审核判定 0/1 (+ 冲突标记)
```
