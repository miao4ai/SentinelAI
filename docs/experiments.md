# 实验组织 (Experiment Plan)

如何把融合对比从**合成数据**升级为有说服力的**真实实验**。相关文档：融合分类见 [`fusion.md`](fusion.md)，各层实现见 [`fusion_layers.md`](fusion_layers.md)，首个合成结果见 [`fusion.md`](fusion.md) §5。

---

## 1. 实验要回答的问题

| Q | 问题 | 怎么判定 |
|---|---|---|
| Q1 | **哪一层融合最好**（①③④⑤，②另比） | P/R/F1/AUC 排序 |
| Q2 | **多模态是否值得**——融合能否超单模态 | 融合 vs 单模态基线 |
| Q3 | **各模态贡献** | 逐个去掉模态看掉多少 (ablation) |
| Q4 | **机制对比**：② CLIP 学习对齐 vs ③ 拼接 | 两者 P/R/AUC |
| Q5 | **Bad case**：跨模态冲突误判 | 冲突集上的 FP 率 |

---

## 2. 公平比较协议（每个实验都遵守）

- **同一 train/val/test split + 同 seed**——不同测试集比的是运气不是方法。`compare.py` 的 `FusionDataset.split()` 已强制共享 split。
- **统一指标**：Precision / Recall / F1 / **AUC**（阈值无关）+ 逐类别 P/R。
- **必设基线**（融合必须超过它们才有意义）：
  - `majority-class`（全判多数类）
  - `random`
  - **单模态专家**（只用视觉 / 只用音频 / 只用文本）
- **报告**：不只报单阈值 P/R——AUC 才反映排序质量（见 `fusion.md` §5 里投票基线 F1 差但 AUC 高的教训）。

---

## 3. 数据现状与各自能支撑的实验

| 数据集 | 格式 | 模态 | 标签 | 支撑 |
|---|---|---|---|---|
| **UCF-Crime** | I3D 特征 `.npy` | 视觉 | 视频级 | 单模态视觉基线 |
| **XD-Violence** | I3D(视觉) `.npy`；官方另有 **VGGish(音频)** | 视觉 (+音频) | 视频级（文件名带类别） | **真实 2 模态融合** |
| **Kinetics-400** | 原始 `.mp4` | 视觉(+音频) | 动作类（非暴力） | 跑通专家 / 取 safe 样本 / CLIP 帧 |

**关键**：核心暴力集是**预提取 I3D 特征**（非原始视频），所以"跑三专家抽特征"的路线只能在**原始视频**上做；而 XD 的 **I3D + VGGish 配对特征**让我们**不碰原始视频**就能做真实 2 模态融合。

---

## 4. 分阶段计划

| 阶段 | 目标 | 数据 | 需要 GPU | 状态 |
|---|---|---|---|---|
| **0** | 合成框架验证（①③④⑤对比） | 合成 | 否 | ✅ 完成（见 `fusion.md` §5） |
| **0.5** | **训练真实 nn.Module**（第四章 MLP + early fusion） | 合成 | 否 | ✅ 完成（见下 §6） |
| **1** | 单模态真实基线 | UCF/XD I3D | 否 | 🟢 现在可做 |
| **2** | **真实 2 模态融合对比** | XD I3D + AST 音频 | 是* | ✅ 完成（见下 §7） |
| **3** | 抽特征管线（跑三专家→缓存） | XD 原始视频子集 | 是 | 🟡 需下原视频 |
| **4** | ② CLIP 真实指标 + ③ cross-attn 训练 | Kinetics/XD 帧 | 是 | 🟡 |

---

## 5. 各实验规格

### 实验 1 — 单模态真实基线（视觉）
- **数据**：XD-Violence（或 UCF-Crime）I3D 特征，视频级标签。
- **做法**：I3D 特征（ten/five-crop 先对 crop 维平均，时序按 §决策层的 max 池化到视频级）→ 一个线性/MLP 头 → 违规概率。
- **产出**：视觉单模态 P/R/F1/AUC，作为 Q2 的**基线**（融合要超它）。
- **代码**：新增 `scripts/exp1_visual_baseline.py`（加载 `.npy` → 池化 → sklearn 分类）。

### 实验 2 — 真实 2 模态融合层级对比 ⭐
- **数据**：XD-Violence，每 clip 配对 **I3D(视觉) + VGGish(音频)** + 标签。
- **做法**：把两模态特征当作各层输入，跑现有 `compare.py` 框架：
  - ③ feature：concat(I3D, VGGish) → MLP
  - ④ decision：各自出分数 → 树
  - ⑤ vote：各自 0/1 → 加权投票
  - ①/② 不适用（无原始信号 / 无 CLIP 对齐）
- **产出**：**合成对比（`fusion.md` §5）的真实版**——真实 2 模态的层级 P/R/AUC 对比 + 对单模态基线的增益（Q1、Q2）。
- **前置**：下载 XD 的 VGGish 特征（官方提供，不大）。

### 实验 3 — 抽特征管线 + 3 模态融合
- **数据**：XD-Violence **原始视频**子集（~几百段，含音频）。
- **做法**：跑三专家（视觉/音频 AST/ASR→文本）→ 缓存每模态 embedding+scores → 喂 `compare.py`（含 ① 需原始信号的近似）。
- **产出**：真实 **3 模态**层级对比；ablation（Q3）。
- **需要**：开 GPU、下原视频、写 `scripts/extract_features.py`。

### 实验 4 — ② CLIP 与 ③ cross-attention
- **② CLIP**：在 XD/Kinetics 抽帧上跑 `clip_screener`，报零样本 P/R；与 ③ 拼接对比（Q4）。
- **③ cross-attn**：用 VideoMAE/I3D 时序特征训 `CrossAttentionFusion`（6.1+6.3），对比 concat-MLP。

### 贯穿实验 — Bad case / 冲突分析（Q5）
- 用 `fusion.evaluate.cross_modal_conflict` 挑出"视觉响音频静"等冲突 clip，统计这些上的误判率；对比加/不加启发式熔断。

---

## 6. 已完成结果（阶段 0.5）：三层融合训练（①②③）

> 日期：2026-08-21（②于同日补充）｜ 代码：`sentinelai/train/train_fusion.py` ｜ 复现：`python -m sentinelai.train.train_fusion`

在**同一份合成 token 数据**上用 PyTorch Lightning 训练**三个**真实 `nn.Module`，一份数据**三种视角**喂三个模型，所以这一跑也顺便**公平对比**了 ①early / ②coordinated / ③late。

### 6.1 数据是怎么生成的（`make_synthetic_tokens`）

设计目标：让**标签信号只藏在少数几个 token 里**，这样"能定位到那几个 token"的模型（early fusion 的注意力）才占优——否则对比没意义。

每个样本：
```python
# 1. 多标签：每个规范类别独立以 p=0.3 的概率激活
labels = (rng.random((N, 3)) < 0.3)                      # (N, 3) multi-hot

# 2. 每个模态生成 token 序列（视觉8×48 / 音频5×24 / 文本6×24 维）
for modality, dim in MODALITY_DIMS.items():
    x = rng.normal(size=(N, T_m, dim))                  # 纯噪声底
    signatures = rng.normal(size=(3, dim))              # 每个类别一个“签名向量”
    # 3. 某类别激活时，把它的签名种进【随机一个】token
    for i, c: 
        if labels[i, c]:
            t = random token index
            x[i, t] += signatures[c] * 2.0              # 信号只在这一个 token 上
```
关键：**信号是稀疏的、局部的**（只在某几个 token 上），不是均匀铺在整段里。

**三种视角**（同一份 `tokens` + `labels`）：
- **① early fusion**：直接用**原始 token 序列** `{视觉:(N,8,48), 音频:(N,5,24), 文本:(N,6,24)}`，让注意力自己去找信号 token。
- **② coordinated（`per_modality_embeddings`）**：每个模态 token mean 池化成一个向量，但**各模态分开**（不拼接）→ 各自独立 encoder 对齐到共享空间。
- **③ late fusion（`pooled_features`）**：每个模态池化后**拼接** → `(N, 96)`。池化会**稀释**那个藏信号的 token。

### 6.2 训练怎么实现的

共享一个 Lightning 基类 `_LitFusionBase`（BCE-with-logits 多标签 + AdamW；验证集算 loss / acc / 逐类平均 AUC）：

| 子类 | 模型 | 一个 batch 怎么算 logits |
|---|---|---|
| `LitEarlyFusion`（①） | `JointFusionTransformer(d_model=128, 2层)` | token 序列 → 联合 Transformer → `[CLS]` → logits |
| `LitCoordinatedFusion`（②） | `CoordinatedFusion(d_model=128)` | 各模态**分开**池化 embedding → 独立 encoder → 对齐共享空间 → 比类别原型相似度 |
| `LitMLPFusion`（③/第四章） | `MLPFusion(input_dim=96)` | 池化拼接特征 → MLP → logits |

- **同一 seed、同一 80/20 train/val split** → 差异只来自"融合位置 + 模型"，不是数据。
- `Trainer(accelerator="auto")`：有 GPU 用 GPU，没有就 CPU（本轮 CPU 跑完）。
- Dict-token 的 batch 靠 `_TokenDataset` + PyTorch 默认 collate（自动把每个模态的张量 stack）。

### 6.3 结果

| 模型 | 位置 | 输入 | val_auc | 终态 val_loss | 收敛 |
|---|---|---|---|---|---|
| `JointFusionTransformer`（early） | ① input | 原始 token 序列 | 0.35 → **1.00** | **0.0007** | **~1 epoch** |
| `CoordinatedFusion`（CLIP-style） | ② embedding model-level | 各模态分开 embedding | 0.59 → **1.00** | 0.0034 | ~1 epoch |
| `MLPFusion`（第四章） | ③ feature | 均值池化 + 拼接 | 0.59 → **1.00** | 0.0071 | ~5 epoch |

**结论**：
- 三个训练循环都正确（反向传播、优化器、指标）——真实数据接上只需把 `make_synthetic_tokens` 换成真实 token 化特征。
- **终态 loss：① early (0.0007) < ② coordinated (0.0034) < ③ late (0.0071)**，越早越低。early 能**注意到藏信号的那几个 token**；late 先池化、稀释了信号；coordinated 居中（各模态对齐到共享空间，比拼接更结构化）。印证"越早融合信息越全"。
- ⚠️ 合成数据，验证的是**训练框架 + 融合深度趋势**，不是生产指标。

## 7. 真实结果（阶段 2）：XD-Violence 全融合位置对比 ⭐

> 日期：2026-08-22 ｜ 数据：XD-Violence **I3D 视觉 + AST 音频**（预提取特征，来自 HF）｜ 脚本：`scripts/exp2_all_positions.py`

**第一个真实多模态结果。** 把预提取的 I3D 视觉特征和 AST 音频特征**按 `video_id` 配对**（788 个片段两模态都有），各池化到视频级；`label_A`=正常、其余=暴力（48% 暴力，基本平衡）。两源的 train/test 划分不一致，故在配对片段上做**自己的分层 80/20 划分**（train 630 / test 158，seed=0）。

各位置的模型：① `JointFusionTransformer`（视觉 I3D 时序 token + 音频 AST snippet token，跨模态注意力）｜② `CoordinatedFusion`（池化 embedding 各自编码→对齐共享空间）｜③ 池化拼接→LogisticRegression｜⑤ 各模态 LR 概率平均。

| 模型 | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| 视觉单模态 (I3D) | 0.917 | 0.868 | 0.892 | 0.960 |
| 音频单模态 (AST) | 0.929 | 0.855 | 0.890 | 0.969 |
| ① early（joint transformer） | 0.896 | 0.908 | 0.902 | **0.977** |
| ② coordinated（CLIP-style） | 0.932 | 0.895 | 0.913 | **0.977** |
| ③ feature-concat | 0.932 | 0.895 | 0.913 | 0.965 |
| **⑤ late-fusion (avg)** | **0.945** | 0.908 | **0.926** | **0.978** |

**关键发现（真实数据）：**
1. **多模态值得**：所有融合 F1 (0.90–0.926) 都 > 单模态 (0.89)。
2. **⑤ 晚期融合(avg) F1 最高 (0.926)**——真实数据上简单概率平均最稳、最能打。
3. **①early / ②coordinated AUC 最高 (0.977)**，排序质量顶尖，但 0.5 阈值下 F1 略低于 ⑤。
4. **② coordinated > ③ concat**：F1 相同 (0.913) 但 AUC 更高 (0.977 vs 0.965)——**对齐共享空间比机械拼接排序更好**（印证 CLIP 式 coordinated 与 concat 的机制差异）。
5. **与合成数据的反差**：合成里"越早严格越好"（① > ② > ③），**真实数据不成立**——各位置很接近，晚期融合 F1 领先、early/coordinated AUC 领先。真实场景里融合位置的差距远小于合成 toy，**简单晚期融合往往极具竞争力**。

> *不严格需要 GPU：sklearn 部分 CPU 即可；①② 的 torch 模型在 GPU 上跑（也可 CPU）。

## 8. 建议起点

**先做实验 1 + 2**（用现有数据 + 一个小下载，不用 GPU）：
1. XD I3D → 视觉单模态真实基线
2. 下 XD VGGish → 视觉+音频真实 2 模态层级对比

这把合成对比（`fusion.md` §5）升级为"真实"，且成本最低。跑完再决定是否上原始视频做 3 模态（实验 3）。

## 9. 可复现

- 每个实验固定 seed、记录 split、把结果表写进 `docs/experiment_N.md`。
- 特征缓存存到 `data/cache/`（已 gitignore），脚本存 `scripts/`。
- 数值对比统一走 `python -m sentinelai.fusion.compare`（真实数据只需换 `FusionDataset` 的来源）。
