# 五层融合：各层具体如何实现

对照代码，逐层讲清楚**每一层怎么融合、代码在哪、输入输出是什么、怎么调用**。五层的整体分类和取舍见 [`fusion.md`](fusion.md)，这里只讲**实现**。

五层一览（最早 → 最晚）：

| # | 层 | 表征 | 代码 |
|---|---|---|---|
| ① | Input / 数据层 | joint | `fusion/compare.py` (`early-fusion`) + `fusion/synthetic.py` |
| ② | Embedding model-level (CLIP) | coordinated | `clip_screener.py` |
| ③ | Feature 层 | joint | `fusion/mlp_fusion.py`、`cross_attention.py`、`fusion/compare.py` (`embedding-mlp`) |
| ④ | Decision 层 | — | `fusion/signals.py` + `fusion/compare.py` (`decision-tree`) |
| ⑤ | Late / Vote | — | `fusion/fusion.py` (`WeightedVotingFusion`) + heuristics |

---

## ① Input / 数据层 —— `early-fusion`

**思想**：在**任何模态编码器之前**，把三个模态的原始信号混进一个联合块，让**一个模型同时学"感知 + 融合"**。这是真正的 early fusion——最难，但信息最全。

**代码**：`sentinelai/fusion/synthetic.py`（造联合输入）+ `sentinelai/fusion/compare.py` 里的 `early-fusion` 策略。

**怎么实现的**（合成版，`synthetic.py`）：
```python
# 把三个模态的原始信号拼成一个大向量，再过一个共享的非线性把它们“搅在一起”
joint_signal = np.concatenate(raw_signals, axis=1)          # 所有模态混在一起
mix = rng.normal(size=(joint_signal.shape[1], 256))
X_early = np.tanh(joint_signal @ mix) + noise               # (N, 256) 纠缠、不可按模态拆分
```
关键：`tanh` 把模态纠缠在一起，**没法把某个模态单独读出来**——所以模型必须联合处理。

**融合模型**：一个**更深**的 MLP（浅模型解不开纠缠）：
```python
MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1200)
```

- **输入**：`X_early` (N, 256)  **输出**：0/1 + 概率
- 这个合成版只是用一个 tanh 块**模拟**"混在一起、不可拆"，真实架构见下。

**真实架构**（`sentinelai/early_fusion.py`，`JointFusionTransformer`）：token 化 + 一个联合 Transformer。
```
image → patches ┐
audio → spec块  ┼─ +模态类型emb → [CLS]+所有token → Transformer → joint embedding → logits
text  → tokens  ┘     (一条序列)          ↑ 从第1层就跨模态注意力
```
- **每个模态只做一件轻活**：一个 `Linear` 投影，把异构 token（像素/波形/词）统一到共享宽度 `d_model`
- 加**模态类型 embedding**（告诉模型这 token 是图/音/文），拼成一条序列，前面放一个 `[CLS]`
- 喂**一个共享 Transformer**——每层都跨模态注意力，融合发生在编码内部
- `[CLS]` 的输出 = **共同的 joint embedding**（一个共享表示，不是三个对齐的——那是 ②CLIP）
- **输入**：每模态 `(B, T_m, dim_m)` 的 token 序列  **输出**：`logits` (B,C) + `joint_embedding` (B, d_model)
- **状态**：结构已实现、CPU 形状测试通过；真训练需要真实 token 化的多模态数据（暂未训）

---

## ② Embedding model-level (CLIP) —— `clip_screener.py`

**思想**：各模态**各自有编码器**，但用**对比学习训练一个模型，把它们对齐到同一个 embedding 空间**，然后**比相似度**来融合。这是 coordinated representation——"model-level"指对齐是模型学出来的，不是拼接。

**代码**：`sentinelai/clip_screener.py`（`ClipScreener`），底座 `openai/clip-vit-base-patch32`。

**怎么实现的**：
1. **Prompt 池**（`build_prompts`）：违规 prompt + 安全对照 prompt（"fighting" vs "hugging"）
2. **一次性编码 prompt**（`__init__`）：CLIP 文本编码器把 prompt 编码好缓存
3. **打分**（`score_frames`）：
   ```python
   # CLIP 内部：图像/文本各自投影→L2 归一化→温度缩放→相似度
   logits_per_image = model(pixel_values, input_ids)     # 帧 vs 各 prompt 的余弦相似度
   probs = softmax(logits_per_image)                     # 帧在 prompt 池上的分布
   ```
4. **聚合**（`_aggregate_prompt_probs`）：违规 prompt 的概率质量 = 违规分

- **输入**：视频帧 + prompt 池  **输出**：逐帧 `ClipFrameScore`（violation_prob、各类分、最像的 prompt）
- **调用**：`ClipScreener().score_frames(frames)`
- **状态**：✅ 已在真实 Kinetics 视频验证（archery 误判被冲突检测兜住）——比合成 sweep 更硬

**可训练版**（`sentinelai/coordinated_fusion.py`，`CoordinatedFusion`）——上面 `clip_screener` 用的是**预训练** CLIP 零样本；要在自己数据上**训**这个机制，用这个模块：
```
各模态 ─▶ [各自独立的 encoder] ─▶ L2归一化 ─┐
                                           ├─ 余弦相似度 × 温度 ─▶ 平均 ─▶ logits
学习的“类别原型” ─▶ L2归一化 ────────────────┘   (原型 = CLIP 里 text prompt 的角色)
```
- **各模态独立 encoder**（这才叫 coordinated，不是一个联合编码器）→ 投影到共享空间
- 学习一组**类别原型**（充当 CLIP 里的文本 anchor），各模态 embedding 和原型比**余弦相似度**（带可学习温度），再跨模态平均
- **输入**：每模态一个 embedding 向量 `{m: (B, dim_m)}`  **输出**：`logits (B, C)`
- **状态**：✅ 结构实现、训练收敛（见 `experiments.md` §6）

- **和③的区别**：② 各模态**独立编码→对齐到共享空间→比相似度**；③ **机械拼接**特征再喂 MLP

---

## ③ Feature 层 —— 拼接 + MLP / cross-attention

**思想**：各模态**各自编码好**，再把特征**拼接**融合。两种实现，一浅一深。

### 3a. 拼接 + MLP（浅，joint）

**代码**：`sentinelai/fusion/mlp_fusion.py`（`MLPFusion`），以及 sweep 里的 `embedding-mlp`。
```python
# [cv_emb | audio_emb | text_emb] → 一个 MLP
nn.Sequential(
    nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, num_categories),        # 裸 logits（调用方选 sigmoid/softmax）
)
```
sweep 版：`MLPClassifier(hidden_layer_sizes=(128,))` 跑在拼接的编码特征 `X_embedding` (N, 96) 上。

### 3b. Cross-Attention（深，joint）

**代码**：`sentinelai/cross_attention.py`（`CrossAttentionFusion`）。
```python
# 音频/文本作 Query，视频帧序列作 Key/Value —— “听到尖叫时关注哪几帧”
attn_out, attn_w = MultiheadAttention(Q=guide, K=video, V=video)
# 残差+LayerNorm+FFN → 池化 → 分类头
```
- **输入**：`video_seq` (B,T,video_dim) + `guide` (B,guide_dim)
- **输出**：`logits` (B,C) + `attention` (B,S,T)（可解释：看了哪几帧）
- K/V 是**帧序列**（含时序）→ 能区分"切菜"和"砍人"

---

## ④ Decision 层 —— `decision-tree`

**思想**：每个模态**先出最终的类别分数**，再把这些分数喂给一个分类器合并。

**前置：统一到规范类别**（`sentinelai/fusion/signals.py`）
```python
# 各专家类别 → 规范类别；跨帧/窗口用 max 池化到视频级
gunshot/explosion/scream → violence      # 音频
toxic/hate → hate_speech, sexual → nsfw  # 文本
# 用 max：10 分钟里 2 秒枪声也能把 violence 顶到 ~1.0（mean 会稀释）
```
得到每模态一个 `ModalitySignal`（3 个规范类别的分数）。

**融合模型**：把 3 模态 × 3 类别 = 9 维分数向量喂决策树：
```python
DecisionTreeClassifier(max_depth=5)       # 低维、可解释
```
- **输入**：`X_decision` (N, 9)  **输出**：0/1
- 树输出的概率是"阶梯状"的，排序（AUC）不如 MLP 平滑

---

## ⑤ Late / Vote 层 —— `WeightedVotingFusion` + 启发式熔断

**思想**：每个模态**各自出结论**，再投票合并。免训练、可解释。

**代码**：`sentinelai/fusion/fusion.py`（`WeightedVotingFusion`、`V1Moderator`）。

**加权投票**：
```python
DEFAULT_WEIGHTS = {"text": 1.0, "visual": 0.8, "audio": 0.6}
# text 最高（干净转写上语义模型可靠），audio 最低（音效易伪造）
# soft: 加权平均分数；hard: 各模态先 0/1 投票再加权
# 只对“在场”模态归一化 → 没音轨不会把分数拖低
```

**启发式熔断（快车道，4.1）**：`sentinelai/fusion/heuristics.py`
```python
if 文本命中违禁词:  # 铁证
    return 违规(confidence=0.99)  # 直接判，跳过融合
```

**总指挥** `V1Moderator.moderate()`：先熔断，没命中再走加权投票。
- **输入**：三专家原始输出  **输出**：`FusedVerdict`（is_violating、category、confidence、reason、source）
- sweep 里的 `mean-voting` 是它的极简基线：`max(分数) ≥ 0.5` 就判违规

---

## 跑一遍看对比

```bash
# ①③④⑤ 合成数据数值对比（CPU，无需 GPU）
python -m sentinelai.fusion.compare

# ② CLIP 在真实帧上（需要装 transformers + 帧图）
python -c "from sentinelai.clip_screener import ClipScreener; print(ClipScreener().score_frames(['f.jpg']))"
```

## 一句话记忆

- ① 编码前混在一起，一个模型**感知+融合**
- ② 各自编码，**学一个模型对齐**（CLIP），比相似度
- ③ 各自编码，**拼接**特征喂 MLP（或 cross-attention）
- ④ 各自出**分数**，分类器合并
- ⑤ 各自出 **0/1**，加权投票（+ 违禁词熔断）
