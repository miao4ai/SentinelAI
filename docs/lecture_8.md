# Lecture 8 总结 — CLIP 零样本审核机制（第五章）

承接 Lecture 7（CLIP / 对比学习**原理**）。上一讲讲 CLIP 怎么**训练**出图文共享空间；
这一讲讲怎么**用**它——不训练、只写几句 prompt，就做违规**初筛**。对应代码
`sentinelai/clip_screener.py`，真实结果见 `docs/experiments.md §9`。

---

## 1. 为什么能"零样本"：共享空间的一次免费午餐

V1 视觉专家的困境：ImageNet 的 1000 类里**没有"暴力"这个类**，想判暴力就得自己
标注数据、训练一个分类头。

CLIP 把这一步整个绕过。它训练完就有一个性质（Lecture 7）：**图像**和**描述它的
句子**被放到**同一个向量空间**里、语义相近就距离相近。于是：

> "这帧像不像暴力" ≈ "这帧的 embedding 离句子 *'people fighting'* 的 embedding 近不近"

判定被转成一次**相似度查询**，不需要任何训练。这就是 zero-shot：目标类别在训练里
从没出现过，靠**自然语言描述**即时定义。**写 prompt = 定义类别。**

---

## 2. Prompt 池设计（5.2）

```
违规 prompt（每类多写几句同义改写）        安全对照 prompt（关键！）
  violence: "people physically fighting"     "people hugging"
            "violently attacking another"     "people talking calmly"
            "threatening with a weapon"       "food being prepared in a kitchen"
  nsfw:     "explicit sexual content" ...     "a normal everyday photo" ...
```

两个设计点：

- **每类多写几句**：同义改写让信号更稳，不押宝在单一句子的措辞上。
- **必须有"安全对照"prompt** —— 这是最容易漏、也最关键的一步。softmax 是**相对**的：
  如果池子里**只有违规 prompt**，那正常帧也只能把概率质量分给违规 prompt（矬子里
  拔将军），于是**一切都"违规"**。安全 prompt 的作用是**吸走正常帧的质量**，把它们的
  违规分数压下去。
  - 典型例子：厨房切菜 vs 持刀威胁。加一句 *"food being prepared in a kitchen"*，
    就把"切菜"的质量从"weapon"prompt 那里拉回来，减少误报。

---

## 3. 打分：余弦相似度 → softmax → 按类聚合（5.3）

```
帧 ─▶ image encoder ─▶ 投影 ─▶ L2 归一化 ─┐
                                          ├─▶ 余弦相似度(点积) ─▶ ×温度 ─▶ softmax
prompt 池 ─▶ text encoder ─▶ 投影 ─▶ 归一化 ┘        (在所有 prompt 上得到一个分布)
```

- **余弦相似度**：两边都 L2 归一化后做点积，就是夹角余弦（只看方向、不看长度）。
- **×温度再 softmax**：乘上 CLIP 训练时学到的 `logit_scale`，再 softmax 成 prompt 上
  的概率分布。温度控制分布的"尖锐度"。
- **按类聚合**：某一类的分数 = 该类**所有违规 prompt 的质量之和**（用 sum 不是 max：
  同义改写是"或"的关系，帧在两句 violence 上各 0.3 就是 0.6 暴力）。
- **违规总分** = 所有违规 prompt 质量之和 = `1 − 安全 prompt 质量之和`。

**工程要点**：

- prompt 池是**固定的**，所以只编码一次；之后每帧只要一次 image-encode + 一个矩阵乘，
  很便宜。
- 实现上走 CLIP 的完整 forward 取 `logits_per_image`（内部帮你做好投影+归一化+温度，
  且**跨 transformers 版本稳定**），不要用 `get_text_features`——它在 5.x 改了返回类型。

---

## 4. 帧 → 片段：为什么用 max 池化

一个片段抽 K 帧，每帧一个违规分，**片段分 = 帧分的最大值**。

- 初筛逻辑是"**任一帧像暴力，就该标记送审**"，所以取 max。
- 若用 mean，一个几帧的短暴力镜头会被大量正常帧**稀释**掉，漏筛。

---

## 5. 怎么读结果 + 校准陷阱（连到 exp5）

真实 XD（299 片段、零训练）：**AUC 0.825**、**precision@top-10% 0.931**。但——

- 正常片段的违规分**中位也有 0.87**（暴力 0.97）。**绝对分数没校准**：CLIP 总倾向给
  违规 prompt 也分一杯羹，安全 prompt 没能吸走全部质量。
- 所以**别用 0.5 当阈值**（会几乎全标记：recall 1.0、precision 0.52）。它是一个
  **排序信号**——按分数排序、取 top-k 或用高阈值(如 0.91)送审。
- **定位：初筛，不是终判**。0.825 明显低于训练好的融合（① early **0.951**）。
  CLIP 零样本 = **便宜的第一道粗筛**，把最可疑的一批快速挑出来（top-10% 里 93% 真
  暴力），只把高分片段送给贵的下游专家/融合做**终判**。两者是**流水线上下游**，不是
  替代。

---

## 6. 中文 / 其他语言

prompt 的**语言和视频无关**（视频只是像素）。处理中文内容时，把模型换成
Chinese-CLIP（`OFA-Sys/chinese-clip-vit-base-patch16`）+ 中文 prompt 即可；
`clip_screener.py` 的 `--model` / `model_name` 已留好接口。

---

## 一句话总结

CLIP 把"这帧违规吗"变成"这帧离哪句描述更近"，于是**写 prompt 就等于定义类别**，
不训练就能做又快又便宜的违规**初筛**；代价是**分数只可比、不可信**（要用排序或高阈值），
终判仍交给训练过的融合。
