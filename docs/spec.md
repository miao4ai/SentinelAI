# SentinelAI - 多模态视频内容安全审核系统

## 项目简介

SentinelAI 是一个面向短视频、直播回放、UGC（User Generated Content）场景的多模态内容安全审核系统。

项目目标：

* 构建完整的视频审核技术栈
* 从传统专家模型逐步演进到多模态大模型（VLM）
* 实现纯本地训练与推理
* 输出工业级审核流水线
* 支持结构化审核结果输出
* 支持高并发生产部署

最终系统能够对视频内容进行自动审核，并输出：

```json
{
  "is_violating": true,
  "category": "violence",
  "confidence": 0.97,
  "reason": "Multiple people are engaged in physical fighting with visible aggressive actions."
}
```

---

# 一、业务目标

## 审核类别

### NSFW

包括：

* 色情
* 裸露
* 性暗示
* 低俗内容

---

### Violence

包括：

* 打架斗殴
* 血腥画面
* 枪击
* 爆炸
* 虐待

---

### Hate Speech

包括：

* 仇恨言论
* 歧视内容
* 极端主义表达

---

### Dangerous Activity

包括：

* 危险驾驶
* 自残行为
* 毒品使用
* 非法活动展示

---

### Safe

正常内容

---

# 二、总体架构

```text
Video Input
     │
     ▼
Preprocessing Layer
 ├── Frame Extraction
 ├── Audio Extraction
 └── ASR
     │
     ▼
Feature Layer
 ├── Vision Expert
 ├── Audio Expert
 └── NLP Expert
     │
     ▼
Fusion Layer
 ├── V1 Rule Engine
 ├── V2 Cross Attention
 └── V3 Qwen2-VL
     │
     ▼
Decision Layer
     │
     ▼
JSON Output
```

---

# 三、技术栈

## 深度学习

* PyTorch
* PyTorch Lightning
* HuggingFace Transformers
* PEFT
* Accelerate

---

## 多模态模型

### V1

* ResNet50
* EfficientNetV2
* VGGish
* AST
* RoBERTa
* DeBERTa

### V2

* CLIP
* Chinese-CLIP
* VideoMAE
* TimeSformer

### V3

* Qwen2-VL-7B-Instruct

---

## 推理框架

* Triton Inference Server
* vLLM
* SGLang

---

## 数据处理

* FFmpeg
* OpenCV
* Faster-Whisper
* librosa

---

## API

* FastAPI
* Uvicorn

---

# 四、项目阶段规划

# Phase 1

## 基建与单模态融合基线

### 目标

建立传统审核系统基线。

---

## 1.1 标签体系设计

建立审核分类体系：

```python
[
    "safe",
    "nsfw",
    "violence",
    "hate_speech",
    "dangerous_activity"
]
```

---

## 1.2 本地环境

配置：

* CUDA
* cuDNN
* PyTorch

搭建：

* FFmpeg + NVDEC

实现：

GPU 视频解码

---

## 1.3 审核流水线

输入：

```text
video.mp4
```

输出：

```text
frames/
audio.wav
transcript.txt
```

---

# Phase 1.2

## 数据集准备

### 核心数据集

#### UCF-Crime

异常行为检测

#### XD-Violence

多模态暴力检测

---

### 辅助数据集

#### Kinetics-400

提取 Safe 样本

---

## 数据预处理

### 视频

抽帧：

```bash
ffmpeg -i video.mp4 -r 1 output/%06d.jpg
```

支持：

* Fixed FPS
* Key Frame

---

### 音频

提取：

```bash
ffmpeg -i video.mp4 audio.wav
```

---

### ASR

采用：

```text
faster-whisper
```

输出：

```json
[
  {
    "start": 0,
    "end": 5,
    "text": "..."
  }
]
```

---

# Phase 1.3

## 专家模型

### Vision Expert

模型：

* ResNet50
* EfficientNetV2

输出：

```python
frame_embedding
frame_logits
```

---

### Audio Expert

模型：

* VGGish
* AST

检测：

* 枪声
* 爆炸
* 尖叫

---

### NLP Expert

模型：

* RoBERTa
* DeBERTa

检测：

* 敏感词
* 仇恨言论
* 危险表达

---

# Phase 1.4

## V1 晚期融合

### Heuristic Layer

高危关键词直接熔断：

```python
if hit_high_risk_keyword:
    return violation
```

---

### Late Fusion

输入：

```python
vision_logits
audio_logits
nlp_logits
```

方案：

* Weighted Voting
* MLP Fusion

输出：

```python
final_prediction
```

---

### Evaluation

指标：

* Precision
* Recall
* F1

分析：

* False Positive
* False Negative
* Bad Case

---

# Phase 2

## 跨模态对齐与深度融合

---

## CLIP 零样本审核

Prompt 示例：

```text
A photo of people fighting.

A photo of people hugging.

A photo containing blood.

A photo of dangerous driving.
```

---

评分：

```python
cosine_similarity(
    image_embedding,
    prompt_embedding
)
```

---

## 时序建模

采用：

* VideoMAE
* TimeSformer

解决：

```text
单帧误判问题
```

例如：

做饭切菜

vs

持刀攻击

---

## Cross-Attention

目标：

让模型学习：

```text
声音发生时
应该关注哪些视频区域
```

结构：

```text
Audio/Text → Query

Video → Key

Video → Value
```

### 实现

模块：`sentinelai/cross_attention.py` → `CrossAttentionFusion`

输入 / 输出：

```text
video_seq (B, T, video_dim)   # 逐帧时序特征 (VideoMAE/TimeSformer) → K, V
guide     (B, [S,] guide_dim) # 音频/文本特征 → Q
      ↓
logits    (B, n_categories)   # 各类违规分数
attention (B, S, T)           # 每个 query 对每帧的注意力（可解释：模型看了哪几帧）
```

计算流程：

```text
video → Linear → K,V
guide → Linear → Q
MultiheadAttention(Q, K, V) → 残差+LayerNorm → FFN+残差+LayerNorm
      → 池化 → 分类头 → logits
```

要点：

* Q=引导模态（听到什么），K/V=视频时序（去哪帧找它）——"听到尖叫，聚焦相关帧"
* K/V 是**帧序列**（含时序），故能区分"切菜"与"砍人"（单帧做不到）
* 返回 `attention` 供解释：直接看模型关注了哪几帧
* 本模块只做融合层；VideoMAE 抽时序特征 (6.1) 仍为独立步骤（待做）

测试：`tests/test_cross_attention.py`（形状 / 2维guide / 注意力归一 / 概率范围，CPU 实测通过）

### 训练 (6.3)

模块：`sentinelai/train/` — PyTorch Lightning 训练循环

```text
train/datamodule.py  FusionFeaturesDataModule + 合成数据生成器（video/guide/label）
train/lit_module.py  LitCrossAttention: BCEWithLogitsLoss + AdamW + val AUC/acc
train/train.py       入口：python -m sentinelai.train.train
```

* 多标签 BCE-with-logits，每类独立 sigmoid
* 验证指标：val_loss / val_acc / 逐类平均 AUC
* 合成数据 CPU 实跑收敛：val_auc 0.49 → 1.00（证明训练循环正确）
* 接真实数据只需把 `make_synthetic_crossattn` 换成缓存的专家特征，训练代码不变

---

# Phase 3

## VLM 审核系统

---

## Qwen2-VL

模型：

```text
Qwen2-VL-7B-Instruct
```

量化：

* 4bit
* 8bit

---

## Prompt Engineering

System Prompt

```text
You are a professional content moderation expert.
```

要求：

1. 描述画面
2. 判断违规
3. 输出原因

---

## 输出格式

强制 JSON：

```json
{
  "is_violating": true,
  "category": "violence",
  "reason": "..."
}
```

---

## SFT

训练数据来源：

### Positive Samples

UCF-Crime

XD-Violence

---

### Hard Examples

V1

V2

误判案例

---

微调方案：

* LoRA
* QLoRA

框架：

* LLaMA-Factory

---

# Phase 4

## 工程化与部署

---

## Cascade 审核系统

```text
Input
  │
  ▼

V1
  │

High Confidence
  │
  └── Output

Low Confidence
  │
  ▼

V2
  │

Hard Case
  │
  ▼

Qwen2-VL
```

---

目标：

降低：

* GPU成本
* 推理延迟

提高：

* Recall
* Accuracy

---

## 高性能推理

### 小模型

Triton

---

### 大模型

vLLM

或者

SGLang

---

开启：

* Continuous Batching
* PagedAttention

---

# 五、评测体系

## 算法指标

分类指标：

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

审核业务优先级：

```text
Recall > Precision
```

尽量避免漏审。

---

## 工程指标

### Latency

视频输入

到

JSON输出

耗时

---

### Throughput

QPS

---

### GPU Memory

显存占用

---

# 六、最终交付成果

## 模型

* V1 Expert Fusion
* V2 Deep Fusion
* V3 Qwen2-VL

---

## 服务

* FastAPI REST API
* Triton Server
* vLLM Service

---

## 数据

* Processed Dataset
* Hard Example Dataset

---

## 文档

* Architecture Design
* Training Report
* Evaluation Report
* Deployment Guide

---

# 七、未来规划

## RAG + 法规知识库

引入：

* 平台审核规则
* 法律法规
* 地区政策

实现动态审核标准。

---

## 多语言支持

* 中文
* 英文
* 日文

---

## 实时直播审核

支持：

* Streaming ASR
* Streaming VLM
* Real-time Moderation

---

## Agentic Moderation

引入多智能体架构：

* Vision Agent
* Audio Agent
* Policy Agent
* Judge Agent

实现复杂违规场景自动协同审核。
