# SentinelAI

## Multimodal Content Safety Platform

SentinelAI is an end-to-end multimodal content moderation platform designed for large-scale Trust & Safety applications.

The system combines Computer Vision, Audio Understanding, Speech Recognition, Natural Language Processing, and Vision-Language Models to detect harmful content in videos and generate structured moderation decisions.

The project demonstrates the complete lifecycle of a modern AI system:

* Data Engineering
* Multimodal Machine Learning
* LLM / VLM Fine-tuning
* Distributed Inference
* Production Deployment
* Evaluation and Monitoring

This project is designed to showcase skills relevant to:

* TikTok Trust & Safety
* Google X
* YouTube Safety
* Meta Integrity
* OpenAI Safety
* Anthropic Safety Research

---

# Problem Statement

Modern content moderation systems must analyze multiple modalities simultaneously.

A single modality is often insufficient.

Examples:

* A harmless image combined with threatening speech.
* A normal conversation containing dangerous instructions.
* A cooking video containing knives versus a violent attack.

SentinelAI aims to build a scalable multimodal moderation system capable of understanding:

* Visual content
* Audio signals
* Spoken language
* Temporal actions
* Cross-modal interactions

---

# Moderation Categories

The initial moderation taxonomy includes:

```text
SAFE
VIOLENCE
NSFW
HATE_SPEECH
HARASSMENT
DANGEROUS_BEHAVIOR
SELF_HARM
ILLEGAL_ACTIVITY
```

Future categories:

```text
MISINFORMATION
EXTREMISM
SCAM
CHILD_SAFETY
```

---

# System Architecture

```text
                Video Upload
                      |
                      v
          +----------------------+
          | Video Preprocessing  |
          +----------------------+
                      |
       +--------------+--------------+
       |                             |
       v                             v

 Frame Extraction            Audio Extraction
       |                             |
       v                             v

 Visual Encoder             Audio Encoder
       |                             |
       +-------------+---------------+
                     |
                     v

            Speech Recognition
             (Faster Whisper)
                     |
                     v

                Text Encoder
                     |
                     v

          Multimodal Fusion Layer
                     |
                     v

           Moderation Decision
                     |
                     v

            Structured JSON
```

---

# Project Evolution

The project is divided into three model generations.

---

# V1 - Expert Models + Late Fusion

Goal:

Build independent experts for each modality.

## Visual Expert

Model options:

* ResNet50
* EfficientNetV2

Responsibilities:

* Frame classification
* Scene understanding
* Violence detection
* NSFW detection

Output:

```python
{
    "violence": 0.85,
    "nsfw": 0.03
}
```

---

## Audio Expert

Model options:

* VGGish
* AST (Audio Spectrogram Transformer)

Responsibilities:

* Gunshot detection
* Explosion detection
* Scream detection
* Abnormal audio events

Output:

```python
{
    "gunshot": 0.71,
    "scream": 0.62
}
```

---

## NLP Expert

Pipeline:

```text
Video
  ↓
Faster-Whisper
  ↓
Transcript
  ↓
DeBERTa
```

Responsibilities:

* Hate speech
* Threat detection
* Explicit language
* Dangerous instructions

---

## Heuristic Safety Rules

Examples:

High-risk keywords:

```text
bomb
shoot
kill
terrorist
```

Immediate escalation:

```python
if keyword_detected:
    return HIGH_RISK
```

Purpose:

Reduce unnecessary GPU inference.

---

## Late Fusion

Combine outputs from:

* Visual Model
* Audio Model
* NLP Model

Fusion methods:

* Weighted Voting
* Soft Voting
* MLP Classifier

Output:

```json
{
  "category": "violence",
  "confidence": 0.87
}
```

---

# V2 - Deep Multimodal Fusion

Goal:

Enable reasoning across modalities.

---

## CLIP-based Zero-Shot Moderation

Model:

```text
OpenAI CLIP
```

Prompt examples:

```text
A photo of people fighting

A photo of violent behavior

A photo of a family gathering

A photo of children playing
```

Applications:

* Zero-shot moderation
* Cold-start categories
* Hard example discovery

---

## Temporal Video Understanding

Problem:

Frame-based models ignore motion.

Solution:

* VideoMAE
* TimeSformer

Example:

Differentiate:

```text
Cooking with a knife

versus

Attacking someone with a knife
```

using temporal context.

---

## Cross-Modal Fusion

Inputs:

* Video Features
* Audio Features
* Text Features

Architecture:

```text
Transformer Encoder

Cross Attention

Feature Fusion
```

Output:

Unified multimodal representation.

---

## Temporal Localization

Instead of:

```text
Video → Violence
```

Predict:

```json
{
  "event": "violence",
  "start": 12.4,
  "end": 17.9
}
```

This mirrors real-world moderation systems used by large platforms.

---

# V3 - Vision Language Moderation

Goal:

Introduce reasoning and explainability.

---

## Foundation Model

Model:

```text
Qwen2.5-VL-7B-Instruct
```

Deployment:

* 4-bit quantization
* vLLM
* Continuous batching

---

## Structured Moderation Prompt

Example:

```text
You are a Trust and Safety reviewer.

Analyze the provided video frames.

Determine whether the content violates policy.

Return JSON only.
```

Expected output:

```json
{
  "is_violating": true,
  "category": "violence",
  "severity": "high",
  "confidence": 0.94,
  "reason": "Physical altercation detected"
}
```

---

## Fine-Tuning

Method:

```text
LoRA
QLoRA
```

Framework:

```text
LLaMA-Factory
```

Training data:

* UCF-Crime
* XD-Violence
* Hard Examples
* False Positives
* False Negatives

Objective:

Improve moderation reasoning while maintaining structured output.

---

# Production Architecture

## Cascade Moderation System

```text
Video
 |
 +--> V1 Expert Models
 |
 +--> V2 Fusion Model
 |
 +--> V3 Qwen-VL
```

Routing Logic:

```python
if confidence > 0.90:
    return result

if 0.40 <= confidence <= 0.60:
    route_to_vlm()

else:
    return safe
```

Benefits:

* Lower latency
* Lower GPU cost
* Higher throughput

---

## Inference Serving

Small Models:

```text
Triton Inference Server
```

VLM Serving:

```text
vLLM
```

Optimizations:

* PagedAttention
* KV Cache Reuse
* Continuous Batching

---

# Evaluation Framework

## Model Metrics

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

Content moderation prioritizes recall.

Missing harmful content is more expensive than over-filtering.

---

## System Metrics

* Throughput (QPS)
* End-to-End Latency
* GPU Utilization
* GPU Memory Consumption

---

## Business Metrics

* False Positive Rate
* False Negative Rate
* Human Review Rate
* Escalation Rate

---

# Future Roadmap

## Human-in-the-Loop Moderation

Moderator feedback collection.

Active learning.

Hard example mining.

---

## Policy-Aware Moderation

Integrate policy knowledge through RAG.

Examples:

* Platform policies
* Regional regulations
* Safety guidelines

---

## Agentic Safety Reviewer

Future architecture:

```text
Moderator Agent

Reasoning Agent

Policy Agent

Audit Agent
```

for explainable and auditable moderation workflows.

---

# Technology Stack

## Machine Learning

* PyTorch
* PyTorch Lightning
* Transformers
* PEFT
* Accelerate

## Vision

* OpenCV
* VideoMAE
* CLIP

## Audio

* AST
* Faster-Whisper

## NLP

* DeBERTa
* Qwen2.5-VL

## Serving

* Triton
* vLLM

## Backend

* FastAPI
* PostgreSQL
* Redis

## Infrastructure

* Docker
* Kubernetes

---

# Expected Outcomes

SentinelAI demonstrates:

* Multimodal AI Engineering
* Video Understanding
* VLM Fine-tuning
* Trust & Safety Systems
* Large-scale Inference Architecture
* Production Machine Learning

and serves as a portfolio project aligned with modern Trust & Safety engineering teams at TikTok, Google, Meta, YouTube, OpenAI, and Anthropic.
