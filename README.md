# SentinelAI

**An end-to-end multimodal content moderation system combining computer vision, audio understanding, speech recognition, and vision-language models for scalable video safety detection.**

SentinelAI ingests video and analyzes it across every modality — frames, audio, and speech — then fuses the signals into a single, explainable moderation decision.

## Overview

Modern video content can hide unsafe material in any modality: a violent frame, a threatening line of speech, an unsafe sound, or context that only emerges when sight and language are reasoned about together. SentinelAI runs specialized experts per modality and fuses their outputs — first with simple late fusion (V1), then with a cross-attention deep-fusion model (V2).

## Architecture

```
                ┌──────────────────────────────────────┐
   Video  ──▶   │  Ingestion & Frame/Audio Extraction   │  (ffmpeg)
                └──────────────────────────────────────┘
                     │            │             │
              ┌──────▼─────┐ ┌────▼─────┐ ┌─────▼──────┐
              │  Visual    │ │  Audio   │ │   Text     │
              │  (ResNet/  │ │  (AST)   │ │ (XLM-R +   │
              │   CLIP)    │ │          │ │  lexicon)  │
              └──────┬─────┘ └────┬─────┘ └─────┬──────┘
                     │            │             │
                ┌────▼────────────▼─────────────▼────┐
                │  Fusion                            │
                │   V1: heuristics + weighted voting │
                │   V2: cross-attention (Q=audio/txt,│
                │        K,V=video frames)           │
                └────────────────┬───────────────────┘
                                 │
                          Moderation Verdict
                       (+ cross-modal conflict flag)
```

## Implemented

### V1 — expert models + late fusion

| Module | File | Notes |
|---|---|---|
| **Visual expert** | `sentinelai/visual_expert.py` | ResNet-50 / EfficientNetV2 frame embeddings + violation head |
| **Audio expert** | `sentinelai/audio_expert.py` | AST (AudioSet) — **zero-shot** gunshot / explosion / scream detection |
| **Text expert** | `sentinelai/text_expert.py` | Bilingual (zh+en) **lexical + semantic** transcript moderation |
| **Heuristics** | `sentinelai/fusion/heuristics.py` | High-risk keyword fast-path (short-circuit) |
| **Late fusion** | `sentinelai/fusion/fusion.py`, `mlp_fusion.py` | Weighted voting + MLP fusion over per-modality signals |
| **Evaluation** | `sentinelai/fusion/evaluate.py` | Precision/Recall/F1 + cross-modal conflict detection |

### V2 — zero-shot screening + deep fusion

| Module | File | Notes |
|---|---|---|
| **CLIP screener** | `sentinelai/clip_screener.py` | Zero-shot frame screening against a violating/safe prompt pool |
| **Cross-attention fusion** | `sentinelai/cross_attention.py` | Audio/text (Q) attend into the video frame sequence (K/V) |
| **Lightning training** | `sentinelai/train/` | Trains the cross-attention head (multi-label BCE, val AUC) |

### Experiments

**Fusion-depth comparison** (`docs/fusion.md`, `docs/experiment_1.md`) — fusing the
three experts at different pipeline depths (early → intermediate → late), on a
shared train/test split. Reproduce with `python -m sentinelai.fusion.compare`.

Fusion has **five positions**: ① input · ② embedding model-level (**CLIP**) ·
③ feature · ④ decision · ⑤ vote. This sweep benchmarks four; position ② (CLIP,
learned-alignment) lives in `clip_screener.py`, verified on real video.

| Fusion strategy | Position | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **early-fusion** | ① input | **1.000** | **1.000** | **1.000** | **1.000** |
| embedding-mlp | ③ feature | 0.983 | 0.993 | 0.988 | 0.999 |
| decision-tree | ④ decision | 0.936 | 0.952 | 0.944 | 0.956 |
| mean-voting (untrained) | ⑤ vote | 0.388 | 1.000 | 0.559 | 0.994 |

Earlier fusion wins (① > ③ > ④ > ⑤). **Early fusion** here is the real thing — all
modalities' raw signals entangled in one block, so the model must perceive *and*
fuse. The untrained voting baseline has a high AUC but poor F1 — good ranking, bad
fixed threshold. _Numbers are on synthetic features validating the framework;
real-data numbers await the feature-extraction pipeline._

### Verified on real data

All three experts run end-to-end on real Kinetics-400 video. A notable bad case: CLIP zero-shot flags **archery** as violence (bow ≈ weapon) — correctly caught by the cross-modal conflict detector (visual loud, audio calm → sent for human review).

## Project layout

```
sentinelai/
├── visual_expert.py     # V1 3.1 — CV expert
├── audio_expert.py      # V1 3.2 — audio expert (AST)
├── text_expert.py       # V1 3.3 — NLP expert
├── clip_screener.py     # V2 5   — CLIP zero-shot
├── cross_attention.py   # V2 6.2 — deep fusion
├── fusion/              # V1 4   — heuristics, voting, MLP, evaluation, comparison
├── train/               # V2 6.3 — PyTorch Lightning training loop
├── video.py             # ffmpeg / NVDEC capability detection
└── hardware.py          # device selection
tests/                   # per-module tests
docs/                    # spec.md, ROADMAP.MD, experiment_1.md
scripts/                 # GCP GPU-box management (start/stop/status)
```

## Getting started

```bash
pip install -e .                       # installs experts' deps (transformers, sklearn, lightning, ...)
                                       # PyTorch is installed separately (see scripts/setup-env.sh)
python -m pytest tests/                # run the test suite
python -m sentinelai.fusion.compare    # fusion-depth comparison (CPU)
python -m sentinelai.train.train       # train cross-attention head on synthetic data
```

## Roadmap status

| Stage | Status |
|---|---|
| V1 — experts (CV / audio / text) + late fusion | ✅ done |
| V2 — CLIP zero-shot screening | ✅ done |
| V2 — cross-attention deep fusion + training | ✅ done |
| V2 — VideoMAE / TimeSformer temporal features (6.1) | ⬜ todo |
| V3 — Qwen2-VL + QLoRA fine-tuning | ⬜ todo |
| Serving — vLLM / Triton, hybrid routing | ⬜ todo |

See `docs/ROADMAP.MD` for the full plan and `docs/spec.md` for the design.

## License

TBD
