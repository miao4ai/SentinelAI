# SentinelAI

**An end-to-end multimodal content moderation system combining computer vision, audio understanding, speech recognition, and vision-language models for scalable video safety detection.**

SentinelAI ingests video and analyzes it across every modality — frames, audio, and speech — then fuses the signals with vision-language reasoning to flag unsafe or policy-violating content at scale.

## Overview

Modern video content can hide unsafe material in any modality: a violent frame, a threatening line of speech, an unsafe sound, or context that only emerges when sight and language are reasoned about together. SentinelAI tackles this by running specialized models per modality and fusing their outputs into a single, explainable moderation decision.

## Capabilities

- **Computer Vision** — frame-level detection of unsafe visual content (violence, explicit imagery, graphic material).
- **Audio Understanding** — classification of non-speech audio events and acoustic safety signals.
- **Speech Recognition (ASR)** — transcription of spoken audio for downstream text-based policy analysis.
- **Vision-Language Models (VLM)** — joint reasoning over frames and language for context-aware moderation that single-modality models miss.
- **Multimodal Fusion** — combines per-modality signals into a unified, explainable safety verdict.
- **Scalable Pipeline** — designed for batch and streaming video processing at scale.

## Architecture

```
                ┌──────────────────────────────────────┐
   Video  ──▶   │  Ingestion & Frame/Audio Extraction   │
                └──────────────────────────────────────┘
                     │            │             │
              ┌──────▼─────┐ ┌────▼─────┐ ┌─────▼──────┐
              │  Computer  │ │  Audio   │ │   Speech   │
              │   Vision   │ │ Underst. │ │ Recog.(ASR)│
              └──────┬─────┘ └────┬─────┘ └─────┬──────┘
                     │            │             │
                ┌────▼────────────▼─────────────▼────┐
                │   Vision-Language Reasoning (VLM)   │
                └────────────────┬───────────────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Fusion & Policy │
                        │   Decisioning    │
                        └────────┬─────────┘
                                 │
                          Moderation Verdict
```

## Status

🚧 Early development. Core pipeline and per-modality modules are being built out.

## License

TBD
