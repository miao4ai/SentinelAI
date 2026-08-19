# Experiment 1 — Fusion depth comparison (V1)

**Question:** where in the pipeline should we fuse the three experts (visual /
audio / text), and with what model? Of the five fusion positions (see
`docs/fusion.md`) this sweep benchmarks four — ① input, ③ feature, ④ decision,
⑤ vote — and measures Precision / Recall / F1 / AUC. (Position ② is CLIP-style
*embedding model-level* fusion, a learned-alignment mechanism implemented in
`clip_screener.py` and verified on real video, not in this synthetic sweep.)

**Status:** run on **synthetic** multimodal features
(`sentinelai/fusion/synthetic.py`). The numbers validate the framework and the
depth trade-offs; real-data numbers await the feature-extraction pipeline over a
labelled video set. Reproduce with:

```bash
python -m sentinelai.fusion.compare
```

## Strategies

| Fusion | Position | Structure |
|---|---|---|
| **early-fusion** | ① input | ONE joint block of all modalities' raw signal (256-d) → Linear → ReLU(256) → ReLU(128) → 2 |
| **embedding-mlp** | ③ feature | concat 3 experts' encoded features (96-d) → Linear → ReLU(128) → Linear → 2 |
| **decision-tree** | ④ decision | concat 3 experts' final category scores (9-d) → `DecisionTree(max_depth=5)` |
| **mean-voting** (baseline) | ⑤ vote | untrained: concat experts' final scores, flag if `max >= 0.5` |

All strategies are trained/evaluated on the **same** seeded train/test split, so
differences reflect the fusion position + model, not the data. (The CLIP-style
*coordinated* representation is a different mechanism — see `docs/fusion.md` — not
part of this numeric sweep.)

## Results (synthetic test set)

| Fusion | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **early-fusion** | **1.000** | **1.000** | **1.000** | **1.000** |
| embedding-mlp | 0.983 | 0.993 | 0.988 | 0.999 |
| decision-tree | 0.936 | 0.952 | 0.944 | 0.956 |
| mean-voting (untrained) | 0.388 | 1.000 | 0.559 | 0.994 |

## Takeaways

1. **Earlier fusion wins.** early-fusion (F1 1.00) > embedding-mlp (0.988) >
   decision-tree (0.944). Earlier positions retain signal the downstream levels
   discard. Early fusion needs a deeper model because its input block entangles
   every modality — the model must perceive *and* fuse at once.

2. **AUC and F1 can disagree — measure both.** mean-voting has a poor F1 (0.559,
   precision 0.388: it over-flags) yet a high AUC (0.994). Its *ranking* of
   samples is good; only the fixed 0.5 threshold is bad. Judging a fusion by a
   single-threshold P/R alone is misleading.

3. **Decision trees rank worse than MLPs.** The tree's coarse, step-wise
   probabilities give a lower AUC (0.956) than the smooth MLPs (0.999 / 1.000),
   even though its F1 is competitive.

4. **Training is worth it.** The weakest trained model (decision-tree, F1 0.944)
   far exceeds the untrained voting baseline (F1 0.559).

## Caveats

- Synthetic data: downstream levels are lossy views of upstream ones and earlier
  tiers carry extra label-correlated signal by construction, so the "earlier is
  better" ordering is partly built in. Real features may differ.
- Binary (violating / safe) task only; per-category metrics are future work.
