# Experiment 1 — Fusion depth comparison (V1)

**Question:** where in the pipeline should we fuse the three experts (visual /
audio / text), and with what model? We compare fusing at three depths and measure
Precision / Recall / F1 / AUC.

**Status:** run on **synthetic** multimodal features
(`sentinelai/fusion/synthetic.py`). The numbers validate the framework and the
depth trade-offs; real-data numbers await the feature-extraction pipeline over a
labelled video set. Reproduce with:

```bash
python -m sentinelai.fusion.compare
```

## Strategies

| Fusion | Fuse at | Structure |
|---|---|---|
| **mean-voting** (baseline) | decision | untrained: concat experts' final scores, flag if `max >= 0.5` |
| **decision-tree** | decision (latest) | concat 3 experts' final category scores (9-d) → `DecisionTree(max_depth=5)` |
| **logit-mlp** | logits (middle) | concat 3 experts' pre-activation logits (9-d) → Linear → ReLU(64) → Linear → 2 |
| **embedding-mlp** | embedding (earliest) | concat 3 experts' backbone embeddings (96-d) → Linear → ReLU(128) → Linear → 2 |

All strategies are trained/evaluated on the **same** seeded train/test split, so
differences reflect the fusion depth + model, not the data.

## Results (synthetic test set)

| Fusion | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| mean-voting (untrained) | 0.388 | 1.000 | 0.559 | 0.994 |
| decision-tree | 0.936 | 0.952 | 0.944 | 0.956 |
| logit-mlp | 0.973 | 0.973 | 0.973 | 0.998 |
| **embedding-mlp** | **0.983** | **0.993** | **0.988** | **0.999** |

## Takeaways

1. **Earlier fusion wins.** embedding-mlp (F1 0.988) > logit-mlp (0.973) >
   decision-tree (0.944). Embeddings retain signal that logits/scores discard —
   at the cost of higher dimensionality (96 vs 9) and more overfitting risk when
   data is scarce.

2. **AUC and F1 can disagree — measure both.** mean-voting has a poor F1 (0.559,
   precision 0.388: it over-flags) yet a high AUC (0.994). Its *ranking* of
   samples is good; only the fixed 0.5 threshold is bad. Judging a fusion by a
   single-threshold P/R alone is misleading.

3. **Decision trees rank worse than MLPs.** The tree's coarse, step-wise
   probabilities give a lower AUC (0.956) than the smooth MLPs (0.998 / 0.999),
   even though its F1 is competitive.

4. **Training is worth it.** The weakest trained model (decision-tree, F1 0.944)
   far exceeds the untrained voting baseline (F1 0.559).

## Caveats

- Synthetic data: downstream levels are lossy views of upstream ones and the
  embedding carries one extra label-correlated dimension by construction, so the
  "earlier is better" ordering is partly built in. Real features may differ.
- Binary (violating / safe) task only; per-category metrics are future work.
