# Experiment 1 — Fusion depth comparison (V1)

**Question:** where in the pipeline should we fuse the three experts (visual /
audio / text), and with what model? We compare fusing at three depths — early
(feature), intermediate (embedding), and late (decision) — and measure
Precision / Recall / F1 / AUC. (An even earlier *signal/data-level* fusion is
discussed in `docs/fusion.md` but not benchmarked here.)

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
| **early-mlp** | early / feature (earliest) | concat 3 experts' raw/input features (256-d) → Linear → ReLU(256) → ReLU(128) → 2 |
| **embedding-mlp** | intermediate / embedding | concat 3 experts' backbone embeddings (96-d) → Linear → ReLU(128) → Linear → 2 |
| **decision-tree** | late / decision | concat 3 experts' final category scores (9-d) → `DecisionTree(max_depth=5)` |
| **mean-voting** (baseline) | late / decision | untrained: concat experts' final scores, flag if `max >= 0.5` |

All strategies are trained/evaluated on the **same** seeded train/test split, so
differences reflect the fusion depth + model, not the data.

## Results (synthetic test set)

| Fusion | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **early-mlp** | **1.000** | **1.000** | **1.000** | **1.000** |
| embedding-mlp | 0.983 | 0.986 | 0.985 | 0.999 |
| decision-tree | 0.945 | 0.942 | 0.943 | 0.957 |
| mean-voting (untrained) | 0.388 | 1.000 | 0.559 | 0.988 |

## Takeaways

1. **Earlier fusion wins.** early-mlp (F1 1.00) > embedding-mlp (0.985) >
   decision-tree (0.943). Earlier tiers retain signal the downstream levels
   discard — at the cost of higher dimensionality (256 vs 9) and, for the raw
   tier, needing a deeper model to untangle it.

2. **AUC and F1 can disagree — measure both.** mean-voting has a poor F1 (0.559,
   precision 0.388: it over-flags) yet a high AUC (0.988). Its *ranking* of
   samples is good; only the fixed 0.5 threshold is bad. Judging a fusion by a
   single-threshold P/R alone is misleading.

3. **Decision trees rank worse than MLPs.** The tree's coarse, step-wise
   probabilities give a lower AUC (0.957) than the smooth MLPs (0.999 / 1.000),
   even though its F1 is competitive.

4. **Training is worth it.** The weakest trained model (decision-tree, F1 0.943)
   far exceeds the untrained voting baseline (F1 0.559).

## Caveats

- Synthetic data: downstream levels are lossy views of upstream ones and earlier
  tiers carry extra label-correlated signal by construction, so the "earlier is
  better" ordering is partly built in. Real features may differ.
- Binary (violating / safe) task only; per-category metrics are future work.
