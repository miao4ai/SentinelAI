# Fusion — how SentinelAI combines the three experts

The visual, audio, and text experts each form their own opinion; fusion turns those
into **one** moderation verdict. This doc explains exactly how, module by module,
and how we compare fusing at different depths.

Code: `sentinelai/fusion/`.

---

## 0. The problem fusion solves

Each expert speaks a different language and outputs many small pieces:

```
visual → per-frame scores over {violence, nsfw}
audio  → per-window scores over {gunshot, explosion, scream}
text   → per-segment scores over {hate_speech, violence, sexual, toxic}
```

Fusion must (a) put them in a **common vocabulary**, (b) reduce each to a
**video-level** opinion, and (c) **combine** them into a final `is_violating` +
`category`.

---

## 1. Normalise to a shared space — `signals.py`

Before combining, everything is mapped onto the canonical categories
`("violence", "nsfw", "hate_speech")` and pooled to one score vector per modality.

- **Category mapping**: e.g. audio's gunshot/explosion/scream all map to `violence`;
  text's toxic/hate map to `hate_speech`; sexual → `nsfw`.
- **Pooling** (`_pool_and_map`) uses **max**, over two axes at once:
  1. across time (many frames/windows/segments → one number), and
  2. across synonyms (several source categories → the same canonical one).
- **Why max, not mean**: a 2-second gunshot in a 10-minute clip should still push
  `violence` to ~1.0; averaging would dilute it to near zero (a missed violation).
- **`ModalitySignal.present`**: `False` when a modality had no data (e.g. no audio
  track), so fusion **skips** it instead of reading silence as "all-clear".

Output: three `ModalitySignal`s, each a `{category: score}` dict in `[0, 1]`.

---

## 2. Heuristic fast-path (4.1) — `heuristics.py`

Some violations are unambiguous, so we short-circuit before any fusion math.

- `banned_word_rule`: if a transcript segment had an **exact lexicon hit**, return
  a near-certain (`0.99`) violation immediately.
- Rules are **conservative by design** — they only fire on near-certain signals.
  Risky single-modality shortcuts (e.g. "loud bang ⇒ violence") are deliberately
  NOT default rules, because a horror sound-effect over a comedy clip would misfire.
- `apply_heuristics` runs the rule chain; the first hit wins, otherwise falls
  through to fusion.

---

## 3. Late fusion (4.2) — `fusion.py` + `mlp_fusion.py`

### Weighted voting (training-free) — `WeightedVotingFusion`

Per canonical category, combine the **present** modalities' scores:

- **soft** (default): weighted average of the scores → graded confidence.
- **hard**: each modality casts a 0/1 vote (score ≥ threshold), then weighted.
- **Weights** `{text: 1.0, visual: 0.8, audio: 0.6}` — text is trusted most (a
  semantic model on a clean transcript is reliable), audio least (sound effects
  are easily faked). Only participating modalities are normalised over, so a
  missing audio track doesn't drag every score down.

### MLP fusion (trainable) — `MLPFusion`

Concatenate each expert's **last-layer embedding** into one vector →
`Linear → ReLU → Dropout → Linear → logits`. Learns cross-modal interactions that
fixed weights can't, but must be trained on labelled clips first. Kept in a
separate module so the voting path stays torch-free.

### The V1 orchestrator — `V1Moderator.moderate`

```
experts' raw outputs
   → reduce to 3 ModalitySignals            (signals.py)
   → try heuristic fast-path                (4.1) ─ if a rule fires, return now
   → else weighted voting                   (4.2)
   → FusedVerdict{is_violating, category, confidence, reason, source, signals}
```

---

## 4. Where to fuse — the five positions

Fusion is described by **two independent axes**:

- **WHERE (depth)**: input → embedding → feature → decision → vote (earliest→latest).
- **HOW (representation)**: *joint* — squash all modalities into one shared vector;
  vs *coordinated* — keep separate per-modality encoders but **align** them in a
  shared space with a learned (contrastive) model — this is what **CLIP** does.

The two distinctions people get wrong:
1. Concatenating **encoded features** is *not* early fusion — each modality already
   ran its own encoder. True **early fusion** is *before* any encoder, where one
   joint model must **perceive and fuse at once** (position ①).
2. **CLIP is its own position** (②): it fuses at the embedding space through a
   **learned alignment model**, not by concatenation — "embedding *model-level*
   fusion", distinct from the mechanical feature concat of position ③.

```
  ① INPUT / data-level — all modalities' raw signals entangled; one joint model
                          perceives AND fuses
        pixels ┐
        wave   ┼──▶ [ one joint model: perceive + fuse ] ──▶ verdict
        tokens ┘

  ② EMBEDDING model-level (CLIP) — separate encoders, then a LEARNED model aligns
                          their embeddings into a shared space; fuse by similarity
        image ─▶ [img enc] ─┐  learned contrastive
        text  ─▶ [txt enc] ─┴─ alignment (CLIP) ─▶ shared space ─▶ similarity ──▶ verdict

  ③ FEATURE-level — separate encoders, then CONCATENATE the features → one model
        v/a/t ─▶ [enc] ─▶ concat ─▶ MLP / cross-attention ──▶ verdict

  ④ DECISION-level — each modality → category scores → a classifier combines them
        v/a/t ─▶ [enc→head] ─▶ scores ─▶ tree ──▶ verdict

  ⑤ LATE / vote — each modality → 0/1 → weighted vote
        v/a/t ─▶ [enc→head→0/1] ─▶ weighted vote ──▶ verdict

   earliest ──────────────────────────────────────────────▶ latest
   most info / hardest to train              most robust / interpretable / lossiest
```

| # | Position | Mechanism | Style | Our code |
|---|---|---|---|---|
| ① | **input / data-level** | one joint model perceives + fuses raw signals | joint | `early-fusion` |
| ② | **embedding model-level (CLIP)** | learned alignment of per-modality embeddings into a shared space, fuse by similarity | coordinated | `clip_screener.py` |
| ③ | **feature-level** | concat encoded features → MLP / cross-attention | joint | `embedding-mlp`, `cross_attention.py` |
| ④ | **decision-level** | combine per-modality scores with a classifier | — | `decision-tree` |
| ⑤ | **late / vote** | combine per-modality 0/1 votes | — | `mean-voting` |

**② vs ③ (the key distinction):** position ② fuses through a *learned alignment
model* (CLIP's contrastive training builds a shared embedding space, fuse by cosine
similarity — "model-level"); position ③ just *concatenates* the encoded features
and lets one MLP sort them out. Same inputs (embeddings), different mechanism.

---

## 5. Comparing the positions — `compare.py` + `synthetic.py`

`compare.py` trains one model per position on the **same** train/test split and
reports Precision / Recall / F1 / AUC. `synthetic.py` generates features at each
tier so the framework runs with **no GPU and no real data**. The tiers are built to
be genuinely different, not just wider/narrower:

- **input** — ONE joint block that mixes *all modalities' raw signal together*
  through a nonlinearity, so a model must perceive + fuse (true early fusion).
- **feature** — per-modality *encoded* blocks concatenated → MLP.
- **decision / vote** — per-modality category scores → tree / threshold-vote.

Position ② (CLIP, embedding model-level) uses a fundamentally different mechanism
(learned contrastive alignment + similarity, not a classifier over concatenated
features), so it is **not** in this synthetic sweep — it is implemented in
`clip_screener.py` and was verified on **real Kinetics video** (the archery case in
§6), which is a stronger test than the synthetic sweep.

Run it:

```bash
python -m sentinelai.fusion.compare
```

Result (synthetic):

| strategy | position | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **early-fusion** | ① input | **1.000** | **1.000** | **1.000** | **1.000** |
| embedding-mlp | ③ feature | 0.983 | 0.993 | 0.988 | 0.999 |
| decision-tree | ④ decision | 0.936 | 0.952 | 0.944 | 0.956 |
| mean-voting | ⑤ vote (untrained) | 0.388 | 1.000 | 0.559 | 0.994 |

_(Position ② CLIP: see `clip_screener.py`; verified on real video, not in this sweep.)_

**Takeaways**

- **Earlier fusion wins** (① > ③ > ④ > ⑤) — it keeps information the downstream
  levels discarded. Early fusion needs a **deeper** model because its input block
  entangles every modality, so the model must perceive *and* fuse (real early
  fusion's central difficulty).
- **AUC vs F1 can disagree**: `mean-voting` has a poor F1 (0.559) but high AUC
  (0.994) — its ranking is good, only the fixed 0.5 threshold is bad. Never judge a
  fusion by a single-threshold P/R alone.
- **Caveat**: synthetic data by construction favours earlier fusion; real features
  may differ. Numbers validate the framework, not a production claim.

### Trainable modules (①②③)

The sweep above uses sklearn classifiers over pre-mixed features. The *real*
`nn.Module` for each of ①②③ is also trained end-to-end (PyTorch Lightning) on one
shared synthetic token dataset — `JointFusionTransformer` (①), `CoordinatedFusion`
(②, CLIP-style), and `MLPFusion` (③). All reach val_auc 1.00, with terminal loss
**① 0.0007 < ② 0.0034 < ③ 0.0071** — earlier fusion again fits the token-level
signal best. Details and how the data is generated: [`experiments.md`](experiments.md)
§6; run with `python -m sentinelai.train.train_fusion`.

> Taxonomy references: [Multimodal Alignment and Fusion: A Survey](https://arxiv.org/pdf/2411.17040),
> [Multimodal Classification: Current Landscape, Taxonomy and Future Directions](https://arxiv.org/pdf/2109.09020).

---

## 6. Evaluation & bad cases (4.3) — `evaluate.py`

- `binary_metrics` — Precision / Recall / F1 baseline that later versions must beat.
- `find_bad_cases` — split misclassifications into false positives / false negatives
  to inspect by hand.
- `cross_modal_conflict` — flag clips where one modality is alarmed while another
  is calm (the "horror SFX on a comedy clip" pattern). On a real archery clip, CLIP
  screamed `violence 0.97` while audio stayed at `0.003`; the conflict detector
  caught it and routed it to human review — fusion flagged, but the conflict flag
  saved the false takedown.

---

## TL;DR

1. **Normalise** each expert to canonical categories, pool over time with **max**.
2. **Short-circuit** on near-certain heuristics (banned words).
3. **Fuse** the rest — weighted voting (default) or a trained model at one of the
   fusion depths (early / intermediate / late); earlier is more powerful but harder.
4. **Evaluate** with P/R/F1/AUC and triage with cross-modal conflict detection.
