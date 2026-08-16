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

Concatenate each expert's **embedding or logits** into one vector →
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

Fusion can happen at five depths along the pipeline. Earlier = more information but
higher-dimensional/harder; later = interpretable/robust but lossy.

| # | Position | Fuse | Method | Code |
|---|---|---|---|---|
| ① | **raw / input** | pre-backbone signals (pixels/waveform/tokens) | deep MLP | `early-mlp` |
| ② | **embedding** | backbone feature vectors | MLP / cross-attention | `embedding-mlp`, `cross_attention.py` |
| ③ | **logits** | pre-activation logits | MLP | `logit-mlp` |
| ④ | **decision** | per-category probabilities | decision tree | `decision-tree` |
| ⑤ | **vote / late** | each expert's 0/1 | weighted voting | `mean-voting` |

Position ② also has a **deep** variant beyond concatenation: **cross-attention**
(`cross_attention.py`), where audio/text act as the query and video frames as
key/value — "when I hear a scream, attend to *these* frames".

---

## 5. Comparing the positions — `compare.py` + `synthetic.py`

`compare.py` trains one model per position on the **same** train/test split and
reports Precision / Recall / F1 / AUC. `synthetic.py` generates correlated
multimodal features at all five tiers so the framework runs with **no GPU and no
real data** (the "synthetic-first" path). By construction the tiers are nested
`raw ⊇ embedding ⊇ logits ⊇ decision`, so earlier fusion has strictly more signal —
the experiment measures whether each model actually uses it.

Run it:

```bash
python -m sentinelai.fusion.compare
```

Result (synthetic):

| strategy | position | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **early-mlp** | ① raw | **1.000** | **1.000** | **1.000** | **1.000** |
| embedding-mlp | ② embedding | 0.983 | 0.986 | 0.985 | 0.999 |
| logit-mlp | ③ logits | 0.983 | 0.979 | 0.981 | 0.998 |
| decision-tree | ④ decision | 0.945 | 0.942 | 0.943 | 0.957 |
| mean-voting | ⑤ vote | 0.388 | 1.000 | 0.559 | 0.988 |

**Takeaways**

- **Earlier fusion wins** (① > ② > ③ > ④ > ⑤) — it keeps information downstream
  levels discarded. Position ① needs a **deeper** MLP because its raw features are
  entangled by a nonlinearity (mirrors real early fusion needing more capacity).
- **AUC vs F1 can disagree**: `mean-voting` has a poor F1 (0.559) but high AUC
  (0.988) — its ranking is good, only the fixed 0.5 threshold is bad. Never judge a
  fusion by a single-threshold P/R alone.
- **Caveat**: synthetic data by construction favours earlier fusion; real features
  may differ. Numbers validate the framework, not a production claim.

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
3. **Fuse** the rest — weighted voting (default) or a trained model at one of five
   depths; earlier is more powerful but harder.
4. **Evaluate** with P/R/F1/AUC and triage with cross-modal conflict detection.
