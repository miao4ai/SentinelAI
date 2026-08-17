"""Synthetic multimodal features for prototyping the fusion-position comparison.

We generate, for each fake clip, the features a real pipeline would produce at three
fusion positions — plus a binary "violating" label — so ``compare.py`` runs without
GPU, real models, or labelled data (the "synthetic-first" path).

The three tiers, and what makes them genuinely different (not just wider/narrower):

* **early / input** (``X_early``) — a *single joint* block that mixes **all three
  modalities together** through a shared nonlinearity. The modalities are entangled
  (you cannot read one off without the others), so one joint model must learn to
  *perceive and fuse at once* — that is what true early / input-level fusion is.
* **intermediate / embedding** (``X_embedding``) — per-modality **encoded** feature
  blocks concatenated. Each modality is already cleanly separated (its own encoder
  did the perceiving); fusion just combines them. This is intermediate fusion.
* **late / decision** (``X_decision``) — per-modality category **scores**. Fusion
  combines finished per-modality opinions.

Modalities also have different **reliability** (text > visual > audio), mirroring
the voting weights, so a model can learn to trust them unequally.
"""

from __future__ import annotations

import numpy as np

from .compare import FusionDataset
from .signals import CANONICAL_CATEGORIES

# How strongly each modality's features reflect the true label (text most, audio
# least), echoing fusion.DEFAULT_WEIGHTS.
_RELIABILITY: dict[str, float] = {"text": 1.2, "visual": 0.9, "audio": 0.6}

# Per-modality embedding width — scaled-down stand-ins for the real 2048/768/768.
_EMB_DIM: dict[str, int] = {"visual": 48, "audio": 24, "text": 24}

# Width of the single joint early/input block (all modalities mixed together).
_EARLY_DIM: int = 256


def make_synthetic_dataset(
    n_samples: int = 3000,
    p_violating: float = 0.4,
    seed: int = 0,
) -> FusionDataset:
    """Generate a :class:`FusionDataset` with features at three fusion positions.

    Procedure:
      1. Decide the label ``y`` and, when violating, which category is active.
      2. Per modality, build **logits** (noise + a boost on the active category),
         **scores** = sigmoid(logits), and an **embedding** block (logits echoed
         into the first dims + noise + one extra label-correlated dim).
      3. Collect every modality's raw signal, then build ONE **early/input** block
         by mixing them all together through a shared random nonlinearity (tanh) —
         modalities entangled, so a joint model must perceive + fuse.
      4. Concatenate embeddings and scores across modalities for the intermediate
         and late tiers.
    """
    rng = np.random.default_rng(seed)
    num_cat = len(CANONICAL_CATEGORIES)

    # 1. labels and (for violating samples) which category is active.
    y = (rng.random(n_samples) < p_violating).astype(int)
    active_cat = rng.integers(0, num_cat, size=n_samples)
    violating = y.astype(bool)
    signed = (2 * y - 1).astype(float)   # -1 / +1 label direction

    emb_blocks: list[np.ndarray] = []
    score_blocks: list[np.ndarray] = []
    raw_signals: list[np.ndarray] = []   # per-modality signal feeding the joint early block

    for modality, reliability in _RELIABILITY.items():
        dim = _EMB_DIM[modality]

        # 2a. logits: standard-normal noise, active category boosted on violating rows.
        logits = rng.normal(0.0, 1.0, size=(n_samples, num_cat))
        boost = reliability * 3.0 + rng.normal(0.0, 0.5, size=violating.sum())
        logits[violating, active_cat[violating]] += boost

        # 2b. scores: sigmoid (multi-label, matching the real audio/text experts).
        scores = 1.0 / (1.0 + np.exp(-logits))

        # 2c. embedding: logits echoed into the first dims + noise + one extra
        #     label-correlated dim the scores never saw.
        embedding = rng.normal(0.0, 1.0, size=(n_samples, dim))
        embedding[:, :num_cat] += logits
        embedding[:, num_cat] += signed * reliability * 1.5 + rng.normal(0.0, 0.5, size=n_samples)

        emb_blocks.append(embedding)
        score_blocks.append(scores)
        # richest per-modality signal (logits + an extra label dim) for the joint mix.
        raw_signals.append(np.concatenate([logits, (signed * reliability)[:, None]], axis=1))

    # 3. early / input: mix ALL modalities' raw signal together through one shared
    #    nonlinearity. Because everything is entangled in a single block, a model
    #    here must jointly perceive and fuse — true early / input-level fusion.
    joint_signal = np.concatenate(raw_signals, axis=1)                 # (N, 3*(num_cat+1))
    mix = rng.normal(size=(joint_signal.shape[1], _EARLY_DIM))
    X_early = np.tanh(joint_signal @ mix) + rng.normal(0.0, 0.5, size=(n_samples, _EARLY_DIM))

    return FusionDataset(
        X_early=X_early,
        X_embedding=np.concatenate(emb_blocks, axis=1),
        X_decision=np.concatenate(score_blocks, axis=1),
        y=y,
    )
