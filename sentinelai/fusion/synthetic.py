"""Synthetic multimodal features for prototyping the fusion-depth comparison.

We generate, for each fake clip, the features a real pipeline would produce at all
three depths — embeddings, logits, decision scores — for three modalities, plus a
binary "violating" label. This lets ``compare.py`` run without GPU, real models or
labelled data, so we can validate the comparison framework first (the chosen
"synthetic-first" path) and swap in real cached features later.

The generator is built to make the comparison *interesting* rather than trivial:

* Modalities have different **reliability** (text > visual > audio), mirroring the
  voting weights — so a model can learn to trust them unequally.
* The downstream levels are **lossy** views of the upstream ones: scores come from
  logits, and the embedding carries one extra label-correlated dimension that
  never reaches the logits/scores. So earlier-fusion strategies have strictly more
  signal to exploit — the comparison can reveal whether they actually use it.
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

# Per-modality *raw*/input width — the widest, messiest tier (position ①). Stands
# in for pre-backbone signals (pixels / waveform / tokens): the label information
# is present but entangled by a nonlinearity, so only a deeper model can decode it.
_RAW_DIM: dict[str, int] = {"visual": 128, "audio": 64, "text": 64}


def make_synthetic_dataset(
    n_samples: int = 3000,
    p_violating: float = 0.4,
    seed: int = 0,
) -> FusionDataset:
    """Generate a :class:`FusionDataset` of correlated multimodal features.

    Procedure per sample:
      1. Decide the label ``y`` (violating with probability ``p_violating``) and,
         when violating, which canonical category is active.
      2. For each modality, build **logits** over the categories: noise everywhere,
         with the active category's logit boosted by the modality's reliability.
      3. **Scores** = sigmoid(logits) — the lossy, decision-level view.
      4. **Embedding** = the logits copied into the first dims + pure-noise dims +
         one extra label-correlated dim that exists only here (upstream-only signal).
      5. **Raw** (position ①) = a wide, entangled tanh mixing carrying the most
         label signal (two extra dims) — only a deeper model decodes it.
      6. Concatenate each level across the three modalities.

    Information is monotonically non-increasing downstream:
    ``raw ⊇ embedding ⊇ logits ⊇ decision``, so earlier fusion has strictly more to
    work with — at the cost of width and (for raw) a nonlinearity to untangle.
    """
    rng = np.random.default_rng(seed)
    num_cat = len(CANONICAL_CATEGORIES)

    # 1. labels and (for violating samples) which category is active.
    y = (rng.random(n_samples) < p_violating).astype(int)
    active_cat = rng.integers(0, num_cat, size=n_samples)
    violating = y.astype(bool)
    signed = (2 * y - 1).astype(float)   # -1 / +1 label direction

    raw_blocks: list[np.ndarray] = []
    emb_blocks: list[np.ndarray] = []
    logit_blocks: list[np.ndarray] = []
    score_blocks: list[np.ndarray] = []

    for modality, reliability in _RELIABILITY.items():
        dim = _EMB_DIM[modality]

        # 2. logits: standard-normal noise, with the active category boosted on
        #    violating rows by an amount that scales with the modality's reliability.
        logits = rng.normal(0.0, 1.0, size=(n_samples, num_cat))
        boost = reliability * 3.0 + rng.normal(0.0, 0.5, size=violating.sum())
        logits[violating, active_cat[violating]] += boost

        # 3. scores: sigmoid (multi-label, matching the real audio/text experts).
        scores = 1.0 / (1.0 + np.exp(-logits))

        # 4. embedding: first `num_cat` dims echo the logits; the rest is noise; one
        #    extra dim carries label signal that the logits/scores never saw.
        embedding = rng.normal(0.0, 1.0, size=(n_samples, dim))
        embedding[:, :num_cat] += logits
        embedding[:, num_cat] += signed * reliability * 1.5 + rng.normal(0.0, 0.5, size=n_samples)

        # 5. raw: a fixed random *nonlinear* mixing of the richest signal — the
        #    logits plus TWO extra label-correlated dims (one more than embedding).
        #    tanh entangles it, so a shallow/linear head can't read it; a deeper MLP
        #    can. This is the "pre-backbone, everything's here but messy" tier.
        raw_dim = _RAW_DIM[modality]
        signal = np.concatenate(
            [logits, (signed * reliability)[:, None], (signed * reliability * 1.2)[:, None]],
            axis=1,
        )
        mix = rng.normal(size=(signal.shape[1], raw_dim))
        raw = np.tanh(signal @ mix) + rng.normal(0.0, 0.5, size=(n_samples, raw_dim))

        raw_blocks.append(raw)
        emb_blocks.append(embedding)
        logit_blocks.append(logits)
        score_blocks.append(scores)

    # 6. concatenate modalities along the feature axis for each level.
    return FusionDataset(
        X_raw=np.concatenate(raw_blocks, axis=1),
        X_embedding=np.concatenate(emb_blocks, axis=1),
        X_logits=np.concatenate(logit_blocks, axis=1),
        X_decision=np.concatenate(score_blocks, axis=1),
        y=y,
    )
