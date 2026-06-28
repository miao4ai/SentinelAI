"""Normalise each expert's output into one shared category space.

Before we can fuse CV / Audio / NLP we must speak a common language. Each expert
reports different categories:

* visual -> "violence" (and optionally "nsfw")
* audio  -> "gunshot" / "explosion" / "scream"   (all kinds of violent sound)
* text   -> "hate_speech" / "violence" / "sexual" / "toxic"

This module maps all of them onto the canonical V1 categories and reduces the
experts' *per-unit* outputs (per frame / per audio window / per transcript
segment) into a single **video-level** score vector per modality. Everything here
is pure Python (no torch/numpy), so it imports cheaply and is easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# The categories V1 reasons about after fusion. Keep this the single source of truth.
CANONICAL_CATEGORIES: tuple[str, ...] = ("violence", "nsfw", "hate_speech")

# Per-modality maps from an expert's own category -> a canonical category. A source
# category absent from a map (e.g. visual "safe") is intentionally ignored.
VISUAL_TO_CANONICAL: dict[str, str] = {"violence": "violence", "nsfw": "nsfw"}
AUDIO_TO_CANONICAL: dict[str, str] = {
    "gunshot": "violence",
    "explosion": "violence",
    "scream": "violence",
}
TEXT_TO_CANONICAL: dict[str, str] = {
    "violence": "violence",
    "sexual": "nsfw",
    "hate_speech": "hate_speech",
    "toxic": "hate_speech",
}


@dataclass(frozen=True)
class ModalitySignal:
    """One modality's video-level opinion, as a score per canonical category.

    Attributes:
        modality: "visual" / "audio" / "text".
        scores:   canonical category -> score in [0, 1]; categories the modality
                  cannot judge are simply 0 (e.g. visual has no opinion on hate
                  speech).
        present:  False when the modality contributed no data (empty input), so
                  fusion can skip it instead of treating it as "all-clear".
    """

    modality: str
    scores: dict[str, float]
    present: bool = True

    @property
    def max_score(self) -> float:
        """The strongest single-category score — this modality's overall alarm level."""
        return max(self.scores.values()) if self.scores else 0.0


def _pool_and_map(units: Sequence[dict[str, float]], mapping: dict[str, str]) -> dict[str, float]:
    """Pool per-unit category scores into one canonical score vector via max.

    Two reductions happen here, both using **max** ("did this ever happen, at its
    most confident moment?"):
      1. across time — many frames/windows/segments collapse to one number, and
      2. across synonyms — several source categories that map to the same canonical
         one (e.g. gunshot+explosion+scream -> violence) take their max.

    Max (not mean) is deliberate: a 2-second gunshot in a 10-minute clip should
    still raise "violence" to ~1.0, which averaging would dilute to near zero.
    """
    canonical = {c: 0.0 for c in CANONICAL_CATEGORIES}
    for unit_scores in units:
        for src_cat, prob in unit_scores.items():
            canon = mapping.get(src_cat)
            if canon is not None:
                canonical[canon] = max(canonical[canon], float(prob))
    return canonical


def reduce_visual(frames: Sequence[Any] | None) -> ModalitySignal:
    """Reduce a list of ``FramePrediction`` into the visual modality signal.

    Reads each frame's ``.scores`` dict (duck-typed, so we don't import the
    torch-heavy visual_expert module just for a type). Empty input -> not present.
    """
    if not frames:
        return ModalitySignal("visual", {c: 0.0 for c in CANONICAL_CATEGORIES}, present=False)
    return ModalitySignal("visual", _pool_and_map([f.scores for f in frames], VISUAL_TO_CANONICAL))


def reduce_audio(events: Sequence[Any] | None) -> ModalitySignal:
    """Reduce a list of ``AudioEvent`` into the audio modality signal."""
    if not events:
        return ModalitySignal("audio", {c: 0.0 for c in CANONICAL_CATEGORIES}, present=False)
    return ModalitySignal("audio", _pool_and_map([e.scores for e in events], AUDIO_TO_CANONICAL))


def reduce_text(verdicts: Sequence[Any] | None) -> ModalitySignal:
    """Reduce a list of ``TextVerdict`` into the text modality signal.

    Uses each verdict's ``.semantic`` scores (the model's view); explicit lexical
    hits are handled separately by the heuristics fast-path, not here.
    """
    if not verdicts:
        return ModalitySignal("text", {c: 0.0 for c in CANONICAL_CATEGORIES}, present=False)
    return ModalitySignal("text", _pool_and_map([v.semantic for v in verdicts], TEXT_TO_CANONICAL))
