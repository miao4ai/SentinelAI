"""4.2 Simple late fusion: weighted voting over modality signals, plus the V1 orchestrator.

"Late" fusion means each modality decides on its own first, then we combine their
*decisions* (here: per-category scores) — as opposed to "early" fusion which would
mix raw features before deciding. This file implements the training-free option:
**weighted soft/hard voting**. The trainable MLP alternative lives in
``mlp_fusion.py`` (kept separate so this module needs no torch).

It also defines :class:`V1Moderator`, the end-to-end V1 model: it runs the 4.1
heuristic fast-path first, and only if nothing fires does it fall back to fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .heuristics import DEFAULT_RULES, apply_heuristics
from .signals import (
    CANONICAL_CATEGORIES,
    ModalitySignal,
    reduce_audio,
    reduce_text,
    reduce_visual,
)

# Default trust placed in each modality when voting. Text is weighted highest (a
# semantic toxicity model on a clean transcript is reliable); audio lowest (sound
# effects are easily faked — the "comedy clip + horror SFX" failure mode).
DEFAULT_WEIGHTS: dict[str, float] = {"text": 1.0, "visual": 0.8, "audio": 0.6}


@dataclass(frozen=True)
class FusedVerdict:
    """The final V1 decision, shaped to match the project's structured output.

    Attributes:
        is_violating: whether the clip is flagged.
        category:     the most-likely violation category (None when nothing fires).
        confidence:   score of that category in [0, 1].
        reason:       human-readable explanation.
        scores:       fused score per canonical category.
        source:       which stage decided — "heuristic" or "voting" (or "mlp").
        signals:      the per-modality signals that fed the decision (for analysis).
    """

    is_violating: bool
    category: str | None
    confidence: float
    reason: str
    scores: dict[str, float]
    source: str
    signals: list[ModalitySignal] = field(default_factory=list)


class WeightedVotingFusion:
    """Combine modality signals into one verdict by weighted voting.

    Two modes:
      * ``"soft"`` (default) — weighted average of each modality's per-category
        score, giving a graded confidence. Best when scores are calibrated.
      * ``"hard"`` — each modality casts a yes/no vote per category (score >=
        ``vote_threshold``); the weighted vote share becomes the score. More robust
        to one modality's miscalibrated magnitudes.
    """

    def __init__(
        self,
        weights: dict[str, float] = DEFAULT_WEIGHTS,
        mode: str = "soft",
        threshold: float = 0.5,
        vote_threshold: float = 0.5,
    ) -> None:
        if mode not in ("soft", "hard"):
            raise ValueError("mode must be 'soft' or 'hard'")
        self.weights = weights
        self.mode = mode
        self.threshold = threshold          # flag when top fused score >= this
        self.vote_threshold = vote_threshold  # per-modality yes/no cut for hard voting

    def fuse(self, signals: Sequence[ModalitySignal]) -> FusedVerdict:
        """Fuse signals -> :class:`FusedVerdict`. Only *present* modalities count.

        For each canonical category we combine the present modalities' opinions,
        normalising by the weights that actually participated (so a missing audio
        track doesn't drag every score down).
        """
        present = [s for s in signals if s.present]
        fused = {c: 0.0 for c in CANONICAL_CATEGORIES}

        if present:
            for category in CANONICAL_CATEGORIES:
                weighted_sum = 0.0
                weight_total = 0.0
                for sig in present:
                    w = self.weights.get(sig.modality, 1.0)
                    raw = sig.scores.get(category, 0.0)
                    # In hard mode the modality's opinion becomes a 0/1 vote.
                    value = (1.0 if raw >= self.vote_threshold else 0.0) if self.mode == "hard" else raw
                    weighted_sum += w * value
                    weight_total += w
                fused[category] = weighted_sum / weight_total if weight_total else 0.0

        # The flagged category is the strongest fused one.
        category = max(fused, key=fused.__getitem__)
        confidence = fused[category]
        is_violating = confidence >= self.threshold
        reason = (
            f"{self.mode} voting: {category}={confidence:.2f} "
            f"from {[s.modality for s in present] or 'no modalities'}"
        )
        return FusedVerdict(
            is_violating=is_violating,
            category=category if is_violating else None,
            confidence=confidence,
            reason=reason,
            scores=fused,
            source="voting",
            signals=list(signals),
        )


class V1Moderator:
    """End-to-end V1 model: heuristic fast-path (4.1) then late fusion (4.2).

    ``moderate`` takes the three experts' raw per-unit outputs, tries the cheap
    high-confidence rules first, and only runs fusion if no rule fired.
    """

    def __init__(
        self,
        fusion: WeightedVotingFusion | None = None,
        rules: Sequence[Any] = DEFAULT_RULES,
    ) -> None:
        self.fusion = fusion or WeightedVotingFusion()
        self.rules = rules

    def moderate(
        self,
        *,
        frames: Sequence[Any] | None = None,
        audio_events: Sequence[Any] | None = None,
        text_verdicts: Sequence[Any] | None = None,
    ) -> FusedVerdict:
        """Produce the final verdict for one clip from the three experts' outputs."""
        signals = [
            reduce_visual(frames),
            reduce_audio(audio_events),
            reduce_text(text_verdicts),
        ]

        # 4.1 — fast circuit-breaker. If a high-confidence rule fires, return now
        # and skip fusion entirely (the whole point of the heuristic path).
        hit = apply_heuristics(
            text_verdicts=text_verdicts,
            frames=frames,
            audio_events=audio_events,
            rules=self.rules,
        )
        if hit is not None:
            scores = {c: 0.0 for c in CANONICAL_CATEGORIES}
            scores[hit.category] = hit.confidence
            return FusedVerdict(
                is_violating=True,
                category=hit.category,
                confidence=hit.confidence,
                reason=hit.reason,
                scores=scores,
                source="heuristic",
                signals=signals,
            )

        # 4.2 — fall back to late fusion of all modalities.
        return self.fusion.fuse(signals)
