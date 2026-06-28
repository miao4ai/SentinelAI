"""4.3 V1 baseline evaluation: Precision/Recall and bad-case analysis.

Two jobs:
  1. Establish the headline numbers — Precision, Recall, F1 — that every later
     version (V2 fusion, V3 VLM) must beat.
  2. Surface *why* V1 is wrong: list the false positives / false negatives, and
     flag **cross-modal conflicts** — clips where modalities strongly disagree,
     the classic source of V1 errors (e.g. a comedy video dubbed with a horror
     sound effect makes the audio expert scream "violence" while the frames are
     clearly safe).

Pure Python so it runs anywhere and is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .signals import ModalitySignal


@dataclass(frozen=True)
class Metrics:
    """Binary classification metrics for the "is this clip violating?" task.

    Precision = of the clips we flagged, how many were truly violating (low =
    annoying false alarms). Recall = of the truly violating clips, how many we
    caught (low = unsafe content slips through). F1 balances the two.
    """

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Metrics:
    """Compute Precision/Recall/F1 from ground-truth vs predicted violating flags.

    Counts the confusion-matrix cells, then derives the rates. Division-by-zero is
    guarded (e.g. precision is 0 when nothing was flagged) so the function never
    raises on degenerate inputs.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


@dataclass(frozen=True)
class BadCase:
    """A single misclassified clip, kept for manual inspection.

    Attributes:
        clip_id: identifier of the offending clip.
        kind:    "false_positive" (flagged a safe clip) or "false_negative"
                 (missed a violating clip).
        verdict: the V1 FusedVerdict we produced (duck-typed; has .reason/.signals).
    """

    clip_id: str
    kind: str
    verdict: Any


def find_bad_cases(records: Sequence[tuple[str, bool, Any]]) -> dict[str, list[BadCase]]:
    """Split (clip_id, is_violating_truth, verdict) records into FP and FN buckets.

    Reads ``verdict.is_violating`` to compare against truth. The buckets are what
    you actually open and watch during bad-case analysis — start with the bucket
    that hurts your use case most (FNs for safety, FPs for creator experience).
    """
    false_positives: list[BadCase] = []
    false_negatives: list[BadCase] = []
    for clip_id, truth, verdict in records:
        pred = bool(verdict.is_violating)
        if pred and not truth:
            false_positives.append(BadCase(clip_id, "false_positive", verdict))
        elif truth and not pred:
            false_negatives.append(BadCase(clip_id, "false_negative", verdict))
    return {"false_positives": false_positives, "false_negatives": false_negatives}


@dataclass(frozen=True)
class Conflict:
    """How much the modalities disagree about a clip.

    Attributes:
        score:      loudest modality's alarm minus the quietest's, in [0, 1].
        loudest:    modality with the highest single-category score.
        quietest:   modality with the lowest.
        is_conflict: True when one modality is alarmed while another is calm —
                     a red flag for a likely V1 false positive worth reviewing.
    """

    score: float
    loudest: str
    quietest: str
    is_conflict: bool


def cross_modal_conflict(
    signals: Sequence[ModalitySignal],
    high: float = 0.6,
    margin: float = 0.5,
) -> Conflict:
    """Measure disagreement among the present modalities for one clip.

    We compare each present modality's overall alarm level (its max category
    score). A big gap — one modality loud (>= ``high``) while another stays quiet
    (gap >= ``margin``) — is exactly the "horror SFX on a comedy clip" pattern,
    where naive fusion would over-flag. Use it to triage which flagged clips a
    human should re-check.

    With fewer than two present modalities there is nothing to disagree, so the
    result is ``is_conflict=False``.
    """
    present = [s for s in signals if s.present]
    if len(present) < 2:
        return Conflict(score=0.0, loudest="", quietest="", is_conflict=False)

    loud = max(present, key=lambda s: s.max_score)
    quiet = min(present, key=lambda s: s.max_score)
    score = loud.max_score - quiet.max_score
    is_conflict = loud.max_score >= high and score >= margin
    return Conflict(
        score=score,
        loudest=loud.modality,
        quietest=quiet.modality,
        is_conflict=is_conflict,
    )
