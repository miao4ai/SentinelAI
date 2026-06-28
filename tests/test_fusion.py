"""Tests for the V1 fusion package (heuristics + voting + evaluation).

The whole package (except mlp_fusion) is torch-free, so these run offline. We
duck-type the experts' outputs with tiny namespace objects carrying just the
attributes the reducers/heuristics read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sentinelai.fusion import (
    V1Moderator,
    WeightedVotingFusion,
    binary_metrics,
    cross_modal_conflict,
    find_bad_cases,
    reduce_audio,
    reduce_text,
    reduce_visual,
)


# -- fake expert outputs ----------------------------------------------------

def _frame(violence: float):
    """A FramePrediction stand-in: only ``.scores`` is read by reduce_visual."""
    return SimpleNamespace(scores={"safe": 1 - violence, "violence": violence})


def _audio(gunshot: float = 0.0, scream: float = 0.0):
    return SimpleNamespace(scores={"gunshot": gunshot, "explosion": 0.0, "scream": scream})


def _text(semantic: dict, hits=None):
    """A TextVerdict stand-in with semantic scores and an optional lexical hit."""
    lexical = SimpleNamespace(hits=hits or [], categories=sorted({c for _, c in (hits or [])}))
    return SimpleNamespace(semantic=semantic, lexical=lexical)


# -- signals ----------------------------------------------------------------

def test_reduce_visual_maxpools_over_frames() -> None:
    """The strongest frame sets the video-level violence score (max, not mean)."""
    sig = reduce_visual([_frame(0.1), _frame(0.9), _frame(0.2)])
    assert sig.present is True
    assert sig.scores["violence"] == pytest.approx(0.9)


def test_reduce_audio_maps_all_violent_sounds_to_violence() -> None:
    """gunshot/scream both collapse into the canonical 'violence' category."""
    sig = reduce_audio([_audio(gunshot=0.8), _audio(scream=0.95)])
    assert sig.scores["violence"] == pytest.approx(0.95)


def test_empty_modality_is_marked_absent() -> None:
    """No data -> present=False, so fusion can skip it rather than read it as safe."""
    assert reduce_audio([]).present is False
    assert reduce_text(None).present is False


# -- heuristic fast-path (4.1) ----------------------------------------------

def test_banned_word_short_circuits_to_violation() -> None:
    """A lexical hit makes V1Moderator return immediately via the heuristic source."""
    text = [_text({"hate_speech": 0.1}, hits=[("杀了你", "violence")])]
    verdict = V1Moderator().moderate(text_verdicts=text)
    assert verdict.is_violating is True
    assert verdict.source == "heuristic"
    assert verdict.category == "violence"
    assert verdict.confidence >= 0.99


# -- weighted voting (4.2) --------------------------------------------------

def test_soft_voting_weighted_average() -> None:
    """Soft voting averages present modalities' scores by their weights."""
    fusion = WeightedVotingFusion(weights={"text": 1.0, "visual": 1.0}, threshold=0.5)
    signals = [reduce_visual([_frame(0.8)]), reduce_text([_text({"violence": 0.4})])]
    verdict = fusion.fuse(signals)
    # equal weights -> (0.8 + 0.4) / 2 = 0.6 for violence
    assert verdict.scores["violence"] == pytest.approx(0.6)
    assert verdict.is_violating is True
    assert verdict.category == "violence"


def test_missing_modality_does_not_dilute_score() -> None:
    """An absent audio track is ignored, not averaged in as a zero."""
    fusion = WeightedVotingFusion(weights={"text": 1.0, "audio": 1.0})
    signals = [reduce_text([_text({"violence": 0.9})]), reduce_audio([])]
    verdict = fusion.fuse(signals)
    assert verdict.scores["violence"] == pytest.approx(0.9)  # not 0.45


# -- evaluation (4.3) -------------------------------------------------------

def test_binary_metrics_precision_recall() -> None:
    """Confusion-matrix counts turn into the right precision/recall."""
    y_true = [True, True, False, False]
    y_pred = [True, False, True, False]   # 1 TP, 1 FN, 1 FP, 1 TN
    m = binary_metrics(y_true, y_pred)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)


def test_find_bad_cases_splits_fp_and_fn() -> None:
    """Mispredictions sort into false-positive and false-negative buckets."""
    records = [
        ("a", True, SimpleNamespace(is_violating=False)),   # FN
        ("b", False, SimpleNamespace(is_violating=True)),   # FP
        ("c", True, SimpleNamespace(is_violating=True)),    # correct
    ]
    buckets = find_bad_cases(records)
    assert [c.clip_id for c in buckets["false_negatives"]] == ["a"]
    assert [c.clip_id for c in buckets["false_positives"]] == ["b"]


def test_cross_modal_conflict_flags_horror_sfx_on_comedy() -> None:
    """Loud audio + calm visuals is flagged as a conflict (likely false positive)."""
    signals = [reduce_audio([_audio(gunshot=0.95)]), reduce_visual([_frame(0.05)])]
    conflict = cross_modal_conflict(signals)
    assert conflict.is_conflict is True
    assert conflict.loudest == "audio"
    assert conflict.quietest == "visual"


def test_agreeing_modalities_are_not_conflict() -> None:
    """When modalities agree (both high), there is no conflict to flag."""
    signals = [reduce_audio([_audio(gunshot=0.9)]), reduce_visual([_frame(0.85)])]
    assert cross_modal_conflict(signals).is_conflict is False
