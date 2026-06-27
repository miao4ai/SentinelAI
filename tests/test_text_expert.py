"""Tests for the text expert's pure logic.

Covers the lexical scan and the semantic-label aggregation — no model download,
no torch — so they run instantly offline. The Transformer path runs on the GPU box.
"""

from __future__ import annotations

import pytest

from sentinelai.text_expert import (
    _aggregate_semantic,
    _term_matches,
    lexical_scan,
)


def test_term_matches_uses_word_boundaries_for_ascii() -> None:
    """ASCII terms match whole words only — 'ass' must not fire inside 'class'."""
    assert _term_matches("ass", "what an ass") is True
    assert _term_matches("ass", "this class is fun") is False


def test_term_matches_substring_for_chinese() -> None:
    """Chinese terms match as substrings (CJK has no word boundaries)."""
    assert _term_matches("杀了你", "我真的要杀了你信不信") is True
    assert _term_matches("杀了你", "今天天气很好") is False


def test_lexical_scan_flags_bilingual_violence() -> None:
    """Both English and Chinese violent threats are caught by the default lexicon."""
    en = lexical_scan("I will kill you")
    assert en.score == 1.0 and "violence" in en.categories

    zh = lexical_scan("再废话我就杀了你")
    assert zh.score == 1.0 and "violence" in zh.categories


def test_lexical_scan_clean_text_scores_zero() -> None:
    """Benign text produces no hits and a zero score."""
    result = lexical_scan("thanks for watching, see you tomorrow")
    assert result.hits == []
    assert result.score == 0.0


def test_aggregate_semantic_maps_labels_to_categories() -> None:
    """Raw toxicity labels collapse onto our categories via max-pooling."""
    label_probs = {
        "toxicity": 0.9,
        "threat": 0.8,
        "identity_attack": 0.6,
        "insult": 0.7,
        "sexual_explicit": 0.2,
    }
    scores = _aggregate_semantic(label_probs)
    assert scores["violence"] == pytest.approx(0.8)          # threat
    assert scores["hate_speech"] == pytest.approx(0.7)       # max(identity, insult)
    assert scores["sexual"] == pytest.approx(0.2)            # sexual_explicit
    assert scores["toxic"] == pytest.approx(0.9)             # toxicity


def test_aggregate_semantic_clean_text_is_zero() -> None:
    """When no toxic label is present, every category is zero."""
    scores = _aggregate_semantic({"toxicity": 0.01, "threat": 0.0})
    assert scores["hate_speech"] == 0.0
    assert scores["sexual"] == 0.0
