"""Tests for the audio expert's pure logic.

These cover the parts that need neither a downloaded model nor torch — the
AudioSet->category aggregation and the windowing — so they run instantly offline.
The AST model path itself is exercised separately on the GPU box.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinelai.audio_expert import TARGET_SR, _aggregate_violation, _window


def test_aggregate_maps_audioset_labels_to_categories() -> None:
    """Gunshot/explosion/scream AudioSet labels light up the right categories."""
    class_probs = {
        "Speech": 0.9,
        "Gunshot, gunfire": 0.8,
        "Machine gun": 0.6,
        "Explosion": 0.3,
        "Screaming": 0.7,
        "Music": 0.5,
    }
    scores = _aggregate_violation(class_probs)
    # "gunshot" takes the MAX over its synonyms (0.8), not the sum.
    assert scores["gunshot"] == pytest.approx(0.8)
    assert scores["explosion"] == pytest.approx(0.3)
    assert scores["scream"] == pytest.approx(0.7)


def test_aggregate_is_zero_when_no_abnormal_sound() -> None:
    """Ordinary sounds (speech/music) produce zero violation across categories."""
    scores = _aggregate_violation({"Speech": 0.95, "Music": 0.4, "Silence": 0.1})
    assert set(scores) == {"gunshot", "explosion", "scream"}
    assert all(v == 0.0 for v in scores.values())


def test_window_short_audio_is_single_clip() -> None:
    """Audio shorter than one window comes back as exactly one clip starting at 0."""
    wav = np.zeros(TARGET_SR * 3, dtype=np.float32)  # 3 s < 10 s window
    windows = _window(wav, TARGET_SR, window_s=10.0)
    assert len(windows) == 1
    assert windows[0][0] == 0.0


def test_window_long_audio_is_split_with_timestamps() -> None:
    """A 25 s track at 10 s windows -> three windows at 0/10/20 s, covering the tail."""
    wav = np.zeros(TARGET_SR * 25, dtype=np.float32)
    windows = _window(wav, TARGET_SR, window_s=10.0)
    starts = [round(start, 3) for start, _ in windows]
    assert starts == [0.0, 10.0, 20.0]
    # The final (tail) window holds the leftover 5 s, not a full 10 s.
    assert len(windows[-1][1]) == TARGET_SR * 5
