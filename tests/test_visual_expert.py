"""Smoke tests for the visual expert.

These run on CPU and use ``pretrained=False`` so they need no network and no GPU:
we only check the *plumbing* (shapes, probability normalization, save/load,
input-type handling), not model accuracy — accuracy is meaningless until the head
is trained on real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinelai.visual_expert import VisualExpert, available_backbones


def _fake_frames(n: int = 3) -> list[np.ndarray]:
    """A few random RGB frames as uint8 H×W×3 arrays (stand-ins for real frames)."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(n)]


@pytest.mark.parametrize("backbone", available_backbones())
def test_extract_features_shape(backbone: str) -> None:
    """Embeddings come back as (num_frames, feature_dim) float32."""
    expert = VisualExpert(backbone=backbone, device="cpu", pretrained=False)
    frames = _fake_frames(3)
    feats = expert.extract_features(frames, batch_size=2)
    assert feats.shape == (3, expert.feature_dim)
    assert feats.dtype == np.float32


def test_predict_probabilities_are_valid() -> None:
    """Each frame yields a distribution that sums to 1 and a violation prob in [0,1]."""
    cats = ("safe", "violence", "nsfw")
    expert = VisualExpert(categories=cats, device="cpu", pretrained=False)
    preds = expert.predict(_fake_frames(4))
    assert len(preds) == 4
    for i, p in enumerate(preds):
        assert p.index == i
        assert set(p.scores) == set(cats)
        assert pytest.approx(sum(p.scores.values()), abs=1e-5) == 1.0
        assert 0.0 <= p.violation_prob <= 1.0
        # violation_prob is exactly the mass not on the "safe" class.
        assert pytest.approx(p.violation_prob, abs=1e-5) == 1.0 - p.scores["safe"]


def test_empty_input_is_handled() -> None:
    """Empty inputs return empty results, not errors."""
    expert = VisualExpert(device="cpu", pretrained=False)
    assert expert.extract_features([]).shape == (0, expert.feature_dim)
    assert expert.predict([]) == []


def test_save_and_load_roundtrip(tmp_path) -> None:
    """A saved checkpoint loads back, and loading marks the head as trained."""
    expert = VisualExpert(device="cpu", pretrained=False)
    ckpt = tmp_path / "head.pt"
    expert.save(ckpt)

    reloaded = VisualExpert(device="cpu", pretrained=False)
    assert reloaded.head_is_trained is False
    reloaded.load(ckpt)
    assert reloaded.head_is_trained is True


def test_load_rejects_mismatched_categories(tmp_path) -> None:
    """Loading a checkpoint into a different category set is refused."""
    ckpt = tmp_path / "head.pt"
    VisualExpert(categories=("safe", "violence"), device="cpu", pretrained=False).save(ckpt)

    mismatched = VisualExpert(categories=("safe", "nsfw"), device="cpu", pretrained=False)
    with pytest.raises(ValueError):
        mismatched.load(ckpt)
