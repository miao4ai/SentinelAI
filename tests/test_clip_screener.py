"""Tests for the CLIP screener's pure logic (prompt pool + aggregation).

No CLIP download and no torch — just the prompt bookkeeping and the
softmax-mass aggregation. The model path runs on the GPU box. (Imports numpy
transitively, so run where the project deps are installed.)
"""

from __future__ import annotations

import pytest

from sentinelai.clip_screener import (
    Prompt,
    _aggregate_prompt_probs,
    build_prompts,
)


def test_build_prompts_orders_violating_then_safe() -> None:
    """Violating prompts come first (grouped by category), safe prompts last."""
    prompts = build_prompts(
        violation_prompts={"violence": ("a", "b"), "nsfw": ("c",)},
        safe_prompts=("d", "e"),
    )
    assert [p.category for p in prompts] == ["violence", "violence", "nsfw", "safe", "safe"]
    assert [p.violating for p in prompts] == [True, True, True, False, False]


def test_aggregate_sums_category_mass_and_overall() -> None:
    """Per-category score sums its violating prompts; overall = 1 - safe mass."""
    prompts = [
        Prompt("fight", "violence", True),
        Prompt("attack", "violence", True),
        Prompt("nude", "nsfw", True),
        Prompt("hugging", "safe", False),
    ]
    scores, overall, top = _aggregate_prompt_probs([0.3, 0.3, 0.1, 0.3], prompts)
    assert scores["violence"] == pytest.approx(0.6)   # two violence prompts summed
    assert scores["nsfw"] == pytest.approx(0.1)
    assert overall == pytest.approx(0.7)              # 1 - 0.3 safe
    assert top == "fight"                             # most-similar prompt


def test_aggregate_safe_frame_has_low_violation() -> None:
    """A frame whose mass sits on the safe prompt scores near-zero violation."""
    prompts = [
        Prompt("fight", "violence", True),
        Prompt("nude", "nsfw", True),
        Prompt("a normal photo", "safe", False),
    ]
    _, overall, top = _aggregate_prompt_probs([0.05, 0.05, 0.90], prompts)
    assert overall == pytest.approx(0.10)
    assert top == "a normal photo"
