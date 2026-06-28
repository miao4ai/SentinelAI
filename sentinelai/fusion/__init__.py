"""V1 late fusion: heuristics (4.1) + simple fusion (4.2) + evaluation (4.3).

Everything exported here is torch-free and imports cheaply. The trainable MLP
alternative needs torch, so import it explicitly when wanted::

    from sentinelai.fusion.mlp_fusion import MLPFusion
"""

from .evaluate import (
    BadCase,
    Conflict,
    Metrics,
    binary_metrics,
    cross_modal_conflict,
    find_bad_cases,
)
from .fusion import DEFAULT_WEIGHTS, FusedVerdict, V1Moderator, WeightedVotingFusion
from .heuristics import HeuristicHit, apply_heuristics, banned_word_rule
from .signals import (
    CANONICAL_CATEGORIES,
    ModalitySignal,
    reduce_audio,
    reduce_text,
    reduce_visual,
)

__all__ = [
    # signals
    "CANONICAL_CATEGORIES",
    "ModalitySignal",
    "reduce_visual",
    "reduce_audio",
    "reduce_text",
    # heuristics (4.1)
    "HeuristicHit",
    "banned_word_rule",
    "apply_heuristics",
    # fusion (4.2)
    "FusedVerdict",
    "WeightedVotingFusion",
    "V1Moderator",
    "DEFAULT_WEIGHTS",
    # evaluation (4.3)
    "Metrics",
    "binary_metrics",
    "BadCase",
    "find_bad_cases",
    "Conflict",
    "cross_modal_conflict",
]
