"""4.1 Heuristic fast-path: high-confidence rules that short-circuit fusion.

Some violations are so unambiguous that running the full fusion stack is wasteful.
The classic example from the brief: if NLP matched an explicit high-risk banned
word, we can output "violating" immediately and skip the CV/Audio/MLP math.

Each rule inspects the *raw* expert outputs (which carry stronger evidence than
the reduced signals — e.g. exactly which banned word matched) and either returns a
:class:`HeuristicHit` (fire) or ``None`` (defer to fusion). Rules are tried in
order; the first hit wins.

Design note: heuristics here are intentionally **conservative** — they only fire
on near-certain signals (an exact banned-word match), because a wrong fast-path
verdict can't be corrected downstream. Riskier single-modality shortcuts (e.g.
"loud bang => violence") are deliberately NOT default rules: a horror sound effect
over a comedy clip would make them misfire (see evaluate.cross_modal_conflict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class HeuristicHit:
    """A fired fast-path rule.

    Attributes:
        category:   the violation category to report.
        confidence: how sure the rule is (banned-word hits are ~certain).
        reason:     human-readable explanation for the structured output.
        rule:       name of the rule that fired (for auditing / metrics).
    """

    category: str
    confidence: float
    reason: str
    rule: str


def banned_word_rule(
    *, text_verdicts: Sequence[Any] | None = None, **_: Any
) -> HeuristicHit | None:
    """Fire if any transcript segment matched a lexicon banned term.

    A lexical hit is an exact, explainable match against the curated banned-word
    list, so we treat it as a near-certain violation and report the first match's
    category. Returns ``None`` if there are no transcripts or no lexical hits.

    Accepts (and ignores) other modalities' kwargs so every rule shares one
    signature and the dispatcher can call them uniformly.
    """
    for verdict in text_verdicts or []:
        # `.lexical.hits` is a list of (term, category); non-empty == matched.
        if getattr(verdict, "lexical", None) and verdict.lexical.hits:
            term, category = verdict.lexical.hits[0]
            return HeuristicHit(
                category=category,
                confidence=0.99,
                reason=f"banned term matched in speech: {term!r}",
                rule="banned_word",
            )
    return None


# The default rule chain. A rule is any callable taking the keyword expert outputs
# and returning HeuristicHit | None; we wrap each so they share one call signature.
DEFAULT_RULES: tuple[Callable[..., HeuristicHit | None], ...] = (banned_word_rule,)


def apply_heuristics(
    *,
    text_verdicts: Sequence[Any] | None = None,
    frames: Sequence[Any] | None = None,
    audio_events: Sequence[Any] | None = None,
    rules: Sequence[Callable[..., HeuristicHit | None]] = DEFAULT_RULES,
) -> HeuristicHit | None:
    """Run the rule chain; return the first hit, or ``None`` to fall through to fusion.

    Expert outputs are passed by keyword so rules can pick whichever modality they
    need. The default chain only uses ``text_verdicts``; ``frames`` / ``audio_events``
    are accepted so custom rules can use them without changing this signature.
    """
    for rule in rules:
        # Every rule shares one signature and ignores kwargs it doesn't use.
        hit = rule(text_verdicts=text_verdicts, frames=frames, audio_events=audio_events)
        if hit is not None:
            return hit
    return None
