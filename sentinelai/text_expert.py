"""Text expert (NLP): lexical + semantic violation judgement on transcripts.

Pipeline role (ROADMAP 3.3)
---------------------------
This is the *text* expert of the V1 expert-model stack. Its input is the **ASR
transcript** of a video's speech (transcription itself — faster-whisper — is an
upstream module, out of scope here). It decides whether the spoken text is
violating, combining two complementary signals the brief calls for:

* **Lexical (词法)** — match against a bilingual banned-term lexicon. Fast, exact,
  needs no model, and is fully explainable ("matched the word X"). Catches
  explicit slurs / banned words but is blind to context.
* **Semantic (语义)** — a multilingual Transformer toxicity classifier. Slower but
  understands context, euphemism and implicit hate that no word list catches.

Model choice
------------
The brief mentions RoBERTa-wwm / DeBERTa. Those are *base* language models with no
violation head, so they would need training first (like the visual expert). To be
useful out of the box we default instead to a multilingual (zh+en) toxicity
classifier already fine-tuned on Jigsaw (like the audio expert, which reused
AudioSet). The model is swappable, so once you fine-tune a RoBERTa-wwm / DeBERTa
head on your own data you can drop it in via ``model_name``.

How text becomes a verdict
--------------------------
    transcript --lexicon scan----------> lexical hits  ─┐
               --tokenizer--> Transformer-> per-label    ├─> combined verdict
                              (sigmoid)    probabilities ─┘   (is_violating, category, reason)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .hardware import get_device

# A multilingual (incl. Chinese) toxicity classifier, fine-tuned on Jigsaw. Emits
# independent per-label probabilities (toxicity / threat / insult / identity / ...).
DEFAULT_MODEL = "unitary/multilingual-toxic-xlm-roberta"

# Map our violation categories to substrings of the model's raw label names. The
# model's exact labels vary, so we match loosely (e.g. "identity" catches
# "identity_attack") against ``model.config.id2label`` at runtime.
SEMANTIC_CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "hate_speech": ("identity", "insult"),
    "violence": ("threat",),
    "sexual": ("sexual", "obscene"),
    "toxic": ("toxic", "severe"),
}

# A tiny illustrative bilingual lexicon. Real deployments load a curated list via
# ``Lexicon.from_file``; kept deliberately small (and mild) here. Terms are matched
# case-insensitively, with word boundaries for ASCII terms (so "ass" won't fire on
# "class") and plain substring for Chinese (which has no word boundaries).
DEFAULT_LEXICON: dict[str, tuple[str, ...]] = {
    "violence": ("kill you", "杀了你", "弄死你", "打死你"),
    "hate_speech": (),   # populate from a curated file — slurs are context/locale specific
    "sexual": (),
}


def _term_matches(term: str, text_low: str) -> bool:
    """True if ``term`` occurs in already-lowercased ``text_low``.

    ASCII terms are matched on **word boundaries** via regex so a short term like
    "ass" does not falsely fire inside "class". Terms containing non-ASCII
    characters (e.g. Chinese) are matched as plain substrings, because CJK text is
    not whitespace-delimited and has no word boundaries to anchor on.
    """
    if term.isascii():
        return re.search(rf"\b{re.escape(term)}\b", text_low) is not None
    return term in text_low


@dataclass(frozen=True)
class LexicalResult:
    """Outcome of the word-list scan.

    Attributes:
        hits:       list of (matched_term, category) pairs found in the text.
        categories: distinct categories that were triggered.
        score:      1.0 if any banned term matched, else 0.0 (lexical is a hard
                    binary signal, not a graded probability).
    """

    hits: list[tuple[str, str]]
    categories: list[str]
    score: float


def lexical_scan(text: str, lexicon: dict[str, tuple[str, ...]] = DEFAULT_LEXICON) -> LexicalResult:
    """Scan text for banned terms. Pure function — no model, instant, testable.

    Lowercases the text once, then checks every lexicon term against it using
    :func:`_term_matches`. This is the "词法" half: explicit, explainable, and
    language-agnostic, but unaware of context.
    """
    text_low = text.lower()
    hits: list[tuple[str, str]] = []
    for category, terms in lexicon.items():
        for term in terms:
            if _term_matches(term.lower(), text_low):
                hits.append((term, category))
    categories = sorted({c for _, c in hits})
    return LexicalResult(hits=hits, categories=categories, score=1.0 if hits else 0.0)


def _aggregate_semantic(
    label_probs: dict[str, float],
    mapping: dict[str, tuple[str, ...]] = SEMANTIC_CATEGORY_MAP,
) -> dict[str, float]:
    """Collapse raw model labels into our violation categories.

    For each category, take the **max** probability over the model labels mapped
    to it (max because the labels are independent multi-label signals and we want
    "is any aspect of this category present"). Pure function, so it is unit-tested
    without downloading the model.
    """
    scores: dict[str, float] = {}
    for category, needles in mapping.items():
        best = 0.0
        for label, prob in label_probs.items():
            low = label.lower()
            if any(needle in low for needle in needles):
                best = max(best, prob)
        scores[category] = best
    return scores


@dataclass(frozen=True)
class TextVerdict:
    """The combined verdict for one transcript segment.

    Attributes:
        text:           the input text judged.
        is_violating:   violation_prob >= threshold.
        violation_prob: combined headline score (max of lexical and semantic).
        category:       most relevant violation category.
        lexical:        the raw lexical scan result.
        semantic:       per-category semantic probabilities.
        reason:         short human-readable explanation (for the structured output).
    """

    text: str
    is_violating: bool
    violation_prob: float
    category: str | None
    lexical: LexicalResult
    semantic: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class TextExpert:
    """Bilingual transcript moderator: lexicon + Transformer toxicity classifier.

    Typical use::

        expert = TextExpert()
        verdict = expert.predict("我要杀了你")     # lexical + semantic, works zero-shot
        if verdict.is_violating:
            print(verdict.category, verdict.reason)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lexicon: dict[str, tuple[str, ...]] = DEFAULT_LEXICON,
        device: str | None = None,
        threshold: float = 0.5,
        max_tokens: int = 512,
    ) -> None:
        """Load the tokenizer + classifier and keep the lexicon for scanning.

        ``transformers`` is imported lazily so the pure helpers above (and their
        tests) don't require the dependency or pay the import cost.

        Args:
            model_name: a HF sequence-classification model (multi-label toxicity).
            lexicon:    category -> banned terms for the lexical pass.
            device:     "cuda"/"cpu"; defaults to the host's best device.
            threshold:  probability at/above which text is flagged violating.
            max_tokens: transcripts longer than this are truncated (callers should
                        feed ASR *segments* rather than one giant blob).
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or get_device()
        self.lexicon = lexicon
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        # index -> raw label name (e.g. 0 -> "toxicity").
        self.id2label: dict[int, str] = self.model.config.id2label

    def _semantic_scores_batch(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Run the Transformer over a batch of texts -> per-category probabilities.

        Tokenizes with padding/truncation, does one forward pass, applies
        **sigmoid** (multi-label: a text can be both an insult AND a threat), then
        folds the raw labels into our categories via :func:`_aggregate_semantic`.
        """
        import torch

        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits          # (B, num_labels)
        probs = torch.sigmoid(logits).cpu().numpy()     # independent per-label probs
        results: list[dict[str, float]] = []
        for row in probs:
            label_probs = {self.id2label[i]: float(p) for i, p in enumerate(row)}
            results.append(_aggregate_semantic(label_probs))
        return results

    def _combine(self, text: str, lexical: LexicalResult, semantic: dict[str, float]) -> TextVerdict:
        """Fuse the lexical and semantic signals into a single verdict.

        The headline probability is the **max** of the two signals: a banned-word
        hit (lexical score 1.0) alone is enough to flag, and otherwise we defer to
        the strongest semantic category. The chosen category prefers an explicit
        lexical hit (more explainable) and falls back to the top semantic one.
        """
        sem_cat = max(semantic, key=semantic.__getitem__) if semantic else None
        sem_prob = semantic.get(sem_cat, 0.0) if sem_cat else 0.0
        violation_prob = max(sem_prob, lexical.score)
        category = lexical.categories[0] if lexical.hits else sem_cat

        # Build a short, honest explanation from whichever signals fired.
        parts: list[str] = []
        if lexical.hits:
            terms = ", ".join(sorted({t for t, _ in lexical.hits}))
            parts.append(f"banned term(s): {terms}")
        if sem_cat and sem_prob >= self.threshold:
            parts.append(f"semantic {sem_cat}={sem_prob:.2f}")
        reason = "; ".join(parts) if parts else "no violation signal"

        return TextVerdict(
            text=text,
            is_violating=violation_prob >= self.threshold,
            violation_prob=violation_prob,
            category=category,
            lexical=lexical,
            semantic=semantic,
            reason=reason,
        )

    def predict(self, text: str) -> TextVerdict:
        """Judge a single transcript segment (lexical + semantic)."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[TextVerdict]:
        """Judge several segments at once; the semantic model batches efficiently."""
        if not texts:
            return []
        semantics = self._semantic_scores_batch(texts)
        return [
            self._combine(text, lexical_scan(text, self.lexicon), semantic)
            for text, semantic in zip(texts, semantics)
        ]
