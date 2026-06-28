"""V2 ch.5 — CLIP zero-shot screening: rank frames against a prompt pool.

The V1 visual expert needed a *trained* head because ImageNet has no "violence"
class. CLIP sidesteps that entirely: it was trained to put an image and its
describing sentence near each other in a shared space, so we can score a frame
against free-text prompts like *"a photo of people fighting"* with **no training**
(ROADMAP 5.1 "抛弃专家模型").

How it works
------------
1. **Prompt pool (5.2)** — for each violation category we write several "violating"
   prompts, plus a shared pool of "safe" contrast prompts (the brief's
   *"fighting" vs "hugging"*). The safe prompts matter: they absorb probability
   mass for benign frames so the violating prompts don't always "win" by default.
2. **Cosine similarity (5.3)** — encode the frame and every prompt into CLIP's
   shared space, take cosine similarities, temperature-scale (CLIP's logit_scale)
   and softmax across all prompts. The mass landing on a category's violating
   prompts is that category's zero-shot violation score.

The per-frame output exposes a ``.scores`` dict over canonical categories, so it
drops straight into ``sentinelai.fusion.reduce_visual`` as an alternative to the
CNN visual expert — a fast zero-shot pre-filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .hardware import get_device

# OpenAI CLIP (small/fast). Swap to a larger CLIP or "OFA-Sys/chinese-clip-..."
# for Chinese prompts; the prompt language is independent of the video.
DEFAULT_MODEL = "openai/clip-vit-base-patch32"

ImageInput = Union[str, Path, np.ndarray, object]

# Violating prompts per canonical category. Several phrasings per category make the
# zero-shot signal more robust than any single sentence.
VIOLATION_PROMPTS: dict[str, tuple[str, ...]] = {
    "violence": (
        "a photo of people physically fighting",
        "a photo of a person violently attacking another person",
        "a photo of someone threatening others with a weapon",
        "a photo of a bloody, graphic injury",
    ),
    "nsfw": (
        "a photo with explicit sexual content",
        "a photo of a fully nude person",
        "a pornographic image",
    ),
}

# Safe contrast prompts — benign scenes that should "win" for normal frames. The
# kitchen prompt is deliberate: it pulls "cooking / cutting vegetables" away from
# the "weapon" prompts, the exact confusion ch.6 also targets.
SAFE_PROMPTS: tuple[str, ...] = (
    "a photo of people hugging",
    "a photo of people talking calmly",
    "a photo of people playing sports",
    "a photo of food being prepared in a kitchen",
    "a normal everyday photo",
    "a photo of a peaceful outdoor scene",
)


@dataclass(frozen=True)
class Prompt:
    """One entry of the prompt pool: its text, category, and whether it's violating."""

    text: str
    category: str          # a canonical category, or "safe"
    violating: bool


def build_prompts(
    violation_prompts: dict[str, tuple[str, ...]] = VIOLATION_PROMPTS,
    safe_prompts: tuple[str, ...] = SAFE_PROMPTS,
) -> list[Prompt]:
    """Flatten the prompt dicts into one ordered list of :class:`Prompt`.

    The order here defines the column order of the similarity matrix, so we build
    it once and keep it fixed for the screener's lifetime.
    """
    prompts = [
        Prompt(text=text, category=category, violating=True)
        for category, texts in violation_prompts.items()
        for text in texts
    ]
    prompts += [Prompt(text=text, category="safe", violating=False) for text in safe_prompts]
    return prompts


def _aggregate_prompt_probs(
    probs: Sequence[float], prompts: Sequence[Prompt]
) -> tuple[dict[str, float], float, str]:
    """Turn one frame's per-prompt softmax probabilities into category scores.

    For each category we **sum** the probability mass on its violating prompts
    (sum, not max: the prompts are alternative phrasings of the same concept, so a
    frame split 0.3/0.3 across two "violence" prompts is 0.6 violent). The overall
    violation probability is the total mass on all violating prompts (equivalently
    ``1 - mass on safe prompts``). Pure function -> unit-testable without CLIP.

    Returns ``(category_scores, overall_violation_prob, most_similar_prompt_text)``.
    """
    scores: dict[str, float] = {}
    overall = 0.0
    for prompt, prob in zip(prompts, probs):
        if prompt.violating:
            scores[prompt.category] = scores.get(prompt.category, 0.0) + float(prob)
            overall += float(prob)
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    return scores, overall, prompts[top_idx].text


@dataclass(frozen=True)
class ClipFrameScore:
    """Zero-shot screening result for one frame.

    Attributes:
        index:          frame position in the input batch.
        violation_prob: total probability mass on violating prompts.
        scores:         per-category violation score (sums of violating-prompt mass).
        top_prompt:     the single most-similar prompt (for explainability/debugging).
    """

    index: int
    violation_prob: float
    scores: dict[str, float]
    top_prompt: str


class ClipScreener:
    """CLIP-based zero-shot frame screener.

    Usage::

        screener = ClipScreener()
        scores = screener.score_frames(frame_paths)     # works with no training
        # or feed straight into V1 fusion as a visual signal:
        #   reduce_visual(screener.score_frames(frames))
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prompts: list[Prompt] | None = None,
        device: str | None = None,
    ) -> None:
        """Load CLIP and **pre-encode the prompt pool once** (it never changes).

        Caching the text features is the key efficiency win: encoding prompts is as
        expensive as encoding images, but the pool is fixed, so we pay for it a
        single time and then every frame is just one image-encode + a matmul.
        """
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or get_device()
        self.prompts = prompts or build_prompts()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

        # Pre-compute L2-normalised text features for the whole prompt pool.
        text_inputs = self.processor(
            text=[p.text for p in self.prompts], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            tf = self.model.get_text_features(**text_inputs)
        self.text_features = tf / tf.norm(dim=-1, keepdim=True)
        # CLIP stores a learned temperature; exp() turns it into the logit scale
        # that sharpens the cosine similarities before softmax.
        self.logit_scale = self.model.logit_scale.exp()

    def _to_pil(self, image: ImageInput):
        """Coerce a path / RGB array / PIL image into a PIL RGB image for CLIP."""
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"unsupported image type: {type(image)!r}")

    def score_frames(
        self, images: Sequence[ImageInput], batch_size: int = 32
    ) -> list[ClipFrameScore]:
        """Score each frame against the prompt pool via cosine similarity + softmax.

        For every batch: encode images, L2-normalise, compute scaled cosine
        similarities against the cached text features, softmax across prompts, then
        aggregate to category scores with :func:`_aggregate_prompt_probs`.
        """
        import torch

        results: list[ClipFrameScore] = []
        for start in range(0, len(images), batch_size):
            chunk = [self._to_pil(im) for im in images[start : start + batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            # (batch, num_prompts) cosine similarities, temperature-scaled.
            logits = self.logit_scale * feats @ self.text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()
            for offset, row in enumerate(probs):
                scores, overall, top = _aggregate_prompt_probs(row, self.prompts)
                results.append(
                    ClipFrameScore(
                        index=start + offset,
                        violation_prob=overall,
                        scores=scores,
                        top_prompt=top,
                    )
                )
        return results
