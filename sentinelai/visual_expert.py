"""Visual expert (CV): per-frame embeddings and violation probability.

Pipeline role (ROADMAP 3.1)
---------------------------
This is the *visual* expert of the V1 expert-model stack. It turns a single
video frame into two things:

1. a frame-level **embedding** — a fixed-length feature vector, and
2. a per-frame **violation probability**.

Two interchangeable ImageNet-pretrained backbones are supported so you can run
both and compare them head-to-head (the task explicitly asks for this):

* ``resnet50``           — ResNet-50,        2048-d embedding
* ``efficientnet_v2_s``  — EfficientNetV2-S, 1280-d embedding

Why there is a separate "head"
------------------------------
The backbones are pretrained on **ImageNet** (1000 everyday-object classes), so
on their own they have no concept of "violence". The violation score is produced
by a small linear **classification head** stacked on top of the backbone:

    frame --preprocess--> CNN backbone --> embedding --> head --> logits --> softmax

Until that head is trained on labelled frames (UCF-Crime / XD-Violence — see
ROADMAP ch.4 / ch.8), its probabilities are **randomly initialised and therefore
meaningless**. The feature-extraction half (:meth:`VisualExpert.extract_features`)
is useful immediately — e.g. as cached inputs for training or for late fusion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights

from .hardware import get_device

# An "image" the expert accepts: a file path, a decoded RGB array, or a PIL image.
# (PIL is typed loosely as object to avoid importing PIL at module import time.)
ImageInput = Union[str, Path, np.ndarray, object]

# The backbones we expose. Keep this tuple as the single source of truth so the
# CLI / tests / callers all agree on the valid names.
_BACKBONES = ("resnet50", "efficientnet_v2_s")


def available_backbones() -> tuple[str, ...]:
    """List the backbone names accepted by ``VisualExpert(backbone=...)``.

    Both are ImageNet-pretrained CNNs; ``resnet50`` is the classic strong
    baseline, ``efficientnet_v2_s`` is newer and usually a bit more accurate per
    FLOP. Exposed as a function so callers never hard-code the names.
    """
    return _BACKBONES


def _build_backbone(name: str, pretrained: bool = True):
    """Construct a *headless* pretrained CNN plus its matching preprocessing.

    "Headless" means we delete the network's built-in 1000-class ImageNet
    classifier and replace it with :class:`~torch.nn.Identity`. After that, a
    forward pass returns the **penultimate-layer embedding** (the features the
    classifier would have consumed) instead of ImageNet class logits — exactly
    what we want as a reusable frame representation.

    Returns ``(feature_extractor, feature_dim, preprocess)``:
      * ``feature_extractor`` — the CNN ending at the embedding,
      * ``feature_dim``       — length of that embedding (2048 for ResNet-50,
        1280 for EfficientNetV2-S),
      * ``preprocess``        — the *exact* resize/crop/normalize transform the
        weights were trained with. Using the wrong normalization silently
        degrades accuracy, so we always take it straight from the weights enum.
    """
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        net = tv_models.resnet50(weights=weights if pretrained else None)
        # ResNet's classifier is the single fully-connected layer `fc`. Its input
        # size is the embedding dimension we want to keep.
        feature_dim = net.fc.in_features          # 2048
        net.fc = nn.Identity()                    # forward() now returns the embedding
    elif name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        net = tv_models.efficientnet_v2_s(weights=weights if pretrained else None)
        # EfficientNet's classifier is a Sequential(Dropout, Linear); the Linear's
        # input size is the embedding dimension.
        feature_dim = net.classifier[1].in_features  # 1280
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {_BACKBONES}")
    # `.transforms()` builds the preprocessing pipeline; it needs no downloaded
    # weights, so it works even when pretrained=False (used by fast offline tests).
    return net, feature_dim, weights.transforms()


@dataclass(frozen=True)
class FramePrediction:
    """The violation verdict for one frame.

    Attributes:
        index:          position of this frame within the batch passed to predict().
        violation_prob: P(frame is violating) == 1 - P(safe). The headline number.
        category:       the single most-likely class (may be the "safe" class).
        scores:         full per-class probability distribution (sums to 1).
    """

    index: int
    violation_prob: float
    category: str
    scores: dict[str, float]


class VisualExpert:
    """Frame-level violation scorer built on a pretrained CNN backbone.

    Typical use::

        expert = VisualExpert(backbone="resnet50")
        feats  = expert.extract_features(frame_paths)   # (N, 2048) — usable now
        preds  = expert.predict(frame_paths)            # needs a trained head

    By convention ``categories[0]`` is the "safe"/normal class, so the violation
    probability is ``1 - P(categories[0])``.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        categories: Sequence[str] = ("safe", "violence"),
        weights_path: str | Path | None = None,
        device: str | None = None,
        pretrained: bool = True,
    ) -> None:
        """Assemble backbone + classification head and move them onto the device.

        Args:
            backbone:     one of :func:`available_backbones`.
            categories:   class names; index 0 MUST be the safe/normal class.
            weights_path: optional checkpoint saved by :meth:`save` — loads a
                          *trained* head (and any fine-tuned backbone weights).
            device:       "cuda" / "cpu"; defaults to the host's best device.
            pretrained:   load ImageNet weights for the backbone (set False only
                          for fast offline tests).
        """
        if len(categories) < 2:
            raise ValueError("need at least 2 categories (safe + ≥1 violating)")

        self.backbone_name = backbone
        self.categories = tuple(categories)
        self.safe_category = self.categories[0]
        # Pick the device once and pin everything to it; mixing devices errors out.
        self.device = device or get_device()

        # Build the feature extractor and remember the embedding width so the head
        # can be sized to match it.
        self.backbone, self.feature_dim, self.preprocess = _build_backbone(
            backbone, pretrained=pretrained
        )
        # The head is a single linear layer: embedding (feature_dim) -> one logit
        # per category. It starts randomly initialised — see module docstring.
        self.head = nn.Linear(self.feature_dim, len(self.categories))
        self.head_is_trained = False

        # Inference-only object: eval() disables dropout/batch-norm updates, and
        # we move both modules onto the chosen device.
        self.backbone.to(self.device).eval()
        self.head.to(self.device).eval()

        if weights_path is not None:
            self.load(weights_path)

    # -- internal helpers ---------------------------------------------------

    def _to_pil(self, image: ImageInput):
        """Coerce any accepted input into a PIL RGB image for preprocessing.

        torchvision's preprocessing transforms expect a PIL image (or tensor), so
        this is the single funnel that normalises the three input flavours:
          * ``str`` / ``Path`` -> read the file from disk,
          * ``np.ndarray``     -> treat as an H×W×3 **RGB**, uint8 array,
          * PIL image          -> use as-is.
        Everything is forced to 3-channel RGB so grayscale/RGBA frames don't break
        the fixed-shape batch tensor downstream.
        """
        from PIL import Image  # imported lazily; torchvision pulls Pillow in anyway

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"unsupported image type: {type(image)!r}")

    def _preprocess_batch(self, images: Sequence[ImageInput]) -> torch.Tensor:
        """Turn a list of raw frames into one normalized (N, C, H, W) device tensor.

        Each frame is converted to PIL, run through the backbone's own
        resize/crop/normalize transform, then stacked into a single batch tensor
        so the GPU processes all N frames in one forward pass (far faster than
        looping frame-by-frame).
        """
        tensors = [self.preprocess(self._to_pil(im)) for im in images]
        return torch.stack(tensors).to(self.device)

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self, images: Sequence[ImageInput], batch_size: int = 32
    ) -> np.ndarray:
        """Embed frames into the backbone's feature space. Works without training.

        Runs the (pretrained) backbone only — no head — so the output is the raw
        ImageNet-learned representation of each frame. These embeddings are the
        genuinely-useful product before any violence head is trained: cache them
        as training inputs, feed them to late fusion, or cluster them.

        Args:
            images:     frames as paths / arrays / PIL images.
            batch_size: how many frames to push through the GPU at once.

        Returns:
            float32 array of shape ``(len(images), feature_dim)``.
        """
        if len(images) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        chunks: list[np.ndarray] = []
        # Chunk the input so a long video can't blow up GPU memory: we only ever
        # hold `batch_size` frames on the device at a time.
        for start in range(0, len(images), batch_size):
            batch = self._preprocess_batch(images[start : start + batch_size])
            embeddings = self.backbone(batch)          # (b, feature_dim)
            chunks.append(embeddings.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def predict(
        self, images: Sequence[ImageInput], batch_size: int = 32
    ) -> list[FramePrediction]:
        """Score each frame's violation probability via backbone + head.

        For every frame: embed it, push the embedding through the head to get one
        logit per category, then softmax those logits into a probability
        distribution. The violation probability is ``1 - P(safe)``.

        NOTE: if the head has not been trained (no ``weights_path`` was loaded)
        the numbers are meaningless — we emit a one-time warning so this is never
        silently mistaken for a real model.

        Returns:
            one :class:`FramePrediction` per input frame, in input order.
        """
        if not self.head_is_trained:
            warnings.warn(
                "VisualExpert head is untrained — violation probabilities are "
                "random. Train and load a head before trusting predict().",
                stacklevel=2,
            )
        if len(images) == 0:
            return []

        results: list[FramePrediction] = []
        for start in range(0, len(images), batch_size):
            batch = self._preprocess_batch(images[start : start + batch_size])
            logits = self.head(self.backbone(batch))            # (b, num_categories)
            # softmax turns raw logits into a proper probability distribution.
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for offset, row in enumerate(probs):
                scores = {c: float(p) for c, p in zip(self.categories, row)}
                # Headline score: probability the frame is NOT safe.
                violation_prob = 1.0 - scores[self.safe_category]
                # Most-likely class overall (honest: this can be "safe").
                top_category = max(scores, key=scores.__getitem__)
                results.append(
                    FramePrediction(
                        index=start + offset,
                        violation_prob=violation_prob,
                        category=top_category,
                        scores=scores,
                    )
                )
        return results

    def save(self, path: str | Path) -> None:
        """Persist the trained head (and current backbone) to a checkpoint.

        We store the backbone name and categories alongside the weights so
        :meth:`load` can sanity-check that a checkpoint is being loaded into a
        matching architecture rather than silently mis-aligning tensors.
        """
        torch.save(
            {
                "backbone_name": self.backbone_name,
                "categories": list(self.categories),
                "head": self.head.state_dict(),
                "backbone": self.backbone.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load weights saved by :meth:`save`, validating the architecture first.

        Restores the head (and any fine-tuned backbone weights) and flips
        ``head_is_trained`` to True so :meth:`predict` stops warning. Raises if the
        checkpoint was produced for a different backbone or category set, which
        would otherwise produce subtly wrong outputs.
        """
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("backbone_name") != self.backbone_name:
            raise ValueError(
                f"checkpoint backbone {ckpt.get('backbone_name')!r} != "
                f"expert backbone {self.backbone_name!r}"
            )
        if tuple(ckpt.get("categories", ())) != self.categories:
            raise ValueError("checkpoint categories do not match this expert")
        self.backbone.load_state_dict(ckpt["backbone"])
        self.head.load_state_dict(ckpt["head"])
        self.backbone.to(self.device).eval()
        self.head.to(self.device).eval()
        self.head_is_trained = True
