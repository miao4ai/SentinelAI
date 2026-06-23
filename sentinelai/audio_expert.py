"""Audio expert (Audio): mel-spectrogram features + abnormal-sound detection.

Pipeline role (ROADMAP 3.2)
---------------------------
This is the *audio* expert of the V1 expert-model stack. Given the audio track of
a video it produces, per short time window:

1. an audio **embedding** (a fixed-length feature vector), and
2. **violation probabilities** for abnormal sounds — gunshots, explosions, screams.

Why AST, and why it works without training
------------------------------------------
We use **AST (Audio Spectrogram Transformer)** pretrained on Google's **AudioSet**.
AudioSet's 527 sound classes already include "Gunshot, gunfire", "Explosion" and
"Screaming", which are exactly the abnormal tracks we care about. So AST detects
them **zero-shot** — no extra training head needed.

Contrast with the visual expert: ImageNet has no "violence" class, so the visual
expert had to bolt on a head and train it. AudioSet *does* contain these sound
classes, so the audio expert is useful out of the box. We exploit this by mapping
the relevant AudioSet labels onto our violation categories.

(VGGish — the other option in the brief — only emits 128-d embeddings and would
likewise need a trained head, so AST is the stronger choice for *this* goal.)

How a sound becomes a number
-----------------------------
    waveform --(log-mel spectrogram)--> AST transformer --> 527 AudioSet logits
            --(sigmoid, multi-label)--> per-sound probabilities --> our categories

The log-mel spectrogram is the "梅尔频谱图特征" from the brief; it is produced by
the AST feature extractor (see :meth:`AudioExpert.mel_features`).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .hardware import get_device
from .video import ffmpeg_path

# A pretrained AST fine-tuned on AudioSet (527 classes incl. gunshot/explosion/scream).
DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AudioSet/AST operate on 16 kHz mono audio.
TARGET_SR = 16000

# AST was fine-tuned on ~10 s clips, so we score the track in 10 s windows.
WINDOW_SECONDS = 10.0

# Accepted audio input: a media file path, or an already-decoded mono waveform.
AudioInput = Union[str, Path, np.ndarray]

# Map each violation category to the AudioSet label substrings that signal it.
# Matching is done case-insensitively on the official AudioSet display names, so a
# substring like "gun" catches "Gunshot, gunfire", "Machine gun", etc.
VIOLATION_SOUND_MAP: dict[str, tuple[str, ...]] = {
    "gunshot": ("gunshot", "machine gun", "fusillade", "artillery", "cannon", "cap gun"),
    "explosion": ("explosion", "boom", "eruption", "burst"),
    "scream": ("screaming", "shout", "yell", "battle cry", "children shouting"),
}


def _aggregate_violation(
    class_probs: dict[str, float],
    mapping: dict[str, tuple[str, ...]] = VIOLATION_SOUND_MAP,
) -> dict[str, float]:
    """Collapse 527 AudioSet probabilities into our violation categories.

    For each category we take the **maximum** probability among the AudioSet
    labels mapped to it (max, not sum: AudioSet is multi-label, and "any one of
    these dangerous sounds is clearly present" is what we want — summing would
    double-count synonyms like "Gunshot" and "Machine gun").

    Pure function (no model, no torch) so it is cheap to unit-test.

    Args:
        class_probs: AudioSet label -> probability (already sigmoid'd).
        mapping:     category -> tuple of label substrings to match.

    Returns:
        category -> aggregated probability in [0, 1].
    """
    scores: dict[str, float] = {}
    for category, needles in mapping.items():
        best = 0.0
        for label, prob in class_probs.items():
            low = label.lower()
            if any(needle in low for needle in needles):
                best = max(best, prob)
        scores[category] = best
    return scores


def _window(
    waveform: np.ndarray, sr: int, window_s: float = WINDOW_SECONDS
) -> list[tuple[float, np.ndarray]]:
    """Split a long waveform into consecutive fixed-length windows.

    Returns a list of ``(start_seconds, clip)`` pairs. We score each window
    separately so the expert can say *when* an abnormal sound occurs, not just
    whether the whole track contains one. Audio shorter than one window is
    returned as a single (possibly short) clip — the feature extractor pads it.

    The final partial tail (if any) is emitted as one more window so the end of
    the track is never silently dropped.
    """
    win = int(window_s * sr)
    if len(waveform) <= win:
        return [(0.0, waveform)]

    windows: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(waveform), win):
        clip = waveform[start : start + win]
        if len(clip) == 0:
            break
        windows.append((start / sr, clip))
    return windows


@dataclass(frozen=True)
class AudioEvent:
    """The verdict for one audio window.

    Attributes:
        start:          window start time in seconds.
        end:            window end time in seconds.
        violation_prob: max over categories — overall "abnormal sound" likelihood.
        category:       which abnormal sound is most likely ("gunshot"/"explosion"/"scream").
        scores:         per-category probabilities.
        top_sounds:     the few highest-probability raw AudioSet labels (for debugging
                        and explainability — e.g. shows "Speech 0.9" on normal talk).
    """

    start: float
    end: float
    violation_prob: float
    category: str
    scores: dict[str, float]
    top_sounds: list[tuple[str, float]]


class AudioExpert:
    """Abnormal-sound detector built on a pretrained AudioSet AST model.

    Typical use::

        expert = AudioExpert()
        events = expert.predict("clip.mp4")     # works out of the box (zero-shot)
        for e in events:
            if e.violation_prob > 0.5:
                print(e.start, e.category, e.violation_prob)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        """Load the AST feature extractor + model and move them onto the device.

        We import ``transformers`` lazily (inside __init__, not at module import)
        so that the rest of the package — and the unit tests for the pure helpers
        above — don't pay the heavy import or require the dependency.
        """
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device = device or get_device()
        self.window_seconds = window_seconds
        # The feature extractor turns a raw waveform into the log-mel spectrogram
        # the model expects; the model maps that spectrogram to 527 AudioSet logits.
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        # id2label: index -> human-readable AudioSet name (e.g. 0 -> "Speech").
        self.id2label: dict[int, str] = self.model.config.id2label

    # -- audio loading ------------------------------------------------------

    def _load_waveform(self, audio: AudioInput, sr: int | None = None) -> np.ndarray:
        """Return a mono float32 waveform sampled at :data:`TARGET_SR`.

        Two input paths:
          * **file path** (audio *or* video): we shell out to ffmpeg, which
            decodes any container, downmixes to mono, resamples to 16 kHz and
            streams raw 16-bit PCM to stdout — read straight into numpy, no temp
            file and no extra audio library. This reuses the same ffmpeg the video
            module relies on.
          * **numpy array**: treated as an already-decoded mono waveform; if its
            sample rate ``sr`` differs from 16 kHz we resample with torchaudio.
        """
        if isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32)
            if sr is not None and sr != TARGET_SR:
                import torch
                import torchaudio.functional as AF

                wav = AF.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()
            return wav

        ffmpeg = ffmpeg_path()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found; needed to decode audio from files")
        # -f s16le: signed 16-bit little-endian PCM; -ac 1: mono; -ar: resample.
        cmd = [ffmpeg, "-v", "error", "-i", str(audio),
               "-f", "s16le", "-ac", "1", "-ar", str(TARGET_SR), "-"]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        # int16 PCM -> float32 in [-1, 1] (divide by 2**15).
        return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    # -- core forward pass --------------------------------------------------

    def mel_features(self, clip: np.ndarray):
        """Convert one waveform clip into the model's log-mel spectrogram tensor.

        This is the explicit "梅尔频谱图" step: the AST feature extractor frames the
        audio, applies a mel filterbank and takes the log, yielding a
        (time × mel-bins) image that the transformer treats like a picture.
        Returned as a batch tensor on the device, ready for ``self.model``.
        """
        inputs = self.feature_extractor(
            clip, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        return inputs["input_values"].to(self.device)

    def _forward(self, clips: Sequence[np.ndarray]):
        """Run a batch of clips through AST, returning (probs, embeddings).

        * ``probs``      — sigmoid of the 527 logits. We use **sigmoid, not
          softmax**, because AudioSet is multi-label: a clip can be "Speech" AND
          "Gunshot" at once, so each class gets its own independent probability.
        * ``embeddings`` — the mean of the last hidden state over tokens, a 768-d
          per-clip vector reusable for fusion / training, mirroring the visual
          expert's ``extract_features``.
        """
        import torch

        # Batch all clips' spectrograms into one tensor for a single forward pass.
        spectrograms = torch.cat([self.mel_features(c) for c in clips], dim=0)
        with torch.no_grad():
            out = self.model(spectrograms, output_hidden_states=True)
        probs = torch.sigmoid(out.logits).cpu().numpy()          # (B, 527)
        embeddings = out.hidden_states[-1].mean(dim=1).cpu().numpy()  # (B, 768)
        return probs, embeddings

    # -- public API ---------------------------------------------------------

    def extract_features(self, audio: AudioInput, sr: int | None = None) -> np.ndarray:
        """Embed an audio track window-by-window. Returns ``(num_windows, 768)``.

        Like the visual expert's feature extractor: the raw AST representation of
        each window, useful as cached inputs for later fusion/training regardless
        of the violation mapping.
        """
        waveform = self._load_waveform(audio, sr=sr)
        clips = [clip for _, clip in _window(waveform, TARGET_SR, self.window_seconds)]
        _, embeddings = self._forward(clips)
        return embeddings

    def predict(
        self, audio: AudioInput, sr: int | None = None, top_k: int = 5
    ) -> list[AudioEvent]:
        """Detect abnormal sounds across the track, one verdict per time window.

        For each window: get AudioSet probabilities, collapse them onto our
        categories via :func:`_aggregate_violation`, and record the top raw labels
        for explainability. Unlike the visual expert, this needs **no training** —
        the AudioSet classes carry the meaning.

        Args:
            audio: media file path, or a decoded mono waveform array.
            sr:    sample rate of ``audio`` when it is an array (ignored for files).
            top_k: how many raw AudioSet labels to keep per window for debugging.

        Returns:
            one :class:`AudioEvent` per window, in time order.
        """
        waveform = self._load_waveform(audio, sr=sr)
        windows = _window(waveform, TARGET_SR, self.window_seconds)
        probs, _ = self._forward([clip for _, clip in windows])

        events: list[AudioEvent] = []
        for (start, clip), row in zip(windows, probs):
            # Full AudioSet distribution for this window.
            class_probs = {self.id2label[i]: float(p) for i, p in enumerate(row)}
            scores = _aggregate_violation(class_probs)
            category = max(scores, key=scores.__getitem__)
            violation_prob = scores[category]
            # Highest-probability raw sounds, for human-readable explanations.
            top_sounds = sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            events.append(
                AudioEvent(
                    start=start,
                    end=start + len(clip) / TARGET_SR,
                    violation_prob=violation_prob,
                    category=category,
                    scores=scores,
                    top_sounds=top_sounds,
                )
            )
        return events
