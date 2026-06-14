"""Compute-device detection.

One source of truth for "what should this run on". Works whether or not a GPU
is present, so the same code path runs on the CPU dev VM and a future GPU VM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device: str  # "cuda" or "cpu"
    name: str  # human-readable device name
    cuda_available: bool
    torch_version: str
    torch_cuda_version: str | None  # CUDA toolkit the torch wheel was built for


def get_device_info() -> DeviceInfo:
    """Inspect the installed torch build and the host GPU."""
    import torch

    cuda = torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
        device = "cuda"
    else:
        name = "cpu"
        device = "cpu"
    return DeviceInfo(
        device=device,
        name=name,
        cuda_available=cuda,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,  # None for CPU-only wheels
    )


def get_device() -> str:
    """Return the torch device string to place tensors/models on."""
    return get_device_info().device
