"""Device auto-detection: prefer MPS, then CUDA, then CPU."""
from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference string into a torch.device.

    preference: one of "auto", "mps", "cuda", "cpu".
    "auto" picks the first available of: mps, cuda, cpu.
    """
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference in ("mps", "cuda", "cpu"):
        return torch.device(preference)
    raise ValueError(f"Unknown device preference: {preference!r}")
