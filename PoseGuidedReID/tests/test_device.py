"""Unit tests for device resolution."""
import pytest
import torch
from project.utils.device import get_device


def test_get_device_cpu_explicit():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_unknown_raises():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_get_device_auto_returns_torch_device():
    d = get_device("auto")
    assert isinstance(d, torch.device)
    assert d.type in ("mps", "cuda", "cpu")


def test_get_device_mps_explicit_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host")
    assert get_device("mps") == torch.device("mps")
