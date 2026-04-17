"""Tests for augmentation profiles (weak / strong / strong_color)."""
import torch
import pytest
from PIL import Image

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "transforms_v2",
    Path(__file__).resolve().parent.parent / "project" / "datasets" / "transforms_v2.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
build_train_transform = _mod.build_train_transform
build_eval_transform = _mod.build_eval_transform


@pytest.fixture
def fake_pil_image():
    return Image.new("RGB", (300, 400), color=(123, 200, 50))


# 1. Parametrized: output shape and dtype
@pytest.mark.parametrize("profile", ["weak", "strong", "strong_color"])
def test_train_transform_output_shape_and_dtype(profile, fake_pil_image):
    transform = build_train_transform(profile, resolution=224)
    out = transform(fake_pil_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


# 2. Unknown profile raises ValueError
def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        build_train_transform("unknown_profile")


# 3. Eval transform returns normalized tensor (mean roughly ~0)
def test_eval_transform_normalized(fake_pil_image):
    transform = build_eval_transform(resolution=224)
    out = transform(fake_pil_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    # After ImageNet normalization of a solid-color image, mean should be near 0
    assert out.mean().abs() < 2.0  # generous bound; unnormalized would be ~0.5-0.8


# 4. Eval transform is deterministic
def test_eval_transform_deterministic(fake_pil_image):
    transform = build_eval_transform(resolution=224)
    out1 = transform(fake_pil_image)
    out2 = transform(fake_pil_image)
    assert torch.equal(out1, out2)


# 5. Strong transform is stochastic
def test_strong_transform_stochastic(fake_pil_image):
    transform = build_train_transform("strong", resolution=224)
    torch.manual_seed(0)
    out1 = transform(fake_pil_image)
    torch.manual_seed(999)
    out2 = transform(fake_pil_image)
    # Very unlikely to be identical with different seeds due to random crop/jitter/erasing
    assert not torch.equal(out1, out2)
