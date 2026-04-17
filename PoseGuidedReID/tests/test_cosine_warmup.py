"""Tests for CosineWarmupLR scheduler."""
import torch
import pytest
from project.solver.cosine_warmup import CosineWarmupLR


def make_optimizer(lr=0.01):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=lr)


def _get_lr_at_epoch(sched):
    """Return the current LR from the optimizer (avoids get_last_lr issues)."""
    return sched.optimizer.param_groups[0]["lr"]


def test_warmup_starts_low():
    """At epoch 0 with warmup=5, lr should equal base_lr * 1/5."""
    opt = make_optimizer(lr=0.01)
    # __init__ with last_epoch=-1 calls step() internally -> epoch 0
    sched = CosineWarmupLR(opt, warmup_epochs=5, max_epochs=20)
    lr = _get_lr_at_epoch(sched)
    assert abs(lr - 0.01 * 1 / 5) < 1e-7


def test_warmup_reaches_base_at_warmup_end():
    """After warmup, at epoch=warmup_epochs-1, lr should equal base_lr."""
    opt = make_optimizer(lr=0.01)
    sched = CosineWarmupLR(opt, warmup_epochs=5, max_epochs=20)
    # step 4 more times to reach epoch 4 (last warmup epoch: lr = base*5/5)
    for _ in range(4):
        sched.step()
    lr = _get_lr_at_epoch(sched)
    assert abs(lr - 0.01) < 1e-7


def test_cosine_decay_reaches_min_at_end():
    """At the final epoch, lr should be close to min_lr."""
    opt = make_optimizer(lr=0.01)
    sched = CosineWarmupLR(opt, warmup_epochs=5, max_epochs=20, min_lr=0.0)
    # step from epoch 0 to epoch 19
    for _ in range(19):
        sched.step()
    lr = _get_lr_at_epoch(sched)
    assert lr < 0.01


def test_lr_curve_monotone_after_warmup():
    """Post-warmup LRs should be non-increasing."""
    opt = make_optimizer(lr=0.01)
    sched = CosineWarmupLR(opt, warmup_epochs=5, max_epochs=50)
    lrs = [_get_lr_at_epoch(sched)]  # epoch 0
    for _ in range(49):
        sched.step()
        lrs.append(_get_lr_at_epoch(sched))
    post_warmup = lrs[5:]  # epochs 5..49 (cosine region)
    for i in range(1, len(post_warmup)):
        assert post_warmup[i] <= post_warmup[i - 1] + 1e-9
