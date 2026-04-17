"""Cosine annealing LR scheduler with linear warmup."""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLR(_LRScheduler):
    """Cosine LR decay with a linear warmup phase.

    During warmup (epoch < warmup_epochs):
        lr = base_lr * (epoch + 1) / warmup_epochs

    After warmup (epoch >= warmup_epochs):
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if warmup_epochs >= max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be < max_epochs ({max_epochs})"
            )
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (
                    self.max_epochs - self.warmup_epochs
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            lrs.append(lr)
        return lrs
