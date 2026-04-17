#!/usr/bin/env python3
"""Config-driven training CLI for Pose-Guided ReID.

Usage
-----
python tools/train.py \
    --config configs/experiments/01_fixes_only.yaml \
    --output runs/01_fixes_only/ \
    [--epochs N] [--device auto|mps|cuda|cpu] [--num-workers N]
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# sys.path: allow ``from project.…`` imports when run from tools/
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # adds PoseGuidedReID/ to path

from project.config.loader import load_config
from project.config.defaults import _C
from project.datasets.make_dataloader import make_csv_dataloaders
from project.models import make_model
from project.losses.make_loss import make_loss
from project.solver.make_optimizer import make_optimizer
from project.solver.scheduler_factory import create_scheduler
from project.processor.processor import do_train_v2
from project.utils.device import get_device
from project.utils.run_output import init_run_output

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_yaml_to_cfg(yaml_dict: dict, cfg_node) -> None:
    """Recursively walk *yaml_dict* and set matching fields on *cfg_node*.

    Keys in *yaml_dict* must match the yacs CfgNode hierarchy exactly.
    Type enforcement is handled by yacs itself on assignment.
    """
    for key, value in yaml_dict.items():
        if not hasattr(cfg_node, key):
            raise KeyError(
                f"YAML key {key!r} not found in config node "
                f"(available: {list(cfg_node.keys())})"
            )
        sub = getattr(cfg_node, key)
        if isinstance(value, dict):
            # Recurse into the sub-node
            _apply_yaml_to_cfg(value, sub)
        else:
            setattr(cfg_node, key, value)


def _build_loss_fn(cfg, num_classes):
    """Build a loss callable with signature ``loss_fn(logits, pids, feat) -> Tensor``.

    ``make_loss()`` returns ``(criterion_ce, criterion_triplet)``.
    ``do_train_v2`` calls ``loss_fn(logits, pids, feat)``.

    We combine CE + triplet with configurable weights (matching the legacy
    ``make_loss_old`` behaviour).
    """
    criterion_ce, criterion_triplet = make_loss()

    id_weight = cfg.MODEL.ID_LOSS_WEIGHT
    tri_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT

    def loss_fn(logits, pids, feat):
        # logits may be a list [logits, ...] from multi-branch heads
        if isinstance(logits, (list, tuple)):
            ce = criterion_ce(logits[0], pids)
        else:
            ce = criterion_ce(logits, pids)

        tri = criterion_triplet(feat, pids)

        return id_weight * ce + tri_weight * tri

    return loss_fn


def _setup_logging(log_path: Path | None = None) -> None:
    """Configure root + ``train`` logger to stderr (+ optional file)."""
    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_path is not None:
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pose-Guided ReID -- config-driven training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to YAML experiment config")
    p.add_argument("--output", required=True, help="Run output directory")
    p.add_argument("--epochs", type=int, default=None, help="Override SOLVER.MAX_EPOCHS")
    p.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device preference",
    )
    p.add_argument("--num-workers", type=int, default=None, help="Override DATALOADER.NUM_WORKERS")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. Load YAML config ------------------------------------------------
    yaml_dict = load_config(args.config)

    # 2. Merge YAML into yacs defaults -----------------------------------
    cfg = _C.clone()
    cfg.defrost()
    _apply_yaml_to_cfg(yaml_dict, cfg)

    # 3. Apply CLI overrides ---------------------------------------------
    if args.epochs is not None:
        cfg.SOLVER.MAX_EPOCHS = args.epochs
    if args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = args.output

    # Device is resolved separately (not stored in yacs cfg)
    device = get_device(args.device)

    # 4. Freeze config ---------------------------------------------------
    cfg.freeze()

    # 5. Init run output -------------------------------------------------
    frozen_dict = yaml_dict  # already a plain dict suitable for YAML dump
    output_paths = init_run_output(args.output, frozen_dict)

    # 6. Logging ---------------------------------------------------------
    _setup_logging(output_paths["log"])
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", args.output)
    logger.info("Device: %s", device)
    logger.info("Frozen config:\n%s", cfg)

    # 7. Seed ------------------------------------------------------------
    _set_seed(cfg.SOLVER.SEED)

    # 8. Dataloaders -----------------------------------------------------
    train_loader, val_loader, num_classes = make_csv_dataloaders(cfg)
    logger.info("Dataloaders ready -- %d train batches, %d val batches, %d classes",
                len(train_loader), len(val_loader), num_classes)

    # 9. Model -----------------------------------------------------------
    #    make_model reads cfg.MODEL.* extensively. We need to temporarily
    #    set NUM_CLASSES if the config hierarchy requires it, but make_model
    #    takes num_classes as an explicit argument.
    model = make_model(
        cfg,
        num_classes=num_classes,
        logger=logger,
        load_weights=True,
        return_feature=True,
        device=str(device),
    )
    model = model.to(device)
    logger.info("Model built and moved to %s", device)

    # 10. Optimizer + scheduler ------------------------------------------
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    # 11. Loss -----------------------------------------------------------
    loss_fn = _build_loss_fn(cfg, num_classes)

    # 12. Train ----------------------------------------------------------
    logger.info("Starting training for %d epochs", cfg.SOLVER.MAX_EPOCHS)
    metrics = do_train_v2(
        cfg, model, train_loader, val_loader,
        optimizer, scheduler, loss_fn, device, output_paths,
    )

    # 13. Log final metrics ----------------------------------------------
    if metrics:
        logger.info("Training complete. Final metrics: %s", metrics)
    else:
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
