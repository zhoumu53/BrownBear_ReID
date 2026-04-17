"""Lightweight validation: compute Rank-1 accuracy for checkpoint selection."""

import torch


@torch.no_grad()
def compute_val_rank1(model, val_loader, device) -> float:
    """Compute Rank-1 nearest-neighbour accuracy on a validation set.

    Args:
        model: Model whose forward returns (logits, features).
        val_loader: Iterable of batches (dict or tuple).
        device: torch device for inference.

    Returns:
        Rank-1 accuracy as a float in [0, 1].
    """
    was_training = model.training
    model.eval()

    all_feats = []
    all_pids = []

    for batch in val_loader:
        # Support both dict and tuple batches
        if isinstance(batch, dict):
            imgs = batch["image"]
            pids = batch["pid"]
        else:
            imgs, pids = batch[0], batch[1]

        imgs = imgs.to(device)
        _, features = model(imgs)

        # L2-normalize
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        all_feats.append(features.cpu())
        if isinstance(pids, torch.Tensor):
            all_pids.append(pids.cpu())
        else:
            all_pids.append(torch.tensor(pids))

    all_feats = torch.cat(all_feats, dim=0)
    all_pids = torch.cat(all_pids, dim=0)

    # Cosine similarity (features already L2-normed)
    sim = all_feats @ all_feats.T

    # Mask diagonal so a sample never matches itself
    sim.fill_diagonal_(float("-inf"))

    # Nearest neighbour
    nn_indices = sim.argmax(dim=1)
    nn_pids = all_pids[nn_indices]

    rank1 = (nn_pids == all_pids).float().mean().item()

    # Restore previous training state
    if was_training:
        model.train()

    return rank1
