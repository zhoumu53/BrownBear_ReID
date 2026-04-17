"""Augmentation profiles for training and evaluation transforms."""

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(profile: str, resolution: int = 224) -> transforms.Compose:
    """Return a training transform pipeline for the given augmentation profile.

    Profiles:
        weak          — resize + flip + pad/crop + erasing
        strong        — random-resized-crop + flip + color-jitter + erasing
        strong_color  — same as strong with additional random rotation
    """
    if profile == "weak":
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.5),
        ])
    elif profile == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.5),
        ])
    elif profile == "strong_color":
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.5),
        ])
    else:
        raise ValueError(f"Unknown augmentation profile: {profile!r}")


def build_eval_transform(resolution: int = 224) -> transforms.Compose:
    """Return a deterministic evaluation transform pipeline."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
