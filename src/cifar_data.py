"""CIFAR-100 training/dev splits with an explicit, separate final-test loader.

The split seed is independent of model, sampler, and augmentation seeds. This
module never mixes the official test images into training or development data.
Mixup/CutMix and the ordering of samples belong to the training runner.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from PIL import ImageFilter
from timm.data import create_transform
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100


CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
CIFAR100_ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


@dataclass(frozen=True)
class CIFARDataConfig:
    root: str | Path
    image_size: int = 32
    split_seed: int = 20261015
    dev_per_class: int = 50
    augmentation: str = "deit"
    color_jitter: float = 0.3
    random_erasing: float = 0.25
    download: bool = False

    def __post_init__(self):
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if isinstance(self.dev_per_class, bool) or not isinstance(self.dev_per_class, int):
            raise ValueError("dev_per_class must be an integer")
        if not 0 < self.dev_per_class < 500:
            raise ValueError("dev_per_class must leave training and development images in every class")
        if self.augmentation not in {"basic", "deit", "autoaugment", "threeaugment"}:
            raise ValueError("Unknown augmentation policy")
        if not math.isfinite(self.color_jitter) or self.color_jitter < 0:
            raise ValueError("color_jitter must be finite and nonnegative")
        if not math.isfinite(self.random_erasing) or not 0 <= self.random_erasing <= 1:
            raise ValueError("random_erasing must lie in [0, 1]")


@dataclass
class CIFARDataBundle:
    train: Dataset
    dev: Dataset
    train_indices: list[int]
    dev_indices: list[int]
    classes: list[str]
    metadata: dict


def split_train_indices(
    targets: Sequence[int], dev_per_class: int = 50, seed: int = 20261015
) -> tuple[list[int], list[int]]:
    """Split each class once using only labels from the official training set."""
    labels = np.asarray(targets)
    if labels.ndim != 1 or not labels.size or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("targets must be a nonempty one-dimensional integer sequence")
    if (labels < 0).any():
        raise ValueError("targets must be nonnegative")
    if isinstance(dev_per_class, bool) or not isinstance(dev_per_class, int) or dev_per_class <= 0:
        raise ValueError("dev_per_class must be a positive integer")
    rng = np.random.default_rng(seed)
    train_indices, dev_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= dev_per_class:
            raise ValueError(f"Class {label} has no training images after the requested split")
        rng.shuffle(indices)
        dev_indices.extend(indices[:dev_per_class].tolist())
        train_indices.extend(indices[dev_per_class:].tolist())
    return sorted(train_indices), sorted(dev_indices)


class _GaussianBlur:
    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def _crop_transform(image_size):
    if image_size <= 32:
        return transforms.RandomCrop(image_size, padding=4)
    return transforms.RandomResizedCrop(
        image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
    )


def _eval_transform(config):
    operations = []
    if config.image_size != 32:
        operations.append(transforms.Resize(
            (config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BICUBIC
        ))
    return transforms.Compose([
        *operations,
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_transforms(config: CIFARDataConfig):
    """Build train and deterministic eval transforms, without loading a dataset.

    ``deit`` uses timm RandAugment (rand-m9-mstd0.5-inc1) and pixel erasing.
    ``autoaugment`` uses the CIFAR10Policy used by the small-data SPT/LSA
    implementation on CIFAR-100, followed by zero-valued random erasing.
    ``threeaugment`` selects grayscale, solarization, or PIL Gaussian blur.
    Native 32-pixel inputs use a padded random crop; larger inputs use RRC.
    These are configurable recipe adaptations, not an ImageNet reproduction.
    """
    if config.augmentation == "deit":
        train_transform = create_transform(
            input_size=config.image_size,
            is_training=True,
            color_jitter=config.color_jitter,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=config.random_erasing,
            re_mode="pixel",
            re_count=1,
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
        )
        if config.image_size <= 32:
            train_transform.transforms[0] = _crop_transform(config.image_size)
    else:
        operations = [_crop_transform(config.image_size), transforms.RandomHorizontalFlip()]
        if config.augmentation == "autoaugment":
            operations.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif config.augmentation == "threeaugment":
            operations.append(transforms.RandomChoice([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomSolarize(threshold=128, p=1.0),
                _GaussianBlur(),
            ]))
            if config.color_jitter:
                operations.append(transforms.ColorJitter(
                    config.color_jitter, config.color_jitter, config.color_jitter
                ))
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        if config.augmentation == "autoaugment" and config.random_erasing:
            operations.append(transforms.RandomErasing(p=config.random_erasing))
        train_transform = transforms.Compose(operations)
    return train_transform, _eval_transform(config)


class _TransformedSubset(Dataset):
    """Independent transforms over a shared, unaugmented official train set."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.targets = [int(dataset.targets[index]) for index in self.indices]
        self.classes = list(dataset.classes)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, target = self.dataset[self.indices[index]]
        return self.transform(image), int(target)


def _check_official_split(dataset, train):
    expected_per_class = 500 if train else 100
    labels = np.asarray(dataset.targets)
    if len(dataset) != 100 * expected_per_class or labels.shape != (100 * expected_per_class,):
        raise ValueError("Unexpected official CIFAR-100 split size")
    if len(dataset.classes) != 100 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Unexpected official CIFAR-100 classes")
    if (labels < 0).any() or (labels >= 100).any():
        raise ValueError("Official CIFAR-100 labels must be in [0, 99]")
    if not np.all(np.bincount(labels, minlength=100) == expected_per_class):
        raise ValueError("Unexpected official CIFAR-100 per-class counts")


def build_datasets(config: CIFARDataConfig) -> CIFARDataBundle:
    """Load official train only; construct the fixed stratified train/dev split.

    torchvision verifies the official file checksums. ``download=True`` obtains
    its standard archive, which contains both splits, but this builder reads
    labels/images only from the training split. Official test is never created.
    """
    dataset = CIFAR100(root=config.root, train=True, transform=None, download=config.download)
    _check_official_split(dataset, train=True)
    train_indices, dev_indices = split_train_indices(
        dataset.targets, dev_per_class=config.dev_per_class, seed=config.split_seed
    )
    train_transform, eval_transform = build_transforms(config)
    metadata = {
        "dataset": "CIFAR-100",
        "root": str(config.root),
        "official_archive_md5": CIFAR100_ARCHIVE_MD5,
        "split_seed": config.split_seed,
        "train_images": len(train_indices),
        "dev_images": len(dev_indices),
        "dev_per_class": config.dev_per_class,
        "official_test_used": False,
        "image_size": config.image_size,
        "augmentation": config.augmentation,
        "train_transform": repr(train_transform),
        "eval_transform": repr(eval_transform),
        "normalization_mean": list(CIFAR100_MEAN),
        "normalization_std": list(CIFAR100_STD),
        "train_indices_sha256": hashlib.sha256(
            json.dumps(train_indices, separators=(",", ":")).encode()
        ).hexdigest(),
        "dev_indices_sha256": hashlib.sha256(
            json.dumps(dev_indices, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return CIFARDataBundle(
        train=_TransformedSubset(dataset, train_indices, train_transform),
        dev=_TransformedSubset(dataset, dev_indices, eval_transform),
        train_indices=train_indices,
        dev_indices=dev_indices,
        classes=list(dataset.classes),
        metadata=metadata,
    )


def build_test_dataset(config: CIFARDataConfig) -> Dataset:
    """Explicit final evaluation entry point; call after all choices are frozen."""
    dataset = CIFAR100(
        root=config.root, train=False, transform=_eval_transform(config), download=config.download
    )
    _check_official_split(dataset, train=False)
    return dataset
