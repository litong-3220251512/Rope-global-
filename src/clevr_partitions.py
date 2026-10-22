"""Deterministic image-level partitions for official CLEVR train/val sets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from numpy.random import Generator, PCG64


_TRAIN_SEED = 20261016
_VAL_SEED = 20261017
_OFFICIAL_SPLITS = frozenset({"train", "val"})


@dataclass(frozen=True, slots=True)
class ImageIdentity:
    official_split: str
    image_index: int
    image_filename: str

    def __post_init__(self) -> None:
        if self.official_split not in _OFFICIAL_SPLITS:
            raise ValueError("official_split must be exactly 'train' or 'val'")
        if type(self.image_index) is not int or self.image_index < 0:
            raise ValueError("image_index must be a non-negative integer")
        expected_filename = (
            f"CLEVR_{self.official_split}_{self.image_index:06d}.png"
        )
        if self.image_filename != expected_filename:
            raise ValueError(
                "image_filename must match official_split and image_index"
            )


@dataclass(frozen=True, slots=True)
class ImagePartitions:
    train: tuple[ImageIdentity, ...]
    dev: tuple[ImageIdentity, ...]
    base_test: tuple[ImageIdentity, ...]
    confirmation: tuple[ImageIdentity, ...]

    def __post_init__(self) -> None:
        for name in ("train", "dev", "base_test", "confirmation"):
            partition = getattr(self, name)
            if not isinstance(partition, tuple) or any(
                not isinstance(image, ImageIdentity) for image in partition
            ):
                raise ValueError(f"{name} must be a tuple of ImageIdentity records")


def _validated_sorted_images(
    images: Sequence[ImageIdentity], expected_split: str
) -> tuple[ImageIdentity, ...]:
    if not isinstance(images, Sequence):
        raise ValueError("image mother sets must be sequences")

    seen_indices: set[int] = set()
    seen_filenames: set[str] = set()
    validated: list[ImageIdentity] = []
    for image in images:
        if not isinstance(image, ImageIdentity):
            raise ValueError("image mother sets must contain ImageIdentity records")
        if image.official_split != expected_split:
            raise ValueError(
                f"{expected_split} mother set contains {image.official_split} image"
            )
        if image.image_index in seen_indices:
            raise ValueError(
                f"duplicate {expected_split} image_index: {image.image_index}"
            )
        if image.image_filename in seen_filenames:
            raise ValueError(
                f"duplicate {expected_split} image_filename: {image.image_filename}"
            )
        seen_indices.add(image.image_index)
        seen_filenames.add(image.image_filename)
        validated.append(image)

    return tuple(
        sorted(validated, key=lambda image: (image.image_index, image.image_filename))
    )


def _permuted(
    images: tuple[ImageIdentity, ...], seed: int
) -> tuple[ImageIdentity, ...]:
    order = Generator(PCG64(seed)).permutation(len(images))
    return tuple(images[int(index)] for index in order)


def build_image_partitions(
    train_images: Sequence[ImageIdentity],
    val_images: Sequence[ImageIdentity],
) -> ImagePartitions:
    """Sort, permute once per official split, then apply frozen floor cuts."""
    train_permuted = _permuted(
        _validated_sorted_images(train_images, "train"), _TRAIN_SEED
    )
    val_permuted = _permuted(
        _validated_sorted_images(val_images, "val"), _VAL_SEED
    )

    train_boundary = 9 * len(train_permuted) // 10
    val_boundary = len(val_permuted) // 2
    return ImagePartitions(
        train=train_permuted[:train_boundary],
        dev=train_permuted[train_boundary:],
        base_test=val_permuted[:val_boundary],
        confirmation=val_permuted[val_boundary:],
    )
