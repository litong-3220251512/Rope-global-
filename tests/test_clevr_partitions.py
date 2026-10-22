"""Data-free tests for deterministic CLEVR image-level partitions."""

import unittest
from dataclasses import FrozenInstanceError

from src.clevr_partitions import (
    ImageIdentity,
    ImagePartitions,
    build_image_partitions,
)


def _image(official_split: str, image_index: int) -> ImageIdentity:
    return ImageIdentity(
        official_split=official_split,
        image_index=image_index,
        image_filename=f"CLEVR_{official_split}_{image_index:06d}.png",
    )


class ImageIdentityTests(unittest.TestCase):
    def test_identity_is_immutable_and_requires_the_official_filename_contract(self):
        identity = _image("train", 7)

        with self.assertRaises(FrozenInstanceError):
            identity.image_index = 8

        invalid = (
            ("test", 7, "CLEVR_test_000007.png"),
            ("train", -1, "CLEVR_train_-00001.png"),
            ("train", True, "CLEVR_train_000001.png"),
            ("train", 7, "CLEVR_val_000007.png"),
            ("train", 7, "CLEVR_train_000008.png"),
            ("train", 7, "train_000007.png"),
        )
        for official_split, image_index, image_filename in invalid:
            with self.subTest(
                official_split=official_split,
                image_index=image_index,
                image_filename=image_filename,
            ), self.assertRaises(ValueError):
                ImageIdentity(official_split, image_index, image_filename)


class ImagePartitionTests(unittest.TestCase):
    def setUp(self):
        self.train_images = [
            _image("train", index)
            for index in (6, 4, 0, 8, 1, 7, 5, 9, 2, 3)
        ]
        self.val_images = [_image("val", index) for index in (6, 0, 5, 2, 4, 3, 1)]

    def test_sorts_before_seeded_permutation_and_uses_floor_boundaries(self):
        partitions = build_image_partitions(self.train_images, self.val_images)

        self.assertEqual(
            tuple(image.image_index for image in partitions.train),
            (3, 2, 9, 5, 7, 1, 8, 0, 4),
        )
        self.assertEqual(
            tuple(image.image_index for image in partitions.dev),
            (6,),
        )
        self.assertEqual(
            tuple(image.image_index for image in partitions.base_test),
            (1, 3, 4),
        )
        self.assertEqual(
            tuple(image.image_index for image in partitions.confirmation),
            (2, 0, 5, 6),
        )

    def test_outputs_are_immutable_tuples_with_complete_disjoint_coverage(self):
        partitions = build_image_partitions(self.train_images, self.val_images)

        for partition in (
            partitions.train,
            partitions.dev,
            partitions.base_test,
            partitions.confirmation,
        ):
            self.assertIsInstance(partition, tuple)

        train_partitioned = partitions.train + partitions.dev
        val_partitioned = partitions.base_test + partitions.confirmation
        self.assertEqual(set(train_partitioned), set(self.train_images))
        self.assertEqual(set(val_partitioned), set(self.val_images))
        self.assertEqual(len(train_partitioned), len(set(train_partitioned)))
        self.assertEqual(len(val_partitioned), len(set(val_partitioned)))
        self.assertTrue(set(partitions.train).isdisjoint(partitions.dev))
        self.assertTrue(set(partitions.base_test).isdisjoint(partitions.confirmation))

        with self.assertRaises(FrozenInstanceError):
            partitions.dev = ()
        with self.assertRaises(ValueError):
            ImagePartitions([], (), (), ())

    def test_repeatability_does_not_depend_on_input_order_or_mutate_inputs(self):
        original_train = list(self.train_images)
        original_val = list(self.val_images)

        first = build_image_partitions(self.train_images, self.val_images)
        second = build_image_partitions(
            tuple(reversed(self.train_images)), tuple(reversed(self.val_images))
        )

        self.assertEqual(first, second)
        self.assertEqual(self.train_images, original_train)
        self.assertEqual(self.val_images, original_val)

    def test_rejects_wrong_mother_set_splits_and_duplicate_identities(self):
        invalid_inputs = (
            ([_image("val", 0)], self.val_images),
            (self.train_images, [_image("train", 0)]),
            (self.train_images + [_image("train", 0)], self.val_images),
            (self.train_images, self.val_images + [_image("val", 0)]),
        )
        for train_images, val_images in invalid_inputs:
            with self.subTest(
                train_count=len(train_images), val_count=len(val_images)
            ), self.assertRaises(ValueError):
                build_image_partitions(train_images, val_images)


if __name__ == "__main__":
    unittest.main()
