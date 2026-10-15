"""CPU-only checks for fixed CIFAR-100 splits and held-out evaluation."""

from collections import Counter
import random
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.cifar_data import (
    CIFARDataConfig,
    build_datasets,
    build_test_dataset,
    split_train_indices,
)


class SyntheticCIFAR:
    """Replace external dataset I/O while retaining the torchvision contract."""

    def __init__(self, train=True, transform=None):
        self.targets = np.repeat(np.arange(100), 500 if train else 100).tolist()
        self.classes = [f"class_{index:03d}" for index in range(100)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
        image = Image.fromarray(((pixels + index) % 256).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


class CIFARDataTests(unittest.TestCase):
    def test_split_is_stratified_disjoint_complete_and_independent_of_training_rng(self):
        targets = np.repeat(np.arange(100), 500).tolist()
        train, dev = split_train_indices(targets, dev_per_class=50, seed=123)
        self.assertEqual((len(train), len(dev)), (45000, 5000))
        self.assertEqual(Counter(targets[i] for i in train), {i: 450 for i in range(100)})
        self.assertEqual(Counter(targets[i] for i in dev), {i: 50 for i in range(100)})
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(set(train) | set(dev), set(range(50000)))
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        self.assertEqual((train, dev), split_train_indices(targets, dev_per_class=50, seed=123))
        self.assertNotEqual(dev, split_train_indices(targets, dev_per_class=50, seed=124)[1])

    def test_split_rejects_empty_classes_or_a_development_split_that_removes_all_training(self):
        for labels, count in [([], 1), ([0, 0, 1], 1), ([0, 0, 1, 1], 0), ([0, -1], 1)]:
            with self.subTest(labels=labels, count=count), self.assertRaises(ValueError):
                split_train_indices(labels, dev_per_class=count, seed=1)

    def test_training_bundle_never_loads_official_test_and_dev_has_no_random_augmentation(self):
        def official_train_only(*, root, train, transform=None, download=False):
            if not train:
                raise AssertionError("Official test was opened during training preparation")
            return SyntheticCIFAR(train=True, transform=transform)

        with patch("src.cifar_data.CIFAR100", side_effect=official_train_only):
            bundle = build_datasets(CIFARDataConfig(root="unused", image_size=32))
        self.assertEqual((len(bundle.train), len(bundle.dev)), (45000, 5000))
        self.assertEqual(bundle.metadata["official_test_used"], False)
        self.assertEqual(len(bundle.classes), 100)
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        first_image, first_label = bundle.dev[0]
        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)
        second_image, second_label = bundle.dev[0]
        torch.testing.assert_close(first_image, second_image, rtol=0, atol=0)
        self.assertEqual(first_label, second_label)
        self.assertEqual(tuple(first_image.shape), (3, 32, 32))
        self.assertEqual(first_image.dtype, torch.float32)

    def test_training_transforms_are_seed_reproducible_at_native_and_resized_resolution(self):
        for augmentation in ("basic", "deit", "autoaugment", "threeaugment"):
            for size in (32, 64):
                with self.subTest(augmentation=augmentation, image_size=size):
                    with patch("src.cifar_data.CIFAR100", return_value=SyntheticCIFAR()):
                        bundle = build_datasets(CIFARDataConfig(
                            root="unused", image_size=size, augmentation=augmentation
                        ))
                    outputs = []
                    for _ in range(2):
                        random.seed(11)
                        np.random.seed(11)
                        torch.manual_seed(11)
                        outputs.append(bundle.train[0])
                    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
                    self.assertEqual(outputs[0][1], outputs[1][1])
                    self.assertEqual(tuple(outputs[0][0].shape), (3, size, size))
                    self.assertTrue(torch.isfinite(outputs[0][0]).all())

    def test_official_test_is_loaded_only_by_the_explicit_final_evaluation_builder(self):
        def official_test_only(*, root, train, transform=None, download=False):
            if train:
                raise AssertionError("Final test builder opened official training data")
            return SyntheticCIFAR(train=False, transform=transform)

        with patch("src.cifar_data.CIFAR100", side_effect=official_test_only):
            test = build_test_dataset(CIFARDataConfig(root="unused", image_size=64))
        self.assertEqual(len(test), 10000)
        image, label = test[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(label, 0)

    def test_wrong_official_training_size_and_invalid_transform_settings_are_rejected(self):
        invalid = SyntheticCIFAR()
        invalid.targets = invalid.targets[:-1]
        with patch("src.cifar_data.CIFAR100", return_value=invalid), self.assertRaises(ValueError):
            build_datasets(CIFARDataConfig(root="unused"))
        for kwargs in ({"image_size": 0}, {"augmentation": "unknown"}, {"random_erasing": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_datasets(CIFARDataConfig(root="unused", **kwargs))


if __name__ == "__main__":
    unittest.main()
