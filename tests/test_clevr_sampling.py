"""PUB-R1.1 committed PCG64 training-index stream tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.clevr_sampling import CLEVRSamplingError, CLEVRIndexStream


class CLEVRIndexStreamTests(unittest.TestCase):
    def test_matches_fixed_pcg64_sequence_and_crosses_epoch_tail(self) -> None:
        stream = CLEVRIndexStream(dataset_size=5, run_seed=101)

        self.assertEqual(stream.dataset_size, 5)
        self.assertEqual(stream.run_seed, 101)
        self.assertEqual(stream.sampler_seed, 10101)
        self.assertEqual(stream.preview(7), (0, 2, 1, 4, 3, 4, 3))
        self.assertEqual(stream.preview(4, offset=1), (2, 1, 4, 3))

        stream.commit((0, 2, 1))
        self.assertEqual(stream.preview(4), (4, 3, 4, 3))
        stream.commit((4, 3, 4, 3))

        self.assertEqual(stream.preview(3), (1, 2, 0))

    def test_equal_run_inputs_produce_identical_paired_streams(self) -> None:
        left = CLEVRIndexStream(dataset_size=5, run_seed=101)
        right = CLEVRIndexStream(dataset_size=5, run_seed=101)

        for count in (3, 7, 11):
            with self.subTest(count=count):
                left_indices = left.preview(count)
                right_indices = right.preview(count)
                self.assertEqual(left_indices, right_indices)
                left.commit(left_indices)
                right.commit(list(right_indices))
                self.assertEqual(left.preview(9), right.preview(9))

    def test_preview_offset_and_multi_epoch_lookahead_are_pure(self) -> None:
        stream = CLEVRIndexStream(dataset_size=5, run_seed=101)
        initial = stream.preview(15)

        self.assertEqual(
            initial,
            (0, 2, 1, 4, 3, 4, 3, 1, 2, 0, 3, 0, 4, 2, 1),
        )
        self.assertEqual(
            stream.preview(8, offset=12),
            (4, 2, 1, 2, 3, 1, 0, 4),
        )
        self.assertEqual(stream.preview(15), initial)

    def test_wrong_commit_is_atomic_and_exact_commit_advances(self) -> None:
        stream = CLEVRIndexStream(dataset_size=5, run_seed=101)
        expected = stream.preview(4)

        with self.assertRaisesRegex(CLEVRSamplingError, r"commit|expected|prefix"):
            stream.commit((0, 1, 2, 4))

        self.assertEqual(stream.preview(4), expected)
        stream.commit(expected)
        self.assertEqual(stream.preview(5), (3, 4, 3, 1, 2))

    def test_rejects_invalid_constructor_preview_and_commit_inputs(self) -> None:
        constructor_cases = (
            ({"dataset_size": 0, "run_seed": 101}, r"dataset_size|positive"),
            ({"dataset_size": -1, "run_seed": 101}, r"dataset_size|positive"),
            ({"dataset_size": 5.0, "run_seed": 101}, r"dataset_size|integer"),
            ({"dataset_size": True, "run_seed": 101}, r"dataset_size|integer"),
            (
                {"dataset_size": np.int64(5), "run_seed": 101},
                r"dataset_size|integer",
            ),
            ({"dataset_size": 5, "run_seed": -1}, r"run_seed|non-negative"),
            ({"dataset_size": 5, "run_seed": 1.0}, r"run_seed|integer"),
            ({"dataset_size": 5, "run_seed": False}, r"run_seed|integer"),
            (
                {"dataset_size": 5, "run_seed": np.int64(101)},
                r"run_seed|integer",
            ),
        )
        for arguments, pattern in constructor_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(CLEVRSamplingError, pattern):
                    CLEVRIndexStream(**arguments)  # type: ignore[arg-type]

        stream = CLEVRIndexStream(dataset_size=5, run_seed=101)
        for count in (0, -1, 1.0, True, np.int64(1)):
            with self.subTest(preview_count=count):
                with self.assertRaisesRegex(CLEVRSamplingError, r"count|positive"):
                    stream.preview(count)  # type: ignore[arg-type]
        for offset in (-1, 1.0, True, np.int64(1)):
            with self.subTest(preview_offset=offset):
                with self.assertRaisesRegex(
                    CLEVRSamplingError, r"offset|non-negative|integer"
                ):
                    stream.preview(1, offset=offset)  # type: ignore[arg-type]

        invalid_commits = (
            (),
            "0",
            (index for index in (0,)),
            (True,),
            (0.0,),
            (np.int64(0),),
            (-1,),
            (5,),
        )
        for indices in invalid_commits:
            with self.subTest(commit=repr(indices)):
                with self.assertRaisesRegex(
                    CLEVRSamplingError, r"commit|sequence|integer|range|empty"
                ):
                    stream.commit(indices)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
