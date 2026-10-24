"""PUB-R1.1 committed PCG64 training-index stream tests."""

from __future__ import annotations

import copy
import random
import unittest

import numpy as np
import torch

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

    def test_state_dict_is_exact_and_does_not_alias_stream(self) -> None:
        stream = CLEVRIndexStream(dataset_size=5, run_seed=101)
        stream.commit((0, 2, 1))

        state = stream.state_dict()

        self.assertEqual(
            set(state),
            {
                "protocol_version",
                "dataset_size",
                "run_seed",
                "sampler_seed",
                "epoch_index",
                "permutation",
                "cursor",
                "bit_generator_state",
            },
        )
        self.assertEqual(state["protocol_version"], "PUB-R1.1")
        self.assertEqual(state["dataset_size"], 5)
        self.assertEqual(state["run_seed"], 101)
        self.assertEqual(state["sampler_seed"], 10101)
        self.assertEqual(state["epoch_index"], 0)
        self.assertEqual(state["permutation"], (0, 2, 1, 4, 3))
        self.assertEqual(state["cursor"], 3)
        self.assertEqual(state["bit_generator_state"]["bit_generator"], "PCG64")

        state["cursor"] = 5
        state["permutation"] = (4, 3, 2, 1, 0)
        state["bit_generator_state"]["state"]["state"] = 0

        fresh = stream.state_dict()
        self.assertEqual(fresh["cursor"], 3)
        self.assertEqual(fresh["permutation"], (0, 2, 1, 4, 3))
        self.assertNotEqual(fresh["bit_generator_state"]["state"]["state"], 0)
        self.assertEqual(stream.preview(4), (4, 3, 4, 3))

        boundary = CLEVRIndexStream(dataset_size=5, run_seed=101)
        boundary.commit((0, 2, 1, 4, 3))
        boundary_state = boundary.state_dict()
        self.assertEqual(boundary_state["epoch_index"], 0)
        self.assertEqual(boundary_state["cursor"], 5)
        self.assertEqual(boundary_state["permutation"], (0, 2, 1, 4, 3))

    def test_restore_matches_uninterrupted_stream_inside_and_at_epoch_boundary(
        self,
    ) -> None:
        for committed in ((0, 2, 1, 4, 3), (0, 2, 1, 4, 3, 4, 3)):
            with self.subTest(committed=committed):
                uninterrupted = CLEVRIndexStream(dataset_size=5, run_seed=101)
                uninterrupted.commit(committed)
                restored = CLEVRIndexStream(dataset_size=5, run_seed=101)

                restored.load_state_dict(uninterrupted.state_dict())

                future = uninterrupted.preview(17)
                self.assertEqual(restored.preview(17), future)
                uninterrupted.commit(future)
                restored.commit(future)
                self.assertEqual(restored.state_dict(), uninterrupted.state_dict())

    def test_rejects_invalid_or_inconsistent_state_without_partial_mutation(
        self,
    ) -> None:
        source = CLEVRIndexStream(dataset_size=5, run_seed=101)
        source.commit((0, 2, 1))
        trusted = source.state_dict()

        epoch_one = CLEVRIndexStream(dataset_size=5, run_seed=101)
        epoch_one.commit((0, 2, 1, 4, 3, 4))
        epoch_one_state = epoch_one.state_dict()

        def changed(field: str, value: object) -> dict[str, object]:
            payload = copy.deepcopy(trusted)
            payload[field] = value
            return payload

        missing = copy.deepcopy(trusted)
        missing.pop("cursor")
        extra = copy.deepcopy(trusted)
        extra["extra"] = 1
        short_permutation = changed("permutation", (0, 2, 1, 4))
        duplicate_permutation = changed("permutation", (0, 2, 1, 4, 4))
        out_of_range_permutation = changed("permutation", (0, 2, 1, 4, 5))
        bool_permutation = changed("permutation", (False, 2, 1, 4, 3))
        mismatched_permutation = changed(
            "permutation", epoch_one_state["permutation"]
        )
        mismatched_rng = changed(
            "bit_generator_state", epoch_one_state["bit_generator_state"]
        )
        cases = (
            ([], r"mapping|state"),
            (missing, r"field|key|missing"),
            (extra, r"field|key|extra"),
            (changed("protocol_version", "PUB-R0"), r"protocol"),
            (changed("dataset_size", 6), r"dataset_size|constructor"),
            (changed("run_seed", 102), r"run_seed|constructor"),
            (changed("sampler_seed", 10102), r"sampler_seed|seed"),
            (changed("epoch_index", True), r"epoch_index|integer"),
            (changed("epoch_index", -1), r"epoch_index|non-negative"),
            (changed("cursor", True), r"cursor|integer"),
            (changed("cursor", np.int64(3)), r"cursor|integer"),
            (changed("cursor", -1), r"cursor|range"),
            (changed("cursor", 6), r"cursor|range"),
            (changed("permutation", [0, 2, 1, 4, 3]), r"permutation|tuple"),
            (short_permutation, r"permutation|length"),
            (duplicate_permutation, r"permutation|bijection"),
            (out_of_range_permutation, r"permutation|bijection|range"),
            (bool_permutation, r"permutation|integer"),
            (changed("bit_generator_state", {}), r"PCG64|bit_generator"),
            (mismatched_permutation, r"permutation|replay|consistent"),
            (mismatched_rng, r"PCG64|replay|consistent|state"),
            (changed("epoch_index", 1), r"permutation|PCG64|replay|consistent"),
        )
        for payload, pattern in cases:
            with self.subTest(pattern=pattern):
                target = CLEVRIndexStream(dataset_size=5, run_seed=101)
                target.commit((0, 2, 1))
                before = target.state_dict()
                next_indices = target.preview(9)

                with self.assertRaisesRegex(CLEVRSamplingError, pattern):
                    target.load_state_dict(payload)  # type: ignore[arg-type]

                self.assertEqual(target.state_dict(), before)
                self.assertEqual(target.preview(9), next_indices)

    def test_rejects_noncanonical_pcg64_scalar_types_without_mutation(
        self,
    ) -> None:
        source = CLEVRIndexStream(dataset_size=5, run_seed=101)
        source.commit((0, 2, 1))
        trusted = source.state_dict()
        trusted_rng = trusted["bit_generator_state"]

        malformed_values = (
            ("has_uint32", bool(trusted_rng["has_uint32"])),
            ("has_uint32", np.int64(trusted_rng["has_uint32"])),
            ("has_uint32", float(trusted_rng["has_uint32"])),
            ("uinteger", np.uint64(trusted_rng["uinteger"])),
            ("uinteger", float(trusted_rng["uinteger"])),
        )
        for field, value in malformed_values:
            with self.subTest(field=field, value_type=type(value).__name__):
                payload = copy.deepcopy(trusted)
                payload["bit_generator_state"][field] = value
                target = CLEVRIndexStream(dataset_size=5, run_seed=101)
                before = target.state_dict()
                next_indices = target.preview(9)

                with self.assertRaisesRegex(
                    CLEVRSamplingError, r"PCG64|bit_generator|integer"
                ):
                    target.load_state_dict(payload)

                self.assertEqual(target.state_dict(), before)
                self.assertEqual(target.preview(9), next_indices)

    def test_operations_leave_global_python_numpy_and_torch_rng_unchanged(
        self,
    ) -> None:
        random.seed(91)
        np.random.seed(92)
        torch.random.default_generator.manual_seed(93)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        stream = CLEVRIndexStream(dataset_size=5, run_seed=101)
        stream.commit(stream.preview(7))
        indices = stream.preview(7, offset=2)
        state = stream.state_dict()
        restored = CLEVRIndexStream(dataset_size=5, run_seed=101)
        restored.load_state_dict(state)
        self.assertEqual(restored.preview(7, offset=2), indices)

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
