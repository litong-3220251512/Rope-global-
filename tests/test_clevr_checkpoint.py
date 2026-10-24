"""PUB-R1.1 checkpoint snapshot and publication tests."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import platform
import random
import stat
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src import clevr_checkpoint as checkpoint_module
from src.clevr_checkpoint import (
    CLEVRCheckpointError,
    CLEVRCheckpointMetadata,
    CLEVRCheckpointPublicationError,
    save_clevr_checkpoint,
)
from src.clevr_sampling import CLEVRIndexStream


def _clone_state(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((key, _clone_state(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


class _CheckpointFixture:
    def __init__(self) -> None:
        torch.manual_seed(17)
        self.model = nn.Sequential(nn.Linear(3, 2))
        named_parameters = dict(self.model.named_parameters())
        self.optimizer = AdamW(
            [
                {
                    "params": [named_parameters["0.weight"]],
                    "group_name": "decay",
                    "param_names": ["0.weight"],
                    "lr": 0.01,
                },
                {
                    "params": [named_parameters["0.bias"]],
                    "group_name": "no_decay",
                    "param_names": ["0.bias"],
                    "lr": 0.02,
                    "weight_decay": 0.0,
                },
            ]
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=(lambda _: 1.0, lambda _: 1.0),
        )

        for index, parameter in enumerate(self.model.parameters(), start=1):
            parameter.grad = torch.full_like(parameter, index / 10)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.sampler = CLEVRIndexStream(dataset_size=5, run_seed=101)
        committed = self.sampler.preview(128)
        self.sampler.commit(committed)

        random.seed(31)
        np.random.seed(32)
        torch.manual_seed(33)


class CheckpointSaveTests(unittest.TestCase):
    def assert_state_equal(self, actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertTrue(torch.equal(actual, expected))
            return
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
            return
        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_state_equal(actual_item, expected_item)
            return
        self.assertEqual(actual, expected)

    def assert_numpy_global_state_equal(
        self,
        actual: tuple[object, ...],
        expected: tuple[object, ...],
    ) -> None:
        self.assertEqual(actual[0], expected[0])
        self.assertTrue(np.array_equal(actual[1], expected[1]))
        self.assertEqual(actual[2:], expected[2:])

    def assert_numpy_payload_equal(
        self,
        actual: dict[str, object],
        expected: tuple[object, ...],
    ) -> None:
        self.assertEqual(
            set(actual),
            {"bit_generator", "keys", "pos", "has_gauss", "cached_gaussian"},
        )
        self.assertEqual(actual["bit_generator"], "MT19937")
        self.assertTrue(
            torch.equal(
                actual["keys"],
                torch.from_numpy(expected[1].astype(np.int64)),
            )
        )
        self.assertEqual(actual["pos"], expected[2])
        self.assertEqual(actual["has_gauss"], expected[3])
        self.assertEqual(actual["cached_gaussian"], expected[4])

    def test_saves_complete_weights_only_payload_at_update_boundary(self) -> None:
        fixture = _CheckpointFixture()
        expected_model = _clone_state(fixture.model.state_dict())
        expected_optimizer = _clone_state(fixture.optimizer.state_dict())
        expected_scheduler = _clone_state(fixture.scheduler.state_dict())
        expected_sampler = _clone_state(fixture.sampler.state_dict())
        expected_python = copy.deepcopy(random.getstate())
        expected_numpy = np.random.get_state(legacy=True)
        expected_torch = torch.random.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "update-0001.pt"
            link_observations: list[tuple[int, int, int, int]] = []
            real_link = os.link

            def recording_link(source: str | bytes, destination: str | bytes) -> None:
                source_stat = os.stat(source)
                real_link(source, destination)
                destination_stat = os.stat(destination)
                link_observations.append(
                    (
                        source_stat.st_dev,
                        source_stat.st_ino,
                        destination_stat.st_dev,
                        destination_stat.st_ino,
                    )
                )

            with (
                mock.patch.object(
                    checkpoint_module.os, "link", side_effect=recording_link
                ),
                mock.patch.object(
                    torch.cuda,
                    "get_rng_state",
                    side_effect=AssertionError("empty CUDA tuple touched CUDA RNG"),
                ),
                mock.patch.object(
                    torch.cuda,
                    "get_rng_state_all",
                    side_effect=AssertionError("empty CUDA tuple enumerated CUDA RNG"),
                ),
            ):
                metadata = save_clevr_checkpoint(
                    target,
                    method="G",
                    run_seed=101,
                    completed_updates=1,
                    model=fixture.model,
                    optimizer=fixture.optimizer,
                    scheduler=fixture.scheduler,
                    sampler=fixture.sampler,
                    cuda_devices=(),
                )

            self.assertEqual(random.getstate(), expected_python)
            self.assert_numpy_global_state_equal(
                np.random.get_state(legacy=True),
                expected_numpy,
            )
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_torch))

            payload = torch.load(target, map_location="cpu", weights_only=True)

            self.assertEqual(
                set(payload),
                {
                    "protocol_version",
                    "checkpoint_schema_version",
                    "method",
                    "run_seed",
                    "completed_updates",
                    "dataset_size",
                    "model_state",
                    "optimizer_state",
                    "scheduler_state",
                    "sampler_state",
                    "rng_state",
                    "environment",
                },
            )
            self.assertEqual(payload["protocol_version"], "PUB-R1.1")
            self.assertEqual(payload["checkpoint_schema_version"], 1)
            self.assertEqual(payload["method"], "G")
            self.assertEqual(payload["run_seed"], 101)
            self.assertEqual(payload["completed_updates"], 1)
            self.assertEqual(payload["dataset_size"], 5)
            self.assertEqual(
                set(payload["rng_state"]),
                {"python", "numpy", "torch_cpu", "cuda_devices", "torch_cuda"},
            )
            self.assertEqual(
                payload["environment"],
                {
                    "python": "3.11.14",
                    "numpy": "2.2.6",
                    "torch": "2.6.0+cu124",
                    "cuda_runtime": "12.4",
                },
            )
            self.assertEqual(platform.python_version(), "3.11.14")

            self.assert_state_equal(payload["model_state"], expected_model)
            self.assert_state_equal(payload["optimizer_state"], expected_optimizer)
            self.assert_state_equal(payload["scheduler_state"], expected_scheduler)
            self.assert_state_equal(payload["sampler_state"], expected_sampler)
            self.assertEqual(payload["rng_state"]["python"], expected_python)
            self.assert_numpy_payload_equal(
                payload["rng_state"]["numpy"],
                expected_numpy,
            )
            self.assertTrue(torch.equal(payload["rng_state"]["torch_cpu"], expected_torch))
            self.assertEqual(payload["rng_state"]["cuda_devices"], ())
            self.assertEqual(payload["rng_state"]["torch_cuda"], ())

            self.assertEqual(
                link_observations,
                [
                    (
                        target.stat().st_dev,
                        target.stat().st_ino,
                        target.stat().st_dev,
                        target.stat().st_ino,
                    )
                ],
            )
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertEqual(list(target.parent.glob(f".{target.name}.staging-*")), [])
            self.assertEqual(metadata.path, str(target.resolve()))
            self.assertEqual(metadata.size_bytes, target.stat().st_size)
            self.assertEqual(metadata.method, "G")
            self.assertEqual(metadata.run_seed, 101)
            self.assertEqual(metadata.completed_updates, 1)
            self.assertEqual(metadata.dataset_size, 5)
            self.assertEqual(metadata.cuda_devices, ())

            with torch.no_grad():
                for parameter in fixture.model.parameters():
                    parameter.add_(100)
                for state in fixture.optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.add_(200)
            fixture.optimizer.param_groups[0]["lr"] = 9.0
            fixture.scheduler.last_epoch = 99
            fixture.scheduler._step_count = 100
            fixture.scheduler._last_lr = [9.0, 9.0]
            fixture.sampler.commit(fixture.sampler.preview(1))
            random.random()
            np.random.random()
            torch.rand(1)

            self.assert_state_equal(payload["model_state"], expected_model)
            self.assert_state_equal(payload["optimizer_state"], expected_optimizer)
            self.assert_state_equal(payload["scheduler_state"], expected_scheduler)
            self.assert_state_equal(payload["sampler_state"], expected_sampler)
            self.assertEqual(payload["rng_state"]["python"], expected_python)
            self.assert_numpy_payload_equal(
                payload["rng_state"]["numpy"],
                expected_numpy,
            )
            self.assertTrue(torch.equal(payload["rng_state"]["torch_cpu"], expected_torch))

    def test_boundary_hand_example_preserves_next_committed_indices(self) -> None:
        fixture = _CheckpointFixture()
        literal_next_prefix = (1, 3, 1, 2, 0, 3, 4, 0, 3, 2, 4, 1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "boundary.pt"
            save_clevr_checkpoint(
                target,
                method="G",
                run_seed=101,
                completed_updates=1,
                model=fixture.model,
                optimizer=fixture.optimizer,
                scheduler=fixture.scheduler,
                sampler=fixture.sampler,
                cuda_devices=(),
            )
            payload = torch.load(target, map_location="cpu", weights_only=True)

            sampler_state = payload["sampler_state"]
            consumed = sampler_state["epoch_index"] * 5 + sampler_state["cursor"]
            self.assertEqual(consumed, 128)
            self.assertEqual(fixture.sampler.preview(12), literal_next_prefix)

            restored = CLEVRIndexStream(dataset_size=5, run_seed=101)
            restored.load_state_dict(sampler_state)

            # Catches saving the preview/dispatched position instead of the
            # committed position at the completed optimizer-update boundary.
            self.assertEqual(restored.preview(12), literal_next_prefix)

    def test_complete_epoch_tail_boundary_is_preserved_exactly(self) -> None:
        fixture = _CheckpointFixture()
        fixture.sampler = CLEVRIndexStream(dataset_size=8, run_seed=101)
        fixture.sampler.commit(fixture.sampler.preview(128))

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "epoch-tail.pt"
            save_clevr_checkpoint(
                target,
                method="G",
                run_seed=101,
                completed_updates=1,
                model=fixture.model,
                optimizer=fixture.optimizer,
                scheduler=fixture.scheduler,
                sampler=fixture.sampler,
                cuda_devices=(),
            )
            payload = torch.load(target, map_location="cpu", weights_only=True)

            sampler_state = payload["sampler_state"]
            self.assertEqual(sampler_state["dataset_size"], 8)
            self.assertEqual(sampler_state["epoch_index"], 15)
            self.assertEqual(sampler_state["cursor"], 8)
            self.assertEqual(
                sampler_state["epoch_index"] * 8 + sampler_state["cursor"],
                128,
            )

            uninterrupted_next = fixture.sampler.preview(12)
            restored = CLEVRIndexStream(dataset_size=8, run_seed=101)
            restored.load_state_dict(sampler_state)

            # Catches rejecting or normalizing the legal c == D boundary.
            self.assertEqual(restored.preview(12), uninterrupted_next)


class CheckpointSaveValidationTests(unittest.TestCase):
    def assert_rejected_before_staging(
        self,
        *,
        fixture: _CheckpointFixture | None = None,
        target: Path | str | None = None,
        **overrides: object,
    ) -> None:
        fixture = fixture or _CheckpointFixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            default_target = Path(temporary_directory) / "rejected.pt"
            arguments: dict[str, object] = {
                "path": target if target is not None else default_target,
                "method": "G",
                "run_seed": 101,
                "completed_updates": 1,
                "model": fixture.model,
                "optimizer": fixture.optimizer,
                "scheduler": fixture.scheduler,
                "sampler": fixture.sampler,
                "cuda_devices": (),
            }
            arguments.update(overrides)
            with mock.patch.object(
                checkpoint_module.tempfile,
                "mkstemp",
                side_effect=AssertionError("invalid input reached staging I/O"),
            ):
                with self.assertRaises(CLEVRCheckpointError):
                    save_clevr_checkpoint(**arguments)  # type: ignore[arg-type]
            if target is None:
                self.assertFalse(default_target.exists())
                self.assertEqual(
                    list(default_target.parent.glob(f".{default_target.name}.staging-*")),
                    [],
                )

    def test_public_records_are_frozen_and_reject_inconsistent_construction(
        self,
    ) -> None:
        metadata = CLEVRCheckpointMetadata(
            path="/tmp/checkpoint.pt",
            size_bytes=1,
            method="G",
            run_seed=101,
            completed_updates=1,
            dataset_size=5,
            cuda_devices=(),
        )
        with self.assertRaises(FrozenInstanceError):
            metadata.size_bytes = 2  # type: ignore[misc]

        invalid_metadata = (
            {"path": "", "size_bytes": 1},
            {"path": "/tmp/checkpoint.pt", "size_bytes": True},
            {"path": "/tmp/checkpoint.pt", "size_bytes": -1},
            {"path": "/tmp/checkpoint.pt", "size_bytes": 1, "method": "X"},
            {"path": "/tmp/checkpoint.pt", "size_bytes": 1, "run_seed": False},
            {
                "path": "/tmp/checkpoint.pt",
                "size_bytes": 1,
                "completed_updates": 8001,
            },
            {"path": "/tmp/checkpoint.pt", "size_bytes": 1, "dataset_size": 0},
            {"path": "/tmp/checkpoint.pt", "size_bytes": 1, "cuda_devices": [0]},
        )
        defaults = {
            "method": "G",
            "run_seed": 101,
            "completed_updates": 1,
            "dataset_size": 5,
            "cuda_devices": (),
        }
        for changes in invalid_metadata:
            with self.subTest(changes=changes):
                arguments = dict(defaults)
                arguments.update(changes)
                with self.assertRaises(CLEVRCheckpointError):
                    CLEVRCheckpointMetadata(**arguments)  # type: ignore[arg-type]

        with self.assertRaisesRegex(CLEVRCheckpointError, r"committed|bool"):
            CLEVRCheckpointPublicationError("invalid", committed=1)  # type: ignore[arg-type]

    def test_rejects_scalar_object_and_cuda_contract_violations_before_io(
        self,
    ) -> None:
        scalar_cases = (
            {"method": "global"},
            {"method": 1},
            {"run_seed": True},
            {"run_seed": -1},
            {"completed_updates": False},
            {"completed_updates": -1},
            {"completed_updates": 8001},
            {"cuda_devices": [0]},
            {"cuda_devices": (1, 0)},
            {"cuda_devices": (0, 0)},
            {"cuda_devices": (False,)},
            {"cuda_devices": (-1,)},
        )
        for overrides in scalar_cases:
            with self.subTest(overrides=overrides):
                self.assert_rejected_before_staging(**overrides)

        object_cases = (
            {"path": "checkpoint.pt"},
            {"model": object()},
            {"optimizer": object()},
            {"scheduler": object()},
            {"sampler": object()},
        )
        for overrides in object_cases:
            with self.subTest(overrides=tuple(overrides)):
                self.assert_rejected_before_staging(**overrides)

        with (
            mock.patch.object(torch.cuda, "is_initialized", return_value=False),
            mock.patch.object(
                torch.cuda,
                "get_rng_state",
                side_effect=AssertionError("uninitialized CUDA RNG was accessed"),
            ),
        ):
            self.assert_rejected_before_staging(cuda_devices=(0,))

    def test_rejects_inconsistent_live_state_and_update_boundaries_before_io(
        self,
    ) -> None:
        fixture = _CheckpointFixture()
        fixture.model[0].weight.grad = torch.zeros_like(fixture.model[0].weight)
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        alternate_optimizer = AdamW(fixture.model.parameters(), lr=0.01)
        alternate_scheduler = LambdaLR(alternate_optimizer, lambda _: 1.0)
        self.assert_rejected_before_staging(
            fixture=fixture,
            scheduler=alternate_scheduler,
        )

        fixture = _CheckpointFixture()
        fixture.scheduler.last_epoch = 0
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        fixture.scheduler._step_count = 1
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        fixture.optimizer.param_groups[0]["lr"] = 0.5
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        self.assert_rejected_before_staging(fixture=fixture, run_seed=102)

        fixture = _CheckpointFixture()
        fixture.sampler._sampler_seed += 1
        self.assert_rejected_before_staging(fixture=fixture)

        class MismatchedSizeStream(CLEVRIndexStream):
            @property
            def dataset_size(self) -> int:
                return super().dataset_size + 1

        fixture = _CheckpointFixture()
        self.assert_rejected_before_staging(
            fixture=fixture,
            sampler=MismatchedSizeStream(dataset_size=5, run_seed=101),
        )

        for consumed in (127, 129):
            with self.subTest(consumed=consumed):
                fixture = _CheckpointFixture()
                off_boundary = CLEVRIndexStream(dataset_size=5, run_seed=101)
                off_boundary.commit(off_boundary.preview(consumed))
                self.assert_rejected_before_staging(
                    fixture=fixture,
                    sampler=off_boundary,
                )

    def test_rejects_optimizer_identity_name_order_and_coverage_before_io(
        self,
    ) -> None:
        fixture = _CheckpointFixture()
        fixture.optimizer.param_groups[0]["param_names"] = ["0.bias"]
        fixture.optimizer.param_groups[1]["param_names"] = ["0.weight"]
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        weight = fixture.model[0].weight
        bias = fixture.model[0].bias
        fixture.optimizer.param_groups[0]["params"] = [bias]
        fixture.optimizer.param_groups[1]["params"] = [weight]
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        weight = fixture.model[0].weight
        fixture.optimizer.param_groups[1]["params"] = [weight]
        fixture.optimizer.param_groups[1]["param_names"] = ["0.weight"]
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        fixture.optimizer.param_groups[1]["group_name"] = "decay"
        self.assert_rejected_before_staging(fixture=fixture)

        fixture = _CheckpointFixture()
        fixture.optimizer.param_groups[0]["param_names"] = []
        self.assert_rejected_before_staging(fixture=fixture)

    def test_rejects_missing_or_non_directory_parent_before_io(self) -> None:
        fixture = _CheckpointFixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_target = root / "missing" / "checkpoint.pt"
            self.assert_rejected_before_staging(
                fixture=fixture,
                target=missing_target,
            )

            parent_file = root / "not-a-directory"
            parent_file.write_bytes(b"parent")
            non_directory_target = parent_file / "checkpoint.pt"
            self.assert_rejected_before_staging(
                fixture=fixture,
                target=non_directory_target,
            )
            self.assertEqual(parent_file.read_bytes(), b"parent")


class _FileProxy:
    def __init__(
        self,
        wrapped: object,
        *,
        flush_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.flush_error = flush_error
        self.close_error = close_error

    def __getattr__(self, name: str) -> object:
        return getattr(self.wrapped, name)

    def __enter__(self) -> _FileProxy:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

    def flush(self) -> None:
        if self.flush_error is not None:
            raise self.flush_error
        self.wrapped.flush()

    def close(self) -> None:
        self.wrapped.close()
        if self.close_error is not None:
            raise self.close_error


class CheckpointPublicationTests(unittest.TestCase):
    def save(self, target: Path, fixture: _CheckpointFixture | None = None) -> object:
        fixture = fixture or _CheckpointFixture()
        return save_clevr_checkpoint(
            target,
            method="G",
            run_seed=101,
            completed_updates=1,
            model=fixture.model,
            optimizer=fixture.optimizer,
            scheduler=fixture.scheduler,
            sampler=fixture.sampler,
            cuda_devices=(),
        )

    def assert_no_staging(self, target: Path) -> None:
        self.assertEqual(list(target.parent.glob(f".{target.name}.staging-*")), [])

    def descriptor_is_closed(self, descriptor: int) -> bool:
        try:
            os.fstat(descriptor)
        except OSError:
            return True
        return False

    def test_first_post_mkstemp_interrupt_closes_descriptor_and_removes_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "setup-interrupt.pt"
            descriptors: list[int] = []
            staging_paths: list[str] = []
            primary = KeyboardInterrupt("constructing staging Path")
            real_mkstemp = tempfile.mkstemp

            def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
                descriptor, staging_name = real_mkstemp(*args, **kwargs)
                descriptors.append(descriptor)
                staging_paths.append(staging_name)
                return descriptor, staging_name

            observed: BaseException | None = None
            try:
                with (
                    mock.patch.object(
                        checkpoint_module.tempfile,
                        "mkstemp",
                        side_effect=recording_mkstemp,
                    ),
                    mock.patch.object(checkpoint_module, "Path", side_effect=primary),
                ):
                    try:
                        checkpoint_module._publish_new_file(target, {})
                    except BaseException as error:
                        observed = error

                descriptor_closed = self.descriptor_is_closed(descriptors[0])
                staging_exists = os.path.lexists(staging_paths[0])
            finally:
                if descriptors and not self.descriptor_is_closed(descriptors[0]):
                    os.close(descriptors[0])
                if staging_paths and os.path.lexists(staging_paths[0]):
                    os.unlink(staging_paths[0])

            self.assertIs(observed, primary)
            self.assertTrue(descriptor_closed)
            self.assertFalse(staging_exists)
            self.assertFalse(target.exists())

    def test_first_post_mkstemp_interrupt_reports_cleanup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "setup-cleanup-failure.pt"
            descriptors: list[int] = []
            staging_paths: list[str] = []
            primary = KeyboardInterrupt("constructing staging Path")
            cleanup = OSError("setup staging unlink failed")
            real_mkstemp = tempfile.mkstemp

            def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
                descriptor, staging_name = real_mkstemp(*args, **kwargs)
                descriptors.append(descriptor)
                staging_paths.append(staging_name)
                return descriptor, staging_name

            observed: BaseException | None = None
            try:
                with (
                    mock.patch.object(
                        checkpoint_module.tempfile,
                        "mkstemp",
                        side_effect=recording_mkstemp,
                    ),
                    mock.patch.object(checkpoint_module, "Path", side_effect=primary),
                    mock.patch.object(
                        checkpoint_module.os,
                        "unlink",
                        side_effect=cleanup,
                    ),
                ):
                    try:
                        checkpoint_module._publish_new_file(target, {})
                    except BaseException as error:
                        observed = error

                descriptor_closed = self.descriptor_is_closed(descriptors[0])
                staging_exists = os.path.lexists(staging_paths[0])
            finally:
                if descriptors and not self.descriptor_is_closed(descriptors[0]):
                    os.close(descriptors[0])
                if staging_paths and os.path.lexists(staging_paths[0]):
                    os.unlink(staging_paths[0])

            self.assertIsInstance(observed, CLEVRCheckpointPublicationError)
            self.assertFalse(observed.committed)
            self.assertIs(observed.__cause__, primary)
            self.assertTrue(
                any("unlink" in note for note in observed.__notes__)
            )
            self.assertTrue(descriptor_closed)
            self.assertTrue(staging_exists)
            self.assertFalse(target.exists())

    def test_reconciliation_interrupt_retains_primary_and_continues_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "reconciliation-interrupt.pt"
            descriptors: list[int] = []
            staging_paths: list[str] = []
            primary = OSError("ambiguous link failure")
            reconciliation = KeyboardInterrupt("inode reconciliation interrupted")
            real_mkstemp = tempfile.mkstemp

            def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
                descriptor, staging_name = real_mkstemp(*args, **kwargs)
                descriptors.append(descriptor)
                staging_paths.append(staging_name)
                return descriptor, staging_name

            observed: BaseException | None = None
            try:
                with (
                    mock.patch.object(
                        checkpoint_module.tempfile,
                        "mkstemp",
                        side_effect=recording_mkstemp,
                    ),
                    mock.patch.object(
                        checkpoint_module.os,
                        "link",
                        side_effect=primary,
                    ),
                    mock.patch.object(
                        checkpoint_module,
                        "_paths_are_same_regular_inode",
                        side_effect=reconciliation,
                    ),
                ):
                    try:
                        self.save(target)
                    except BaseException as error:
                        observed = error

                descriptor_closed = self.descriptor_is_closed(descriptors[0])
                staging_exists = os.path.lexists(staging_paths[0])
            finally:
                if descriptors and not self.descriptor_is_closed(descriptors[0]):
                    os.close(descriptors[0])
                if staging_paths and os.path.lexists(staging_paths[0]):
                    os.unlink(staging_paths[0])

            self.assertIsInstance(observed, CLEVRCheckpointPublicationError)
            self.assertFalse(observed.committed)
            self.assertIs(observed.__cause__, primary)
            self.assertTrue(
                any(
                    "reconciliation" in note and "KeyboardInterrupt" in note
                    for note in observed.__notes__
                )
            )
            self.assertTrue(descriptor_closed)
            self.assertFalse(staging_exists)
            self.assertFalse(target.exists())

    def test_existing_file_directory_and_symlink_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            existing_file = root / "existing-file.pt"
            existing_file.write_bytes(b"existing file bytes")
            with self.assertRaises(CLEVRCheckpointError):
                self.save(existing_file)
            self.assertEqual(existing_file.read_bytes(), b"existing file bytes")

            existing_directory = root / "existing-directory.pt"
            existing_directory.mkdir()
            child = existing_directory / "child"
            child.write_bytes(b"child bytes")
            with self.assertRaises(CLEVRCheckpointError):
                self.save(existing_directory)
            self.assertTrue(existing_directory.is_dir())
            self.assertEqual(child.read_bytes(), b"child bytes")

            symlink_source = root / "symlink-source"
            symlink_source.write_bytes(b"symlink bytes")
            existing_symlink = root / "existing-symlink.pt"
            existing_symlink.symlink_to(symlink_source)
            with self.assertRaises(CLEVRCheckpointError):
                self.save(existing_symlink)
            self.assertTrue(existing_symlink.is_symlink())
            self.assertEqual(symlink_source.read_bytes(), b"symlink bytes")

            for target in (existing_file, existing_directory, existing_symlink):
                self.assert_no_staging(target)

    def test_writer_flush_close_and_link_failures_leave_no_artifacts(self) -> None:
        scenarios = ("writer", "flush", "close", "link")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary_directory:
                target = Path(temporary_directory) / f"{scenario}.pt"
                fixture = _CheckpointFixture()
                patches: list[mock._patch] = []
                if scenario == "writer":
                    patches.append(
                        mock.patch.object(
                            checkpoint_module.torch,
                            "save",
                            side_effect=OSError("injected writer failure"),
                        )
                    )
                elif scenario in {"flush", "close"}:
                    real_fdopen = os.fdopen

                    def wrapped_fdopen(
                        descriptor: int,
                        mode: str,
                        *,
                        _scenario: str = scenario,
                    ) -> _FileProxy:
                        return _FileProxy(
                            real_fdopen(descriptor, mode),
                            flush_error=(
                                OSError("injected flush failure")
                                if _scenario == "flush"
                                else None
                            ),
                            close_error=(
                                OSError("injected close failure")
                                if _scenario == "close"
                                else None
                            ),
                        )

                    patches.append(
                        mock.patch.object(
                            checkpoint_module.os,
                            "fdopen",
                            side_effect=wrapped_fdopen,
                        )
                    )
                else:
                    patches.append(
                        mock.patch.object(
                            checkpoint_module.os,
                            "link",
                            side_effect=OSError("injected link failure"),
                        )
                    )

                with patches[0]:
                    with self.assertRaises(OSError):
                        self.save(target, fixture)
                self.assertFalse(target.exists())
                self.assert_no_staging(target)

    def test_target_appearing_before_link_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "raced.pt"
            real_link = os.link

            def racing_link(source: str | bytes, destination: str | bytes) -> None:
                Path(destination).write_bytes(b"concurrent writer")
                real_link(source, destination)

            with mock.patch.object(
                checkpoint_module.os,
                "link",
                side_effect=racing_link,
            ):
                with self.assertRaises(FileExistsError):
                    self.save(target)

            self.assertEqual(target.read_bytes(), b"concurrent writer")
            self.assert_no_staging(target)

    def test_link_side_effect_then_interrupt_reports_committed_true(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "ambiguous-link.pt"
            real_link = os.link
            interrupt = KeyboardInterrupt("after real link")

            def link_then_interrupt(source: str | bytes, destination: str | bytes) -> None:
                real_link(source, destination)
                raise interrupt

            with mock.patch.object(
                checkpoint_module.os,
                "link",
                side_effect=link_then_interrupt,
            ):
                with self.assertRaises(CLEVRCheckpointPublicationError) as caught:
                    self.save(target)

            self.assertTrue(caught.exception.committed)
            self.assertIs(caught.exception.__cause__, interrupt)
            payload = torch.load(target, map_location="cpu", weights_only=True)
            self.assertEqual(payload["completed_updates"], 1)
            self.assert_no_staging(target)

    def test_cleanup_failure_before_commit_preserves_primary_and_owned_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "precommit-cleanup.pt"
            primary = OSError("writer failed")
            cleanup = OSError("staging unlink failed")
            with (
                mock.patch.object(checkpoint_module.torch, "save", side_effect=primary),
                mock.patch.object(checkpoint_module.os, "unlink", side_effect=cleanup),
            ):
                with self.assertRaises(CLEVRCheckpointPublicationError) as caught:
                    self.save(target)

            self.assertFalse(caught.exception.committed)
            self.assertIs(caught.exception.__cause__, primary)
            self.assertTrue(
                any("unlink" in note or "cleanup" in note for note in caught.exception.__notes__)
            )
            self.assertFalse(target.exists())
            self.assertEqual(len(list(target.parent.glob(f".{target.name}.staging-*"))), 1)

    def test_cleanup_failure_after_commit_keeps_final_and_owned_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "postcommit-cleanup.pt"
            cleanup = OSError("staging unlink failed")
            with mock.patch.object(checkpoint_module.os, "unlink", side_effect=cleanup):
                with self.assertRaises(CLEVRCheckpointPublicationError) as caught:
                    self.save(target)

            self.assertTrue(caught.exception.committed)
            self.assertIs(caught.exception.__cause__, cleanup)
            payload = torch.load(target, map_location="cpu", weights_only=True)
            self.assertEqual(payload["method"], "G")
            self.assertEqual(len(list(target.parent.glob(f".{target.name}.staging-*"))), 1)

    def test_cleanup_never_deletes_target_that_won_link_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "race-cleanup.pt"
            real_link = os.link
            cleanup = OSError("staging unlink failed")

            def racing_link(source: str | bytes, destination: str | bytes) -> None:
                Path(destination).write_bytes(b"winner")
                real_link(source, destination)

            with (
                mock.patch.object(
                    checkpoint_module.os,
                    "link",
                    side_effect=racing_link,
                ),
                mock.patch.object(
                    checkpoint_module.os,
                    "unlink",
                    side_effect=cleanup,
                ),
            ):
                with self.assertRaises(CLEVRCheckpointPublicationError) as caught:
                    self.save(target)

            self.assertFalse(caught.exception.committed)
            self.assertEqual(target.read_bytes(), b"winner")
            self.assertEqual(len(list(target.parent.glob(f".{target.name}.staging-*"))), 1)

    def test_interrupt_after_descriptor_acquisition_closes_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "descriptor-interrupt.pt"
            descriptor: list[int] = []
            real_mkstemp = tempfile.mkstemp

            def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
                acquired, staging = real_mkstemp(*args, **kwargs)
                descriptor.append(acquired)
                return acquired, staging

            with (
                mock.patch.object(
                    checkpoint_module.tempfile,
                    "mkstemp",
                    side_effect=recording_mkstemp,
                ),
                mock.patch.object(
                    checkpoint_module.os,
                    "fchmod",
                    side_effect=KeyboardInterrupt("after descriptor acquisition"),
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    self.save(target)

            try:
                os.fstat(descriptor[0])
            except OSError:
                descriptor_is_closed = True
            else:
                descriptor_is_closed = False
                os.close(descriptor[0])
            self.assertTrue(descriptor_is_closed)
            self.assertFalse(target.exists())
            self.assert_no_staging(target)

    def test_serialization_delegates_short_writes_to_buffered_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "buffered.pt"
            with mock.patch.object(
                checkpoint_module.os,
                "write",
                side_effect=AssertionError("publication reimplemented short writes"),
            ):
                self.save(target)

            payload = torch.load(target, map_location="cpu", weights_only=True)
            self.assertEqual(payload["protocol_version"], "PUB-R1.1")
            self.assert_no_staging(target)


if __name__ == "__main__":
    unittest.main()
