"""PUB-R1.1 checkpoint snapshot and publication tests."""

from __future__ import annotations

import copy
from contextlib import ExitStack
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

    def test_one_shot_first_fstat_interrupt_recovers_identity_and_cleans_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "first-fstat-interrupt.pt"
            descriptors: list[int] = []
            staging_paths: list[str] = []
            primary = KeyboardInterrupt("first staging identity inspection")
            real_mkstemp = tempfile.mkstemp
            real_fstat = os.fstat
            fstat_attempts = 0

            def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
                descriptor, staging_name = real_mkstemp(*args, **kwargs)
                descriptors.append(descriptor)
                staging_paths.append(staging_name)
                return descriptor, staging_name

            def interrupt_first_fstat(descriptor: int) -> os.stat_result:
                nonlocal fstat_attempts
                fstat_attempts += 1
                if fstat_attempts == 1:
                    raise primary
                return real_fstat(descriptor)

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
                        "fstat",
                        side_effect=interrupt_first_fstat,
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

            self.assertIs(observed, primary)
            self.assertTrue(descriptor_closed)
            self.assertFalse(staging_exists)
            self.assertFalse(target.exists())

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


class _IntSubclass(int):
    pass


class _StringSubclass(str):
    pass


def _execute_checkpoint_reducer(marker):
    Path(marker).write_text("executed")
    return torch.zeros(2, 3)


class _UnsafeCheckpointValue:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (_execute_checkpoint_reducer, (self.marker,))


class CheckpointLoadTests(unittest.TestCase):
    assert_state_equal = CheckpointSaveTests.assert_state_equal
    assert_numpy_global_state_equal = CheckpointSaveTests.assert_numpy_global_state_equal
    assert_numpy_payload_equal = CheckpointSaveTests.assert_numpy_payload_equal

    def setUp(self) -> None:
        self.fixture = _CheckpointFixture()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "boundary.pt"
        self.metadata = save_clevr_checkpoint(
            self.path, method="G", run_seed=101, completed_updates=1,
            model=self.fixture.model, optimizer=self.fixture.optimizer,
            scheduler=self.fixture.scheduler, sampler=self.fixture.sampler,
            cuda_devices=(),
        )
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)

    def load(self, path=None, **overrides):
        arguments = dict(
            expected_method="G", expected_run_seed=101,
            model=self.fixture.model, optimizer=self.fixture.optimizer,
            scheduler=self.fixture.scheduler, sampler=self.fixture.sampler,
            cuda_devices=(),
        )
        arguments.update(overrides)
        return checkpoint_module.load_clevr_checkpoint(
            self.path if path is None else path, **arguments
        )

    def live_state(self):
        f = self.fixture
        return dict(
            model=_clone_state(f.model.state_dict()),
            optimizer=_clone_state(f.optimizer.state_dict()),
            scheduler=_clone_state(f.scheduler.state_dict()),
            sampler=_clone_state(f.sampler.state_dict()),
            python=random.getstate(), numpy=np.random.get_state(legacy=True),
            torch=torch.random.get_rng_state().clone(),
            gradients=[None if p.grad is None else p.grad.clone()
                       for p in f.model.parameters()],
        )

    def assert_live_equal(self, expected):
        actual = self.live_state()
        self.assert_numpy_global_state_equal(actual.pop("numpy"), expected["numpy"])
        self.assert_state_equal(actual, {k: v for k, v in expected.items() if k != "numpy"})

    def next_observations(self):
        # Private probes observe the next draws without consuming global state.
        python = random.Random(0)
        python.setstate(random.getstate())
        numpy = np.random.RandomState(0)
        numpy.set_state(np.random.get_state(legacy=True))
        cpu = torch.Generator(device="cpu")
        cpu.set_state(torch.random.get_rng_state())
        return (self.fixture.sampler.preview(128), python.random(),
                numpy.random_sample(4), torch.rand(4, generator=cpu))

    def assert_observations_equal(self, expected):
        self.assertEqual(self.fixture.sampler.preview(128), expected[0])
        self.assertEqual(random.random(), expected[1])
        self.assertTrue(np.array_equal(np.random.random_sample(4), expected[2]))
        self.assertTrue(torch.equal(torch.rand(4), expected[3]))

    def perturb(self):
        f = self.fixture
        with torch.no_grad():
            for p in f.model.parameters():
                p.add_(7)
        for state in f.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.add_(3)
        for group in f.optimizer.param_groups:
            group["lr"] = 0.123
        f.scheduler.last_epoch = 4
        f.scheduler._step_count = 5
        f.scheduler._last_lr = [0.123, 0.123]
        f.sampler.commit(f.sampler.preview(128))
        random.random()
        np.random.random_sample(7)
        torch.rand(7)

    def test_restores_all_live_state_and_next_observations_exactly(self):
        expected_next = self.next_observations()
        self.perturb()
        f = self.fixture
        real_load = torch.load
        real_scheduler_load = f.scheduler.load_state_dict
        real_optimizer_load = f.optimizer.load_state_dict

        def restricted_read(path, *, map_location, weights_only):
            self.assertEqual(map_location, "cpu")
            self.assertIs(weights_only, True)
            return real_load(path, map_location=map_location, weights_only=weights_only)

        def scheduler_load(_self, state):
            real_scheduler_load(state)
            # Observable LR side effect requires optimizer restoration afterwards.
            for group in f.optimizer.param_groups:
                group["lr"] = 99.0

        def optimizer_load(_self, state):
            self.assertEqual(f.scheduler.last_epoch, 1)
            real_optimizer_load(state)

        with (
            mock.patch.object(torch, "load", side_effect=restricted_read),
            mock.patch.object(type(f.scheduler), "load_state_dict", new=scheduler_load),
            mock.patch.object(type(f.optimizer), "load_state_dict", new=optimizer_load),
            mock.patch.object(checkpoint_module.torch, "cuda", new=_ForbiddenCUDA()),
        ):
            metadata = self.load()
        self.assertEqual(metadata, self.metadata)
        for component in ("model", "optimizer", "scheduler", "sampler"):
            self.assert_state_equal(
                getattr(f, component).state_dict(), self.payload[component + "_state"]
            )
        self.assertEqual(random.getstate(), self.payload["rng_state"]["python"])
        self.assert_numpy_payload_equal(self.payload["rng_state"]["numpy"], np.random.get_state())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), self.payload["rng_state"]["torch_cpu"]))
        self.assert_observations_equal(expected_next)


    def assert_rejected(self, payload=None, *, path=None, **overrides):
        if payload is not None:
            path = Path(self.directory.name) / "malformed.pt"
            torch.save(payload, path)
        before = self.live_state()
        next_values = self.next_observations()
        targets = {name: overrides.get(name, getattr(self.fixture, name))
                   for name in ("model", "optimizer", "scheduler", "sampler")}
        target_states = {name: _clone_state(obj.state_dict()) for name, obj in targets.items()}
        applications = []
        real_model_load = type(targets["model"]).load_state_dict

        def observe_application(instance, *args, **kwargs):
            applications.append(True)
            return real_model_load(instance, *args, **kwargs)

        try:
            with mock.patch.object(type(targets["model"]), "load_state_dict", new=observe_application):
                with self.assertRaises(CLEVRCheckpointError):
                    self.load(path, **overrides)
        finally:
            self.assert_live_equal(before)
            for name, obj in targets.items():
                self.assert_state_equal(obj.state_dict(), target_states[name])
            self.assert_observations_equal(next_values)
            self.assertEqual(applications, [], "invalid payload reached live application")

    def test_rejects_absent_nonfile_and_nonpath_input(self):
        for path in (Path(self.directory.name) / "absent", Path(self.directory.name),
                     str(self.path)):
            with self.subTest(path=path):
                self.assert_rejected(path=path)
        self.assert_rejected(path=Path(self.directory.name) / "dangling")

    def test_rejects_custom_reducer_without_executing_it(self):
        marker = Path(self.directory.name) / "reducer-executed"
        payload = copy.deepcopy(self.payload)
        payload["model_state"]["0.weight"] = _UnsafeCheckpointValue(str(marker))
        try:
            self.assert_rejected(payload)
        finally:
            self.assertFalse(marker.exists(), "unrestricted reducer executed")

    def test_rejects_missing_and_extra_required_nested_mapping_keys(self):
        def mappings(value, path=()):
            if isinstance(value, dict):
                yield path, value
                for key, item in value.items():
                    yield from mappings(item, path + (key,))
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    yield from mappings(item, path + (index,))

        def at_path(payload, path):
            for key in path:
                payload = payload[key]
            return payload

        for path, mapping in list(mappings(self.payload)):
            for key in mapping:
                if path == ("optimizer_state", "state"):
                    # Parameter IDs are optional lazy state entries; each present
                    # entry still requires all AdamW fields checked below.
                    continue
                with self.subTest(path=path, missing=key):
                    malformed = copy.deepcopy(self.payload)
                    del at_path(malformed, path)[key]
                    self.assert_rejected(malformed)
            with self.subTest(path=path, extra=True):
                malformed = copy.deepcopy(self.payload)
                at_path(malformed, path)["unexpected"] = 0
                self.assert_rejected(malformed)

    def test_rejects_target_gradients_before_mutation(self):
        next(self.fixture.model.parameters()).grad = torch.ones_like(
            next(self.fixture.model.parameters()))
        self.assert_rejected()

    def test_rejects_protocol_environment_and_run_identity(self):
        cases = [(("protocol_version",), "old"),
                 (("checkpoint_schema_version",), 2),
                 (("method",), "B"), (("run_seed",), 102),
                 (("dataset_size",), 6), (("completed_updates",), -1),
                 (("completed_updates",), 8001)]
        for key in self.payload["environment"]:
            cases.append((("environment", key), "different"))
        for path, value in cases:
            with self.subTest(path=path, value=value):
                malformed = copy.deepcopy(self.payload)
                target = malformed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assert_rejected(malformed)
        for overrides in (dict(expected_method="B"), dict(expected_run_seed=102),
                          dict(expected_method=_StringSubclass("G")),
                          dict(expected_run_seed=True), dict(expected_run_seed=np.int64(101)),
                          dict(expected_run_seed=_IntSubclass(101)),
                          dict(expected_run_seed=101.0)):
            with self.subTest(overrides=overrides):
                self.assert_rejected(**overrides)

    def test_rejects_nonordinary_payload_scalars(self):
        integer_paths = [(key,) for key in ("checkpoint_schema_version", "run_seed",
                         "completed_updates", "dataset_size")]
        integer_paths += [("scheduler_state", key) for key in ("last_epoch", "_step_count")]
        integer_paths += [("sampler_state", key) for key in ("dataset_size", "run_seed",
                          "sampler_seed", "epoch_index", "cursor")]
        integer_paths += [("rng_state", "numpy", key) for key in ("pos", "has_gauss")]
        for path in integer_paths:
            original = self.payload
            for key in path:
                original = original[key]
            for value in (True, np.int64(original), _IntSubclass(original), float(original)):
                with self.subTest(path=path, value_type=type(value)):
                    malformed = copy.deepcopy(self.payload)
                    target = malformed
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    self.assert_rejected(malformed)
        for path in (("method",), ("protocol_version",), ("environment", "python")):
            malformed = copy.deepcopy(self.payload)
            target = malformed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = _StringSubclass(target[path[-1]])
            with self.subTest(path=path):
                self.assert_rejected(malformed)

    def test_rejects_model_key_shape_dtype_and_nontensor(self):
        for value in (torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.float64), "tensor"):
            with self.subTest(value=value):
                malformed = copy.deepcopy(self.payload)
                malformed["model_state"]["0.weight"] = value
                self.assert_rejected(malformed)

    def test_rejects_optimizer_group_identity_and_state_corruption(self):
        mutations = [
            lambda p: p["optimizer_state"]["param_groups"].pop(),
            lambda p: p["optimizer_state"]["param_groups"].reverse(),
            lambda p: p["optimizer_state"]["param_groups"][0].update(group_name="other"),
            lambda p: p["optimizer_state"]["param_groups"][0].update(param_names=["0.bias"]),
            lambda p: p["optimizer_state"]["param_groups"][0].update(params=[1]),
            lambda p: p["optimizer_state"]["state"][0].update(exp_avg=torch.zeros(1)),
            lambda p: p["optimizer_state"]["state"][0].update(exp_avg=torch.zeros(2, 3, dtype=torch.float64)),
            lambda p: p["optimizer_state"]["state"][0].update(step=1),
            lambda p: p["optimizer_state"]["state"].update({99: {}}),
            lambda p: p["optimizer_state"]["param_groups"][0].update(lr=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                malformed = copy.deepcopy(self.payload)
                mutate(malformed)
                self.assert_rejected(malformed)
        # Same-shaped parameters cannot hide a name/order cross-splice.
        model = nn.Linear(2, 2, bias=False)
        model.register_parameter("other", nn.Parameter(torch.zeros(2, 2)))
        named = list(model.named_parameters())
        optimizer = AdamW([dict(params=[p for _, p in named], group_name="same",
                               param_names=[n for n, _ in named])])
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        path = Path(self.directory.name) / "same-shape.pt"
        save_clevr_checkpoint(path, method="G", run_seed=101, completed_updates=0,
                              model=model, optimizer=optimizer, scheduler=scheduler,
                              sampler=CLEVRIndexStream(dataset_size=5, run_seed=101), cuda_devices=())
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["optimizer_state"]["param_groups"][0]["param_names"].reverse()
        self.assert_rejected(payload, model=model, optimizer=optimizer, scheduler=scheduler)

    def test_rejects_scheduler_cross_splice_and_optimizer_link(self):
        for field, value in (("last_epoch", 2), ("_step_count", 1),
                             ("_last_lr", [0.5, 0.5]), ("base_lrs", [0.01]),
                             ("lr_lambdas", [None])):
            with self.subTest(field=field):
                malformed = copy.deepcopy(self.payload)
                malformed["scheduler_state"][field] = value
                self.assert_rejected(malformed)
        self.fixture.scheduler.optimizer = AdamW(self.fixture.model.parameters())
        self.assert_rejected()

    def test_rejects_sampler_internal_invalidity_and_wrong_consumed_count(self):
        for field, value in (("cursor", 6), ("permutation", (0, 0, 1, 2, 3)),
                             ("cursor", 2), ("permutation", [0, 1, 2, 3, 4])):
            with self.subTest(field=field, value=value):
                malformed = copy.deepcopy(self.payload)
                malformed["sampler_state"][field] = value
                self.assert_rejected(malformed)


    def test_rejects_malformed_python_rng_without_touching_live_state(self):
        original = self.payload["rng_state"]["python"]
        invalid_states = [None, (3, (1, 2), None), list(original),
                          (True, original[1], None), (2, original[1], None),
                          (3, list(original[1]), None),
                          (3, (True,) + original[1][1:], None),
                          (3, (-1,) + original[1][1:], None),
                          (3, original[1], True), (3, original[1], float("nan"))]
        for state in invalid_states:
            with self.subTest(state_type=type(state), state_head=str(state)[:60]):
                malformed = copy.deepcopy(self.payload)
                malformed["rng_state"]["python"] = state
                self.assert_rejected(malformed)

    def test_rejects_malformed_numpy_rng_without_touching_live_state(self):
        cases = [("bit_generator", "PCG64"), ("keys", torch.zeros(623, dtype=torch.int64)),
                 ("keys", torch.zeros(624, dtype=torch.int32)),
                 ("keys", torch.full((624,), -1, dtype=torch.int64)),
                 ("keys", torch.full((624,), 2**32, dtype=torch.int64)),
                 ("keys", torch.zeros(1, 624, dtype=torch.int64)),
                 ("pos", -1), ("pos", 625), ("has_gauss", -1), ("has_gauss", 2),
                 ("cached_gaussian", 0), ("cached_gaussian", True),
                 ("cached_gaussian", np.float64(0.0)),
                 ("cached_gaussian", float("inf")), ("cached_gaussian", float("nan"))]
        for key, value in cases:
            with self.subTest(key=key, value=str(value)[:70]):
                malformed = copy.deepcopy(self.payload)
                malformed["rng_state"]["numpy"][key] = value
                self.assert_rejected(malformed)

    def test_rejects_malformed_cpu_rng_without_touching_live_state(self):
        for value in (torch.zeros(4), torch.zeros(1, 4, dtype=torch.uint8),
                      torch.zeros(4, dtype=torch.uint8), [1, 2],
                      self.payload["rng_state"]["torch_cpu"][:-8].clone()):
            with self.subTest(value=value):
                malformed = copy.deepcopy(self.payload)
                malformed["rng_state"]["torch_cpu"] = value
                self.assert_rejected(malformed)

    def test_rejects_cuda_tuple_and_state_count_order_type(self):
        cases = [("cuda_devices", []), ("cuda_devices", (True,)),
                 ("cuda_devices", (np.int64(0),)), ("cuda_devices", (-1,)),
                 ("cuda_devices", (0, 0)), ("cuda_devices", (2, 0)),
                 ("cuda_devices", (0,)), ("torch_cuda", []),
                 ("torch_cuda", (torch.zeros(4, dtype=torch.uint8),))]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                malformed = copy.deepcopy(self.payload)
                malformed["rng_state"][key] = value
                with mock.patch.object(checkpoint_module.torch, "cuda", new=_ForbiddenCUDA()):
                    self.assert_rejected(malformed)
        for devices in ([], (True,), (np.int64(0),), (_IntSubclass(0),),
                        (0.0,), (-1,), (0, 0), (2, 0)):
            with self.subTest(caller_devices=devices):
                self.assert_rejected(cuda_devices=devices)

    def test_explicit_cuda_rng_with_local_cpu_adapter_and_private_validation(self):
        adapter = _FakeCUDARNG()
        malformed_path = Path(self.directory.name) / "bad-cuda.pt"
        expected = tuple(state.clone() for state in adapter.states.values())
        payload = copy.deepcopy(self.payload)
        payload["rng_state"].update(cuda_devices=(0, 2), torch_cuda=expected)
        path = Path(self.directory.name) / "fake-cuda.pt"
        torch.save(payload, path)
        with adapter:
            for value in (torch.zeros(4), torch.zeros(1, 4, dtype=torch.uint8),
                          torch.zeros(4, dtype=torch.uint8)):
                with self.subTest(cuda_state=value):
                    malformed = copy.deepcopy(payload)
                    malformed["rng_state"]["torch_cuda"] = (value, expected[1])
                    torch.save(malformed, malformed_path)
                    before = tuple(state.clone() for state in adapter.states.values())
                    self.assert_rejected(path=malformed_path, cuda_devices=(0, 2))
                    for actual, entry in zip(adapter.states.values(), before):
                        self.assertTrue(torch.equal(actual, entry))
            adapter.advance()
            metadata = self.load(path, cuda_devices=(0, 2))
            self.assertEqual(metadata.cuda_devices, (0, 2))
            for actual, target in zip(adapter.states.values(), expected):
                self.assertTrue(torch.equal(actual, target))


    def reset_live(self, state):
        f = self.fixture
        f.model.load_state_dict(state["model"])
        f.scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        f.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        f.sampler.load_state_dict(state["sampler"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"])

    def fault_target(self, boundary):
        f = self.fixture
        if boundary in ("model", "scheduler", "optimizer", "sampler"):
            instance = getattr(f, boundary)
            return type(instance), "load_state_dict", getattr(type(instance), "load_state_dict")
        owner = {"python": random, "numpy": np.random,
                 "cpu": torch.random, "cuda": torch.cuda}[boundary]
        name = {"python": "setstate", "numpy": "set_state",
                "cpu": "set_rng_state", "cuda": "set_rng_state"}[boundary]
        return owner, name, getattr(owner, name)

    def test_application_faults_restore_all_entry_state_for_exception_and_interrupt(self):
        for boundary in ("model", "scheduler", "optimizer", "sampler", "python", "numpy", "cpu", "cuda"):
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(boundary=boundary, primary=error_type.__name__):
                    adapter = _FakeCUDARNG()
                    payload = copy.deepcopy(self.payload)
                    payload["rng_state"].update(cuda_devices=(0, 2),
                                              torch_cuda=tuple(adapter.states.values()))
                    path = Path(self.directory.name) / "fault.pt"
                    torch.save(payload, path)
                    self.perturb()
                    adapter.advance()
                    before = self.live_state()
                    next_values = self.next_observations()
                    cuda_before = {k: v.clone() for k, v in adapter.states.items()}
                    cuda_next = {}
                    for device, state in cuda_before.items():
                        probe = adapter.generator_type(device="cpu")
                        probe.set_state(state)
                        cuda_next[device] = torch.rand(4, generator=probe)
                    primary = error_type(f"{boundary} apply failure after mutation")
                    observed = None
                    calls = []
                    try:
                        with adapter:
                            owner, name, real = self.fault_target(boundary)

                            def mutate_then_fail(*args, **kwargs):
                                result = real(*args, **kwargs)
                                if boundary == "sampler" and args[0] is not self.fixture.sampler:
                                    return result  # The preflight probe is not the live boundary.
                                calls.append(True)
                                if len(calls) == 1:
                                    raise primary
                                return result

                            with mock.patch.object(owner, name, new=mutate_then_fail):
                                try:
                                    self.load(path, cuda_devices=(0, 2))
                                except BaseException as error:
                                    observed = error
                        self.assertIs(observed, primary)
                        self.assert_live_equal(before)
                        self.assert_observations_equal(next_values)
                        for device, state in cuda_before.items():
                            self.assertTrue(torch.equal(adapter.states[device], state))
                            probe = adapter.generator_type(device="cpu")
                            probe.set_state(adapter.states[device])
                            self.assertTrue(torch.equal(torch.rand(4, generator=probe), cuda_next[device]))
                    finally:
                        self.reset_live(before)

    def test_rollback_failures_preserve_primary_and_continue_other_restores(self):
        self.perturb()
        before = self.live_state()
        next_values = self.next_observations()
        primary = KeyboardInterrupt("CPU apply interrupted after mutation")
        cleanup = RuntimeError("model rollback failed")
        cpu_cleanup = KeyboardInterrupt("CPU rollback interrupted after mutation")
        real_cpu = torch.random.set_rng_state
        real_model = type(self.fixture.model).load_state_dict
        cpu_calls = []
        model_calls = []

        def cpu_apply(state):
            real_cpu(state)
            cpu_calls.append(True)
            if len(cpu_calls) == 1:
                raise primary
            raise cpu_cleanup

        def model_apply(instance, state, *args, **kwargs):
            model_calls.append(True)
            if len(model_calls) == 2:
                raise cleanup
            return real_model(instance, state, *args, **kwargs)

        observed = None
        try:
            with (mock.patch.object(torch.random, "set_rng_state", new=cpu_apply),
                  mock.patch.object(type(self.fixture.model), "load_state_dict", new=model_apply)):
                try:
                    self.load()
                except BaseException as error:
                    observed = error
            self.assertIsInstance(observed, CLEVRCheckpointError)
            self.assertIs(observed.__cause__, primary)
            self.assertTrue(any("rollback" in note and "model" in note and str(cleanup) in note
                                for note in observed.__notes__))
            self.assertTrue(any("rollback" in note and "CPU" in note and str(cpu_cleanup) in note
                                for note in observed.__notes__))
            # The early CPU cleanup failure must not prevent the remaining actions.
            for field in ("optimizer", "scheduler", "sampler", "python", "torch"):
                self.assert_state_equal(self.live_state()[field], before[field])
            self.assert_numpy_global_state_equal(np.random.get_state(), before["numpy"])
            self.assert_observations_equal(next_values)
            self.assertFalse(torch.equal(self.fixture.model.state_dict()["0.weight"], before["model"]["0.weight"]))
        finally:
            self.reset_live(before)

    def test_silent_application_corruption_is_detected_and_rolled_back(self):
        self.perturb()
        before = self.live_state()
        next_values = self.next_observations()
        real_model = type(self.fixture.model).load_state_dict
        calls = []

        def corrupt_once(instance, state, *args, **kwargs):
            result = real_model(instance, state, *args, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                with torch.no_grad():
                    next(instance.parameters()).add_(1)
            return result

        try:
            with mock.patch.object(type(self.fixture.model), "load_state_dict", new=corrupt_once):
                with self.assertRaises(CLEVRCheckpointError):
                    self.load()
            self.assert_live_equal(before)
            self.assert_observations_equal(next_values)
        finally:
            self.reset_live(before)

    def test_silent_rollback_corruption_is_reported_instead_of_reraising_primary(self):
        self.perturb()
        before = self.live_state()
        primary = RuntimeError("apply failed")
        real_model = type(self.fixture.model).load_state_dict
        calls = []

        def corrupt_rollback(instance, state, *args, **kwargs):
            result = real_model(instance, state, *args, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                raise primary
            with torch.no_grad():
                next(instance.parameters()).add_(1)
            return result

        observed = None
        try:
            with mock.patch.object(type(self.fixture.model), "load_state_dict", new=corrupt_rollback):
                try:
                    self.load()
                except BaseException as error:
                    observed = error
            self.assertIsInstance(observed, CLEVRCheckpointError)
            self.assertIs(observed.__cause__, primary)
            self.assertTrue(any("rollback" in note for note in observed.__notes__))
        finally:
            self.reset_live(before)


    def test_rejects_model_container_type_and_scheduler_base_lr_before_mutation(self):
        malformed = copy.deepcopy(self.payload)
        malformed["model_state"] = dict(malformed["model_state"])
        self.assert_rejected(malformed)
        malformed = copy.deepcopy(self.payload)
        malformed["scheduler_state"]["base_lrs"] = [0.8, 0.9]
        self.assert_rejected(malformed)

    def test_restores_into_fresh_objects_and_preserves_gaussian_caches(self):
        random.gauss(0, 1)
        np.random.standard_normal()
        path = Path(self.directory.name) / "gaussian.pt"
        expected = self.live_state()
        metadata = save_clevr_checkpoint(
            path, method="G", run_seed=101, completed_updates=1,
            model=self.fixture.model, optimizer=self.fixture.optimizer,
            scheduler=self.fixture.scheduler, sampler=self.fixture.sampler, cuda_devices=(),
        )
        expected_next = self.next_observations()
        model = nn.Sequential(nn.Linear(3, 2))
        named = dict(model.named_parameters())
        optimizer = AdamW([
            dict(params=[named["0.weight"]], group_name="decay", param_names=["0.weight"], lr=0.01),
            dict(params=[named["0.bias"]], group_name="no_decay", param_names=["0.bias"], lr=0.02, weight_decay=0.0),
        ])
        scheduler = LambdaLR(optimizer, (lambda _: 1.0, lambda _: 1.0))
        sampler = CLEVRIndexStream(dataset_size=5, run_seed=101)
        self.assertEqual(optimizer.state_dict()["state"], {})
        self.assertEqual(self.load(path, model=model, optimizer=optimizer,
                                   scheduler=scheduler, sampler=sampler), metadata)
        for component, live in (("model", model), ("optimizer", optimizer),
                                ("scheduler", scheduler), ("sampler", sampler)):
            self.assert_state_equal(live.state_dict(), expected[component])
        self.assert_observations_equal(expected_next)
        self.assertEqual(random.gauss(0, 1), expected["python"][2])
        self.assertEqual(np.random.standard_normal(), expected["numpy"][4])


    def assert_nonzero_optimizer_subset_restore(self, saved_ids, target_ids):
        f = self.fixture
        full_state = _clone_state(f.optimizer.state_dict())
        self.assertEqual(set(full_state["state"]), {0, 1})
        saved_state = _clone_state(full_state)
        saved_state["state"] = {key: saved_state["state"][key] for key in saved_ids}
        f.optimizer.load_state_dict(saved_state)
        expected = self.live_state()
        expected_next = self.next_observations()
        path = Path(self.directory.name) / "nonzero-subset.pt"
        metadata = save_clevr_checkpoint(
            path, method="G", run_seed=101, completed_updates=1,
            model=f.model, optimizer=f.optimizer, scheduler=f.scheduler,
            sampler=f.sampler, cuda_devices=(),
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["completed_updates"], 1)
        self.assertEqual(set(payload["optimizer_state"]["state"]), set(saved_ids))

        target_state = _clone_state(full_state)
        target_state["state"] = {key: target_state["state"][key] for key in target_ids}
        f.optimizer.load_state_dict(target_state)
        self.perturb()
        self.assertEqual(set(f.optimizer.state_dict()["state"]), set(target_ids))
        with mock.patch.object(checkpoint_module.torch, "cuda", new=_ForbiddenCUDA()):
            self.assertEqual(self.load(path), metadata)
        self.assertEqual(set(f.optimizer.state_dict()["state"]), set(saved_ids))
        self.assert_state_equal(f.optimizer.state_dict(), payload["optimizer_state"])
        self.assert_live_equal(expected)
        self.assert_observations_equal(expected_next)

    def test_restores_nonzero_partial_optimizer_state_over_fully_materialized_target(self):
        self.assert_nonzero_optimizer_subset_restore(saved_ids=(0,), target_ids=(0, 1))

    def test_restores_nonzero_full_optimizer_state_over_partially_materialized_target(self):
        self.assert_nonzero_optimizer_subset_restore(saved_ids=(0, 1), target_ids=(0,))

    def test_restores_zero_update_checkpoint_over_initialized_live_optimizer(self):
        f = self.fixture
        optimizer = AdamW([dict(group) for group in f.optimizer.param_groups])
        scheduler = LambdaLR(optimizer, (lambda _: 1.0, lambda _: 1.0))
        sampler = CLEVRIndexStream(dataset_size=5, run_seed=101)
        path = Path(self.directory.name) / "zero.pt"
        metadata = save_clevr_checkpoint(
            path, method="G", run_seed=101, completed_updates=0,
            model=f.model, optimizer=optimizer, scheduler=scheduler,
            sampler=sampler, cuda_devices=(),
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertNotEqual(f.optimizer.state_dict()["state"], {})
        self.assertEqual(self.load(path), metadata)
        for component in ("model", "optimizer", "scheduler", "sampler"):
            self.assert_state_equal(getattr(f, component).state_dict(), payload[component + "_state"])


class _ForbiddenCUDA:
    def __getattr__(self, name):
        raise AssertionError(f"empty cuda_devices touched CUDA: {name}")


class _FakeCUDARNG:
    """Local CPU generators stand in only for explicit CUDA RNG APIs."""
    def __init__(self):
        self.generator_type = torch.Generator
        self.states = {
            device: self.generator_type(device="cpu").manual_seed(90 + device).get_state()
            for device in (0, 2)
        }
        self.stack = ExitStack()

    def get_rng_state(self, device):
        return self.states[device].clone()

    def set_rng_state(self, state, device):
        probe = self.generator_type(device="cpu")
        probe.set_state(state)
        self.states[device] = probe.get_state()

    def generator(self, device="cpu"):
        if str(device).startswith("cuda:"):
            return self.generator_type(device="cpu")
        return self.generator_type(device=device)

    def advance(self):
        for device, state in self.states.items():
            probe = self.generator_type(device="cpu")
            probe.set_state(state)
            torch.rand(4, generator=probe)
            self.states[device] = probe.get_state()

    def __enter__(self):
        self.stack.enter_context(mock.patch.object(torch.cuda, "is_initialized", return_value=True))
        self.stack.enter_context(mock.patch.object(torch.cuda, "get_rng_state", new=self.get_rng_state))
        self.stack.enter_context(mock.patch.object(torch.cuda, "set_rng_state", new=self.set_rng_state))
        self.stack.enter_context(mock.patch.object(torch, "Generator", new=self.generator))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


if __name__ == "__main__":
    unittest.main()
