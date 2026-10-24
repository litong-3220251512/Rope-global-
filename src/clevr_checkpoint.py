"""Strict PUB-R1.1 checkpoint snapshots and new-file publication."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import random
import stat
import tempfile

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.clevr_sampling import CLEVRIndexStream


PROTOCOL_VERSION = "PUB-R1.1"
CHECKPOINT_SCHEMA_VERSION = 1
EFFECTIVE_BATCH_SIZE = 128
TOTAL_UPDATES = 8_000

_METHODS = frozenset({"G", "B", "M10", "Q-only"})
_TOP_LEVEL_FIELDS = {
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
}
_RNG_FIELDS = {"python", "numpy", "torch_cpu", "cuda_devices", "torch_cuda"}
_NUMPY_RNG_FIELDS = {
    "bit_generator",
    "keys",
    "pos",
    "has_gauss",
    "cached_gaussian",
}
_ENVIRONMENT_FIELDS = {"python", "numpy", "torch", "cuda_runtime"}
_UINT32_LIMIT = 1 << 32


class CLEVRCheckpointError(ValueError):
    """Raised when the PUB-R1.1 checkpoint contract is violated."""


class CLEVRCheckpointPublicationError(CLEVRCheckpointError):
    """Raised when publication fails after a private staging file exists."""

    def __init__(self, message: str, *, committed: bool) -> None:
        if type(committed) is not bool:
            raise CLEVRCheckpointError(
                "publication error committed must be an ordinary bool"
            )
        super().__init__(message)
        self.committed = committed


@dataclass(frozen=True, slots=True)
class CLEVRCheckpointMetadata:
    path: str
    size_bytes: int
    method: str
    run_seed: int
    completed_updates: int
    dataset_size: int
    cuda_devices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise CLEVRCheckpointError("metadata path must be a non-empty string")
        _require_ordinary_int(self.size_bytes, "metadata size_bytes")
        if self.size_bytes < 0:
            raise CLEVRCheckpointError("metadata size_bytes must be non-negative")
        _validate_method(self.method)
        _validate_run_seed(self.run_seed)
        _validate_completed_updates(self.completed_updates)
        _require_ordinary_int(self.dataset_size, "metadata dataset_size")
        if self.dataset_size <= 0:
            raise CLEVRCheckpointError("metadata dataset_size must be positive")
        _validate_cuda_devices(self.cuda_devices)


def save_clevr_checkpoint(
    path: Path,
    *,
    method: str,
    run_seed: int,
    completed_updates: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> CLEVRCheckpointMetadata:
    """Capture one complete update boundary and publish it without overwrite."""

    _validate_save_arguments(
        path=path,
        method=method,
        run_seed=run_seed,
        completed_updates=completed_updates,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        cuda_devices=cuda_devices,
    )
    payload = _build_payload(
        method=method,
        run_seed=run_seed,
        completed_updates=completed_updates,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        cuda_devices=cuda_devices,
    )
    _validate_payload_snapshot(payload)
    return _publish_new_file(path, payload)


def _validate_save_arguments(
    *,
    path: Path,
    method: str,
    run_seed: int,
    completed_updates: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> None:
    _validate_target(path)
    _validate_method(method)
    _validate_run_seed(run_seed)
    _validate_completed_updates(completed_updates)
    _validate_cuda_devices(cuda_devices)

    if not isinstance(model, nn.Module):
        raise CLEVRCheckpointError("model must be a torch.nn.Module")
    if not isinstance(optimizer, AdamW):
        raise CLEVRCheckpointError("optimizer must be torch.optim.AdamW")
    if not isinstance(scheduler, LambdaLR):
        raise CLEVRCheckpointError("scheduler must be torch.optim.lr_scheduler.LambdaLR")
    if not isinstance(sampler, CLEVRIndexStream):
        raise CLEVRCheckpointError("sampler must be CLEVRIndexStream")

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise CLEVRCheckpointError(
            "model gradients must all be None at a complete update boundary"
        )
    if sampler.run_seed != run_seed:
        raise CLEVRCheckpointError("sampler run_seed disagrees with run_seed")
    sampler_state = sampler.state_dict()
    dataset_size = sampler.dataset_size
    if sampler_state["dataset_size"] != dataset_size:
        raise CLEVRCheckpointError("sampler dataset_size is internally inconsistent")
    consumed = (
        sampler_state["epoch_index"] * dataset_size + sampler_state["cursor"]
    )
    expected_consumed = EFFECTIVE_BATCH_SIZE * completed_updates
    if consumed != expected_consumed:
        raise CLEVRCheckpointError(
            "sampler committed position is not the completed update boundary: "
            f"consumed={consumed}, expected={expected_consumed}"
        )
    sampler_probe = CLEVRIndexStream(dataset_size=dataset_size, run_seed=run_seed)
    try:
        sampler_probe.load_state_dict(sampler_state)
    except ValueError as exc:
        raise CLEVRCheckpointError("sampler state is invalid") from exc

    _validate_optimizer_groups(model, optimizer)
    _validate_scheduler(optimizer, scheduler, completed_updates)
    if cuda_devices and not torch.cuda.is_initialized():
        raise CLEVRCheckpointError(
            "non-empty cuda_devices requires caller-initialized CUDA"
        )


def _validate_target(path: Path) -> None:
    if not isinstance(path, Path):
        raise CLEVRCheckpointError("path must be a pathlib.Path")
    parent = path.parent
    if not parent.exists():
        raise CLEVRCheckpointError("checkpoint parent directory does not exist")
    if not parent.is_dir():
        raise CLEVRCheckpointError("checkpoint parent must be a directory")
    if os.path.lexists(os.fspath(path)):
        raise CLEVRCheckpointError("checkpoint target already exists")


def _validate_method(method: str) -> None:
    if type(method) is not str or method not in _METHODS:
        raise CLEVRCheckpointError(
            "method must be exactly one of G, B, M10, or Q-only"
        )


def _require_ordinary_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise CLEVRCheckpointError(f"{field} must be an ordinary integer")
    return value


def _validate_run_seed(run_seed: int) -> None:
    _require_ordinary_int(run_seed, "run_seed")
    if run_seed < 0:
        raise CLEVRCheckpointError("run_seed must be non-negative")


def _validate_completed_updates(completed_updates: int) -> None:
    _require_ordinary_int(completed_updates, "completed_updates")
    if completed_updates < 0 or completed_updates > TOTAL_UPDATES:
        raise CLEVRCheckpointError(
            f"completed_updates must be between 0 and {TOTAL_UPDATES}"
        )


def _validate_cuda_devices(cuda_devices: tuple[int, ...]) -> None:
    if type(cuda_devices) is not tuple:
        raise CLEVRCheckpointError("cuda_devices must be a tuple")
    if any(type(device) is not int for device in cuda_devices):
        raise CLEVRCheckpointError(
            "cuda_devices must contain ordinary integer device indices"
        )
    if any(device < 0 for device in cuda_devices):
        raise CLEVRCheckpointError("cuda_devices must be non-negative")
    if tuple(sorted(set(cuda_devices))) != cuda_devices:
        raise CLEVRCheckpointError(
            "cuda_devices must be strictly increasing without duplicates"
        )


def _validate_optimizer_groups(model: nn.Module, optimizer: AdamW) -> None:
    model_parameters = dict(model.named_parameters())
    observed_names: list[str] = []
    observed_groups: set[str] = set()
    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        param_names = group.get("param_names")
        if type(group_name) is not str or not group_name:
            raise CLEVRCheckpointError(
                "every optimizer parameter group needs a non-empty group_name"
            )
        if group_name in observed_groups:
            raise CLEVRCheckpointError("optimizer group_name values must be unique")
        observed_groups.add(group_name)
        if type(param_names) is not list or any(
            type(name) is not str for name in param_names
        ):
            raise CLEVRCheckpointError(
                "every optimizer parameter group needs literal param_names"
            )
        parameters = group.get("params")
        if type(parameters) is not list or len(parameters) != len(param_names):
            raise CLEVRCheckpointError(
                "optimizer param_names must match parameter group length"
            )
        for name, parameter in zip(param_names, parameters):
            if name not in model_parameters or model_parameters[name] is not parameter:
                raise CLEVRCheckpointError(
                    "optimizer param_names/order disagree with model parameters"
                )
            observed_names.append(name)
    if len(observed_names) != len(set(observed_names)):
        raise CLEVRCheckpointError("optimizer parameters must not be duplicated")
    if set(observed_names) != set(model_parameters):
        raise CLEVRCheckpointError(
            "optimizer named groups must cover every model parameter exactly once"
        )


def _validate_scheduler(
    optimizer: AdamW,
    scheduler: LambdaLR,
    completed_updates: int,
) -> None:
    if scheduler.optimizer is not optimizer:
        raise CLEVRCheckpointError("scheduler must own the supplied optimizer")
    if type(scheduler.last_epoch) is not int or scheduler.last_epoch != completed_updates:
        raise CLEVRCheckpointError(
            "scheduler last_epoch must equal completed_updates"
        )
    step_count = getattr(scheduler, "_step_count", None)
    if type(step_count) is not int or step_count != completed_updates + 1:
        raise CLEVRCheckpointError(
            "scheduler _step_count must equal completed_updates + 1"
        )
    last_lr = getattr(scheduler, "_last_lr", None)
    current_lr = [group["lr"] for group in optimizer.param_groups]
    if type(last_lr) is not list or last_lr != current_lr:
        raise CLEVRCheckpointError(
            "scheduler _last_lr must equal optimizer current learning rates"
        )


def _build_payload(
    *,
    method: str,
    run_seed: int,
    completed_updates: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> dict[str, object]:
    python_state = _snapshot_value(random.getstate(), "Python RNG state")
    _validate_python_rng_state(python_state)

    numpy_raw = np.random.get_state(legacy=True)
    numpy_state = {
        "bit_generator": numpy_raw[0],
        "keys": torch.from_numpy(numpy_raw[1].astype(np.int64, copy=True)).clone(),
        "pos": int(numpy_raw[2]),
        "has_gauss": int(numpy_raw[3]),
        "cached_gaussian": float(numpy_raw[4]),
    }
    _validate_numpy_rng_state(numpy_state)

    torch_cpu = torch.random.get_rng_state().detach().cpu().clone()
    _validate_torch_cpu_rng_state(torch_cpu)

    torch_cuda: tuple[torch.Tensor, ...]
    if cuda_devices:
        torch_cuda = tuple(
            torch.cuda.get_rng_state(device).detach().cpu().clone()
            for device in cuda_devices
        )
    else:
        torch_cuda = ()

    return {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": method,
        "run_seed": run_seed,
        "completed_updates": completed_updates,
        "dataset_size": sampler.dataset_size,
        "model_state": _snapshot_value(model.state_dict(), "model state"),
        "optimizer_state": _snapshot_value(
            optimizer.state_dict(), "optimizer state"
        ),
        "scheduler_state": _snapshot_value(
            scheduler.state_dict(), "scheduler state"
        ),
        "sampler_state": _snapshot_value(sampler.state_dict(), "sampler state"),
        "rng_state": {
            "python": python_state,
            "numpy": numpy_state,
            "torch_cpu": torch_cpu,
            "cuda_devices": cuda_devices,
            "torch_cuda": torch_cuda,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
        },
    }


def _snapshot_value(value: object, field: str) -> object:
    if type(value) is torch.Tensor:
        return value.detach().cpu().clone()
    if type(value) is OrderedDict:
        return OrderedDict(
            (_snapshot_key(key, field), _snapshot_value(item, field))
            for key, item in value.items()
        )
    if type(value) is dict:
        return {
            _snapshot_key(key, field): _snapshot_value(item, field)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_snapshot_value(item, field) for item in value]
    if type(value) is tuple:
        return tuple(_snapshot_value(item, field) for item in value)
    if type(value) in (type(None), bool, int, float, str):
        return value
    raise CLEVRCheckpointError(
        f"{field} contains unsupported value type {type(value).__name__}"
    )


def _snapshot_key(key: object, field: str) -> object:
    if type(key) in (int, str):
        return key
    raise CLEVRCheckpointError(
        f"{field} contains unsupported mapping key type {type(key).__name__}"
    )


def _validate_python_rng_state(state: object) -> None:
    try:
        probe = random.Random()
        probe.setstate(state)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CLEVRCheckpointError("Python RNG state is invalid") from exc


def _validate_numpy_rng_state(state: Mapping[str, object]) -> None:
    if set(state) != _NUMPY_RNG_FIELDS:
        raise CLEVRCheckpointError("NumPy RNG state fields disagree")
    if state["bit_generator"] != "MT19937":
        raise CLEVRCheckpointError("NumPy RNG state must use MT19937")
    keys = state["keys"]
    if type(keys) is not torch.Tensor:
        raise CLEVRCheckpointError("NumPy RNG keys must be a tensor")
    if keys.device.type != "cpu" or keys.dtype != torch.int64 or keys.shape != (624,):
        raise CLEVRCheckpointError(
            "NumPy RNG keys must be a 624-element CPU int64 tensor"
        )
    if bool(torch.any(keys < 0)) or bool(torch.any(keys >= _UINT32_LIMIT)):
        raise CLEVRCheckpointError("NumPy RNG keys are outside the uint32 range")
    pos = _require_ordinary_int(state["pos"], "NumPy RNG pos")
    has_gauss = _require_ordinary_int(
        state["has_gauss"], "NumPy RNG has_gauss"
    )
    cached_gaussian = state["cached_gaussian"]
    if type(cached_gaussian) is not float:
        raise CLEVRCheckpointError(
            "NumPy RNG cached_gaussian must be an ordinary float"
        )
    try:
        probe = np.random.RandomState()
        probe.set_state(
            (
                "MT19937",
                keys.numpy().astype(np.uint32, copy=True),
                pos,
                has_gauss,
                cached_gaussian,
            )
        )
    except (TypeError, ValueError) as exc:
        raise CLEVRCheckpointError("NumPy RNG state is invalid") from exc


def _validate_torch_cpu_rng_state(state: object) -> None:
    if type(state) is not torch.Tensor:
        raise CLEVRCheckpointError("PyTorch CPU RNG state must be a tensor")
    if state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1:
        raise CLEVRCheckpointError(
            "PyTorch CPU RNG state must be a one-dimensional CPU byte tensor"
        )
    try:
        probe = torch.Generator(device="cpu")
        probe.set_state(state)
    except RuntimeError as exc:
        raise CLEVRCheckpointError("PyTorch CPU RNG state is invalid") from exc


def _validate_payload_snapshot(payload: Mapping[str, object]) -> None:
    if type(payload) is not dict or set(payload) != _TOP_LEVEL_FIELDS:
        raise CLEVRCheckpointError("checkpoint payload fields disagree")
    rng_state = payload["rng_state"]
    environment = payload["environment"]
    if type(rng_state) is not dict or set(rng_state) != _RNG_FIELDS:
        raise CLEVRCheckpointError("checkpoint RNG fields disagree")
    if type(environment) is not dict or set(environment) != _ENVIRONMENT_FIELDS:
        raise CLEVRCheckpointError("checkpoint environment fields disagree")
    _snapshot_value(payload, "checkpoint payload")


def _publish_new_file(
    path: Path,
    payload: Mapping[str, object],
) -> CLEVRCheckpointMetadata:
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=path.parent,
    )
    staging = Path(staging_name)
    descriptor_owned = True
    output: object | None = None
    staging_identity: tuple[int, int] | None = None
    link_started = False
    link_returned = False
    staging_unlink_started = False
    committed = False
    try:
        acquired_stat = os.fstat(descriptor)
        staging_identity = (acquired_stat.st_dev, acquired_stat.st_ino)
        os.fchmod(descriptor, 0o600)
        output = os.fdopen(descriptor, "wb")
        descriptor_owned = False
        with output:
            torch.save(payload, output)
            output.flush()
        staging_stat = staging.stat(follow_symlinks=False)
        if not stat.S_ISREG(staging_stat.st_mode):
            raise CLEVRCheckpointError("checkpoint staging path is not a regular file")
        if (staging_stat.st_dev, staging_stat.st_ino) != staging_identity:
            raise CLEVRCheckpointError("checkpoint staging inode changed before publication")
        link_started = True
        os.link(staging, path)
        link_returned = True
        committed = True
        staging_unlink_started = True
        os.unlink(staging)
        final_stat = path.stat(follow_symlinks=False)
        return CLEVRCheckpointMetadata(
            path=str(path.resolve()),
            size_bytes=final_stat.st_size,
            method=payload["method"],
            run_seed=payload["run_seed"],
            completed_updates=payload["completed_updates"],
            dataset_size=payload["dataset_size"],
            cuda_devices=payload["rng_state"]["cuda_devices"],
        )
    except BaseException as primary:
        cleanup_failures: list[BaseException] = []

        if output is not None and not bool(getattr(output, "closed", False)):
            try:
                output.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)
        elif descriptor_owned:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)

        if (
            not committed
            and link_started
            and not link_returned
            and staging_identity is not None
        ):
            committed = _paths_are_same_regular_inode(
                staging,
                path,
                staging_identity,
            )

        if not staging_unlink_started:
            try:
                _unlink_owned_staging(staging, staging_identity)
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)

        if committed or cleanup_failures:
            status = "committed" if committed else "not committed"
            publication_error = CLEVRCheckpointPublicationError(
                f"checkpoint publication failed ({status})",
                committed=committed,
            )
            for cleanup_error in cleanup_failures:
                publication_error.add_note(
                    "checkpoint staging cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise publication_error from primary
        raise


def _paths_are_same_regular_inode(
    staging: Path,
    target: Path,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        staging_stat = staging.stat(follow_symlinks=False)
        target_stat = target.stat(follow_symlinks=False)
    except OSError:
        return False
    if not stat.S_ISREG(staging_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
        return False
    staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
    target_identity = (target_stat.st_dev, target_stat.st_ino)
    return staging_identity == expected_identity == target_identity


def _unlink_owned_staging(
    staging: Path,
    expected_identity: tuple[int, int] | None,
) -> None:
    try:
        staging_stat = staging.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if expected_identity is None:
        raise CLEVRCheckpointError(
            "cannot prove ownership of checkpoint staging path"
        )
    if not stat.S_ISREG(staging_stat.st_mode) or (
        staging_stat.st_dev,
        staging_stat.st_ino,
    ) != expected_identity:
        raise CLEVRCheckpointError(
            "checkpoint staging path no longer identifies this invocation's file"
        )
    os.unlink(staging)
