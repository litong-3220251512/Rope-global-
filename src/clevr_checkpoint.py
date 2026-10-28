"""Strict PUB-R1.1 checkpoint snapshots and new-file publication."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import os
import math
from pathlib import Path
import platform
import pickle
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


def load_clevr_checkpoint(
    path: Path,
    *,
    expected_method: str,
    expected_run_seed: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> CLEVRCheckpointMetadata:
    """Preflight an explicit weights-only file, then restore or prove rollback.

    Objects must already be constructed and gradients cleared. Empty device
    tuples make no CUDA call; nonempty tuples require caller-initialized CUDA.
    """
    if not isinstance(path, Path):
        raise CLEVRCheckpointError("path must be a pathlib.Path")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise CLEVRCheckpointError("checkpoint must resolve to a regular file")
    except OSError as exc:
        raise CLEVRCheckpointError("checkpoint file cannot be resolved") from exc
    _validate_method(expected_method)
    _validate_run_seed(expected_run_seed)
    _validate_cuda_devices(cuda_devices)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        raise CLEVRCheckpointError("restricted checkpoint read failed") from exc
    _validate_payload_snapshot(payload)
    _validate_load_payload(
        payload, expected_method, expected_run_seed, model, optimizer,
        scheduler, sampler, cuda_devices,
    )
    # The canonical payload remains private; load hooks receive independent copies.
    canonical = _snapshot_value(payload, "checkpoint")
    metadata = CLEVRCheckpointMetadata(
        path=str(resolved), size_bytes=resolved.stat().st_size,
        method=payload["method"], run_seed=payload["run_seed"],
        completed_updates=payload["completed_updates"],
        dataset_size=payload["dataset_size"], cuda_devices=cuda_devices,
    )
    entry = _capture_load_state(model, optimizer, scheduler, sampler, cuda_devices)
    try:
        _apply_load_state(canonical, model, optimizer, scheduler, sampler, cuda_devices)
        _verify_load_state(canonical, model, optimizer, scheduler, sampler, cuda_devices)
    except BaseException as primary:
        failures = _rollback_load_state(entry, model, optimizer, scheduler, sampler, cuda_devices)
        if failures:
            error = CLEVRCheckpointError("checkpoint application failed; rollback incomplete")
            for component, failure in failures:
                error.add_note(f"rollback {component}: {type(failure).__name__}: {failure}")
            raise error from primary
        raise
    return metadata


def _validate_load_payload(
    payload: dict, expected_method: str, expected_run_seed: int, model: nn.Module,
    optimizer: AdamW, scheduler: LambdaLR, sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> None:
    if type(payload["protocol_version"]) is not str or payload["protocol_version"] != PROTOCOL_VERSION:
        raise CLEVRCheckpointError("checkpoint protocol disagrees")
    if _require_ordinary_int(payload["checkpoint_schema_version"], "schema") != CHECKPOINT_SCHEMA_VERSION:
        raise CLEVRCheckpointError("checkpoint schema disagrees")
    _validate_method(payload["method"])
    _validate_run_seed(payload["run_seed"])
    _validate_completed_updates(payload["completed_updates"])
    if (payload["method"], payload["run_seed"]) != (expected_method, expected_run_seed):
        raise CLEVRCheckpointError("checkpoint run identity disagrees")
    if payload["environment"] != {
        "python": platform.python_version(), "numpy": str(np.__version__),
        "torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda),
    }:
        raise CLEVRCheckpointError("checkpoint environment disagrees")
    if not isinstance(model, nn.Module) or not isinstance(optimizer, AdamW):
        raise CLEVRCheckpointError("load requires nn.Module and AdamW")
    if not isinstance(scheduler, LambdaLR) or scheduler.optimizer is not optimizer:
        raise CLEVRCheckpointError("load requires LambdaLR owning supplied optimizer")
    if not isinstance(sampler, CLEVRIndexStream):
        raise CLEVRCheckpointError("load requires CLEVRIndexStream")
    if any(p.grad is not None for p in model.parameters()):
        raise CLEVRCheckpointError("model gradients must all be None")
    _validate_optimizer_groups(model, optimizer)
    model_state = payload["model_state"]
    live_model = model.state_dict()
    _validate_state_schema(model_state, live_model, "model state")
    k = payload["completed_updates"]
    saved_groups = _validate_load_optimizer(
        payload, model, optimizer, k,
    )
    saved_scheduler = payload["scheduler_state"]
    _validate_state_schema(saved_scheduler, scheduler.state_dict(), "scheduler")
    if saved_scheduler["last_epoch"] != k or saved_scheduler["_step_count"] != k + 1:
        raise CLEVRCheckpointError("scheduler counters disagree with update boundary")
    expected_lrs = _scheduler_learning_rates_at_boundary(scheduler, k)
    _require_state_equal(
        [group["lr"] for group in saved_groups], expected_lrs,
        "optimizer learning rates at update boundary",
    )
    _require_state_equal(
        saved_scheduler["_last_lr"], expected_lrs,
        "scheduler learning rates at update boundary",
    )
    _require_state_equal(
        saved_scheduler["base_lrs"], [group["initial_lr"] for group in saved_groups],
        "scheduler base learning rates",
    )
    _require_state_equal(
        saved_scheduler["base_lrs"], scheduler.state_dict()["base_lrs"],
        "scheduler protocol base learning rates",
    )
    dataset_size = _require_ordinary_int(payload["dataset_size"], "dataset_size")
    if dataset_size != sampler.dataset_size or expected_run_seed != sampler.run_seed:
        raise CLEVRCheckpointError("sampler constructor identity disagrees")
    probe = CLEVRIndexStream(dataset_size=dataset_size, run_seed=expected_run_seed)
    _validate_state_schema(payload["sampler_state"], sampler.state_dict(), "sampler state")
    try:
        probe.load_state_dict(payload["sampler_state"])
    except ValueError as exc:
        raise CLEVRCheckpointError("invalid sampler checkpoint") from exc
    state = probe.state_dict()
    if state["epoch_index"] * dataset_size + state["cursor"] != EFFECTIVE_BATCH_SIZE * k:
        raise CLEVRCheckpointError("sampler consumed count disagrees with update boundary")
    rng = payload["rng_state"]
    _validate_load_rng(rng)
    _validate_torch_cpu_rng_state(rng["torch_cpu"])
    cpu_probe = torch.Generator(device="cpu")
    cpu_probe.set_state(rng["torch_cpu"])
    _require_state_equal(cpu_probe.get_state(), rng["torch_cpu"], "CPU private RNG probe")
    _validate_cuda_devices(rng["cuda_devices"])
    if rng["cuda_devices"] != cuda_devices or len(rng["torch_cuda"]) != len(cuda_devices):
        raise CLEVRCheckpointError("CUDA RNG devices/states disagree")
    if cuda_devices and not torch.cuda.is_initialized():
        raise CLEVRCheckpointError("non-empty cuda_devices requires caller-initialized CUDA")
    for device, state in zip(cuda_devices, rng["torch_cuda"]):
        if (type(state) is not torch.Tensor or state.device.type != "cpu"
                or state.dtype != torch.uint8 or state.ndim != 1):
            raise CLEVRCheckpointError("CUDA RNG state must be a CPU byte vector")
        try:
            probe = torch.Generator(device=f"cuda:{device}")
            probe.set_state(state)
            _require_state_equal(probe.get_state(), state, "CUDA private RNG probe")
        except RuntimeError as exc:
            raise CLEVRCheckpointError("CUDA RNG state is invalid") from exc


def _validate_load_rng(rng: dict) -> None:
    python = rng["python"]
    if type(python) is not tuple or len(python) != 3:
        raise CLEVRCheckpointError("Python RNG state must be a three-element tuple")
    _validate_state_schema(python[:2], random.Random(0).getstate()[:2], "Python RNG")
    if python[2] is not None and (type(python[2]) is not float or not math.isfinite(python[2])):
        raise CLEVRCheckpointError("Python RNG cached Gaussian must be None or finite float")
    if python[0] != 3:
        raise CLEVRCheckpointError("Python RNG version disagrees")
    if any(not 0 <= word < _UINT32_LIMIT for word in python[1][:-1]):
        raise CLEVRCheckpointError("Python RNG words outside uint32 range")
    if not 0 <= python[1][-1] <= 624:
        raise CLEVRCheckpointError("Python RNG position outside MT19937 range")
    # getstate() may legitimately contain a cached Gaussian instead of None.
    _validate_python_rng_state(python)
    numpy = rng["numpy"]
    if type(numpy) is not dict or set(numpy) != _NUMPY_RNG_FIELDS:
        raise CLEVRCheckpointError("NumPy RNG fields disagree")
    if type(numpy["bit_generator"]) is not str:
        raise CLEVRCheckpointError("NumPy bit generator name must be an ordinary string")
    pos = _require_ordinary_int(numpy["pos"], "NumPy pos")
    flag = _require_ordinary_int(numpy["has_gauss"], "NumPy has_gauss")
    if not 0 <= pos <= 624 or flag not in (0, 1):
        raise CLEVRCheckpointError("NumPy RNG position/cache flag invalid")
    if type(numpy["cached_gaussian"]) is not float or not math.isfinite(numpy["cached_gaussian"]):
        raise CLEVRCheckpointError("NumPy RNG cached Gaussian must be a finite float")
    _validate_numpy_rng_state(numpy)
    if type(rng["torch_cuda"]) is not tuple:
        raise CLEVRCheckpointError("CUDA RNG states must be a tuple")


def _validate_state_schema(value: object, reference: object, field: str) -> None:
    """Check exact containers/scalars and tensor contracts, without coercion."""
    if type(value) is not type(reference):
        raise CLEVRCheckpointError(f"{field}: invalid type")
    if type(reference) is torch.Tensor:
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise CLEVRCheckpointError(f"{field}: invalid tensor shape/dtype")
    elif isinstance(reference, Mapping):
        if set(value) != set(reference):
            raise CLEVRCheckpointError(f"{field}: invalid fields")
        if type(reference) is OrderedDict:
            _validate_state_schema(vars(value), vars(reference), f"{field} attributes")
        for key in reference:
            _validate_state_schema(value[key], reference[key], f"{field}.{key}")
    elif type(reference) in (list, tuple):
        if len(value) != len(reference):
            raise CLEVRCheckpointError(f"{field}: invalid length")
        for index, (item, target) in enumerate(zip(value, reference)):
            _validate_state_schema(item, target, f"{field}[{index}]")


def _validate_load_optimizer(
    payload: dict, model: nn.Module, optimizer: AdamW,
    completed_updates: int,
) -> list[dict]:
    saved = payload["optimizer_state"]
    live = optimizer.state_dict()
    if type(saved) is not dict or set(saved) != {"state", "param_groups"}:
        raise CLEVRCheckpointError("optimizer fields disagree")
    _validate_state_schema(saved["param_groups"], live["param_groups"], "optimizer groups")
    parameters = dict(model.named_parameters())
    templates = {}
    for group, target in zip(saved["param_groups"], live["param_groups"]):
        for field in ("group_name", "param_names", "params"):
            _require_state_equal(group[field], target[field], f"optimizer {field}")
        for field in target:
            if field not in {"params", "lr"}:
                _require_state_equal(
                    group[field], target[field], f"optimizer protocol field {field}",
                )
        for identifier, name in zip(group["params"], group["param_names"]):
            parameter = parameters[name]
            step_dtype = (torch.float64 if torch.get_default_dtype() == torch.float64
                          and not group["fused"] else torch.float32)
            template = {
                "step": torch.empty((), dtype=step_dtype),
                "exp_avg": torch.empty_like(parameter, device="cpu"),
                "exp_avg_sq": torch.empty_like(parameter, device="cpu"),
            }
            if group["amsgrad"]:
                template["max_exp_avg_sq"] = torch.empty_like(parameter, device="cpu")
            templates[identifier] = template
    # AdamW materializes state lazily. Validate the saved subset against the
    # named parameter mapping, independently of the live optimizer's subset.
    states = saved["state"]
    if type(states) is not dict or not set(states).issubset(templates):
        raise CLEVRCheckpointError("optimizer state parameter identifiers disagree")
    for identifier, state in states.items():
        if type(state) is dict and not state:
            continue  # Public defaultdict access can create unmaterialized state.
        _validate_state_schema(state, templates[identifier], f"optimizer state {identifier}")
        step = state["step"]
        if not bool(torch.isfinite(step)) or step.item() != completed_updates:
            raise CLEVRCheckpointError(
                "optimizer step disagrees with completed update boundary"
            )
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if field in state and not bool(torch.isfinite(state[field]).all()):
                raise CLEVRCheckpointError(
                    f"optimizer state {identifier}.{field} must be finite"
                )
    return saved["param_groups"]


def _scheduler_learning_rates_at_boundary(
    scheduler: LambdaLR, completed_updates: int,
) -> list[float]:
    """Evaluate the supplied protocol scheduler without mutating live state."""
    lambdas = getattr(scheduler, "lr_lambdas", None)
    base_lrs = scheduler.state_dict().get("base_lrs")
    if (type(lambdas) is not list or type(base_lrs) is not list
            or len(lambdas) != len(base_lrs)):
        raise CLEVRCheckpointError("scheduler protocol functions/base rates disagree")
    expected: list[float] = []
    for base_lr, schedule in zip(base_lrs, lambdas):
        if type(base_lr) not in (int, float) or not math.isfinite(base_lr):
            raise CLEVRCheckpointError("scheduler base learning rate must be finite")
        try:
            multiplier = schedule(completed_updates)
        except BaseException as exc:
            raise CLEVRCheckpointError("scheduler protocol evaluation failed") from exc
        if type(multiplier) not in (int, float) or not math.isfinite(multiplier):
            raise CLEVRCheckpointError("scheduler multiplier must be finite")
        learning_rate = float(base_lr * multiplier)
        if not math.isfinite(learning_rate):
            raise CLEVRCheckpointError("scheduler learning rate must be finite")
        expected.append(learning_rate)
    return expected


def _numpy_legacy_state(state: dict) -> tuple:
    return (state["bit_generator"], state["keys"].numpy().astype(np.uint32, copy=True),
            state["pos"], state["has_gauss"], state["cached_gaussian"])


def _apply_load_state(
    payload: dict, model: nn.Module, optimizer: AdamW, scheduler: LambdaLR,
    sampler: CLEVRIndexStream, cuda_devices: tuple[int, ...],
) -> None:
    model.load_state_dict(_snapshot_value(payload["model_state"], "model"), strict=True)
    scheduler.load_state_dict(_snapshot_value(payload["scheduler_state"], "scheduler"))
    optimizer.load_state_dict(_snapshot_value(payload["optimizer_state"], "optimizer"))
    sampler.load_state_dict(_snapshot_value(payload["sampler_state"], "sampler"))
    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(_numpy_legacy_state(rng["numpy"]))
    torch.random.set_rng_state(rng["torch_cpu"].clone())
    for device, state in zip(cuda_devices, rng["torch_cuda"]):
        torch.cuda.set_rng_state(state.clone(), device)


def _rollback_load_state(
    entry: dict, model: nn.Module, optimizer: AdamW, scheduler: LambdaLR,
    sampler: CLEVRIndexStream, cuda_devices: tuple[int, ...],
) -> list[tuple[str, BaseException]]:
    """Attempt every reverse-order cleanup, including after BaseException."""
    rng = entry["rng_state"]
    actions = [
        (f"CUDA RNG device {device}",
         lambda state=state, device=device: torch.cuda.set_rng_state(state.clone(), device))
        for device, state in reversed(tuple(zip(cuda_devices, rng["torch_cuda"])))
    ]
    actions.extend([
        ("torch CPU RNG", lambda: torch.random.set_rng_state(rng["torch_cpu"].clone())),
        ("NumPy RNG", lambda: np.random.set_state(_numpy_legacy_state(rng["numpy"]))),
        ("Python RNG", lambda: random.setstate(rng["python"])),
        ("sampler", lambda: sampler.load_state_dict(_snapshot_value(entry["sampler_state"], "sampler rollback"))),
        ("optimizer", lambda: optimizer.load_state_dict(_snapshot_value(entry["optimizer_state"], "optimizer rollback"))),
        ("scheduler", lambda: scheduler.load_state_dict(_snapshot_value(entry["scheduler_state"], "scheduler rollback"))),
        ("model", lambda: model.load_state_dict(_snapshot_value(entry["model_state"], "model rollback"), strict=True)),
    ])
    failures = []
    for component, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append((component, error))
    if not failures:
        try:
            _verify_load_state(entry, model, optimizer, scheduler, sampler, cuda_devices)
        except BaseException as error:
            failures.append(("equality verification", error))
    return failures


def _require_state_equal(actual: object, expected: object, field: str) -> None:
    if type(actual) is not type(expected):
        raise CLEVRCheckpointError(f"{field}: state types disagree")
    if type(expected) is torch.Tensor:
        equal = (actual.shape == expected.shape and actual.dtype == expected.dtype
                 and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    elif isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise CLEVRCheckpointError(f"{field}: state fields disagree")
        if type(expected) is OrderedDict:
            _require_state_equal(vars(actual), vars(expected), f"{field} attributes")
        for key in expected:
            _require_state_equal(actual[key], expected[key], f"{field}.{key}")
        return
    elif type(expected) in (tuple, list):
        if len(actual) != len(expected):
            raise CLEVRCheckpointError(f"{field}: state lengths disagree")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _require_state_equal(left, right, f"{field}[{index}]")
        return
    else:
        equal = actual == expected
    if not equal:
        raise CLEVRCheckpointError(f"{field}: state values disagree")


def _capture_load_state(
    model: nn.Module, optimizer: AdamW, scheduler: LambdaLR,
    sampler: CLEVRIndexStream, cuda_devices: tuple[int, ...],
) -> dict[str, object]:
    return _build_payload(
        method="G", run_seed=sampler.run_seed, completed_updates=0,
        model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
        cuda_devices=cuda_devices,
    )


def _verify_load_state(
    payload: dict, model: nn.Module, optimizer: AdamW, scheduler: LambdaLR,
    sampler: CLEVRIndexStream, cuda_devices: tuple[int, ...],
) -> None:
    actual = _capture_load_state(model, optimizer, scheduler, sampler, cuda_devices)
    for field in ("model_state", "optimizer_state", "scheduler_state", "sampler_state", "rng_state"):
        _require_state_equal(actual[field], payload[field], field)


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
        if set(vars(value)) - {"_metadata"}:
            raise CLEVRCheckpointError(f"{field} contains unsupported OrderedDict attributes")
        result = OrderedDict(
            (_snapshot_key(key, field), _snapshot_value(item, field))
            for key, item in value.items()
        )
        if "_metadata" in vars(value):
            if type(value._metadata) not in (dict, OrderedDict):
                raise CLEVRCheckpointError(f"{field}._metadata must be a mapping")
            result._metadata = _snapshot_value(value._metadata, f"{field}._metadata")
        return result
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
    descriptor: int | None = None
    staging_name: str | None = None
    staging: Path | None = None
    output: object | None = None
    staging_identity: tuple[int, int] | None = None
    inode_reference: int | None = None
    reference_release_started = False
    link_started = False
    link_returned = False
    staging_unlink_started = False
    committed = False
    try:
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{path.name}.staging-",
            dir=path.parent,
        )
        acquired_stat = os.fstat(descriptor)
        staging_identity = (acquired_stat.st_dev, acquired_stat.st_ino)
        staging = Path(staging_name)
        inode_reference = os.dup(descriptor)
        os.fchmod(descriptor, 0o600)
        output = os.fdopen(descriptor, "wb")
        descriptor = None
        # Capture writer/flush failures in the outer handler before closing;
        # a simultaneous close failure belongs to cleanup, not the primary.
        torch.save(payload, output)
        output.flush()
        output.close()
        if not _path_matches_live_regular_inode(
            staging, inode_reference, staging_identity,
        ):
            raise CLEVRCheckpointError("checkpoint staging inode changed before publication")
        link_started = True
        os.link(staging, path)
        link_returned = True
        committed = True
        if not _path_matches_live_regular_inode(
            staging, inode_reference, staging_identity,
        ):
            raise CLEVRCheckpointError("checkpoint staging inode changed before cleanup")
        staging_unlink_started = True
        os.unlink(staging)
        reference_release_started = True
        os.close(inode_reference)
        inode_reference = None
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
        cleanup_failures: list[tuple[str, BaseException]] = []

        if staging_identity is None and descriptor is not None:
            try:
                recovered_stat = os.fstat(descriptor)
                staging_identity = (recovered_stat.st_dev, recovered_stat.st_ino)
            except BaseException as cleanup_error:
                cleanup_failures.append(("staging identity recovery", cleanup_error))

        if output is not None and not bool(getattr(output, "closed", False)):
            try:
                output.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(("descriptor close", cleanup_error))

        if (
            not committed
            and link_started
            and not link_returned
            and staging_identity is not None
        ):
            try:
                if staging is None:
                    raise CLEVRCheckpointError(
                        "checkpoint staging path is unavailable for reconciliation"
                    )
                committed = _paths_are_same_regular_inode(
                    staging,
                    path,
                    staging_identity,
                    inode_reference if inode_reference is not None else descriptor,
                )
            except BaseException as cleanup_error:
                cleanup_failures.append(("commit reconciliation", cleanup_error))

        if not staging_unlink_started and staging_name is not None:
            try:
                _unlink_owned_staging(
                    staging_name,
                    staging_identity,
                    inode_reference if inode_reference is not None else descriptor,
                )
            except BaseException as cleanup_error:
                cleanup_failures.append(("staging cleanup", cleanup_error))

        # If duplication failed, the original descriptor pins the inode until
        # cleanup finishes. Otherwise the independent reference does.
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                cleanup_failures.append(("descriptor close", cleanup_error))
        if inode_reference is not None:
            if reference_release_started:
                cleanup_failures.append((
                    "inode reference release",
                    CLEVRCheckpointError(
                        f"fd {inode_reference} release was attempted; disposition unknown"
                    ),
                ))
            else:
                try:
                    os.close(inode_reference)
                except BaseException as cleanup_error:
                    cleanup_error.add_note(
                        f"inode reference fd {inode_reference}; close will not be retried"
                    )
                    cleanup_failures.append(("inode reference release", cleanup_error))

        if committed or cleanup_failures:
            status = "committed" if committed else "not committed"
            publication_error = CLEVRCheckpointPublicationError(
                f"checkpoint publication failed ({status})",
                committed=committed,
            )
            for phase, cleanup_error in cleanup_failures:
                publication_error.add_note(
                    f"checkpoint {phase} failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise publication_error from primary
        raise


def _paths_are_same_regular_inode(
    staging: str | os.PathLike[str],
    target: str | os.PathLike[str],
    expected_identity: tuple[int, int],
    inode_reference: int | None,
) -> bool:
    try:
        staging_stat = os.stat(staging, follow_symlinks=False)
        target_stat = os.stat(target, follow_symlinks=False)
        reference_stat = os.fstat(inode_reference) if inode_reference is not None else None
    except OSError:
        return False
    if (reference_stat is None or not stat.S_ISREG(reference_stat.st_mode)
            or not stat.S_ISREG(staging_stat.st_mode)
            or not stat.S_ISREG(target_stat.st_mode)):
        return False
    staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
    target_identity = (target_stat.st_dev, target_stat.st_ino)
    reference_identity = (reference_stat.st_dev, reference_stat.st_ino)
    return staging_identity == expected_identity == target_identity == reference_identity


def _path_matches_live_regular_inode(
    path: str | os.PathLike[str],
    inode_reference: int | None,
    expected_identity: tuple[int, int],
) -> bool:
    if inode_reference is None:
        return False
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        reference_stat = os.fstat(inode_reference)
    except OSError:
        return False
    if not stat.S_ISREG(path_stat.st_mode) or not stat.S_ISREG(reference_stat.st_mode):
        return False
    return (
        (path_stat.st_dev, path_stat.st_ino)
        == expected_identity
        == (reference_stat.st_dev, reference_stat.st_ino)
    )


def _unlink_owned_staging(
    staging: str | os.PathLike[str],
    expected_identity: tuple[int, int] | None,
    inode_reference: int | None,
) -> None:
    try:
        staging_stat = os.stat(staging, follow_symlinks=False)
    except FileNotFoundError:
        return
    if expected_identity is None or inode_reference is None:
        raise CLEVRCheckpointError(
            "cannot prove ownership of checkpoint staging path"
        )
    try:
        reference_stat = os.fstat(inode_reference)
    except OSError as exc:
        raise CLEVRCheckpointError(
            "checkpoint staging ownership reference is unavailable"
        ) from exc
    if (not stat.S_ISREG(staging_stat.st_mode)
            or not stat.S_ISREG(reference_stat.st_mode)
            or (staging_stat.st_dev, staging_stat.st_ino) != expected_identity
            or (reference_stat.st_dev, reference_stat.st_ino) != expected_identity):
        raise CLEVRCheckpointError(
            "checkpoint staging ownership lost; retained foreign file"
        )
    os.unlink(staging)
