"""Immutable run evidence and exclusive attempt journals for CLEVR PUB-R1.1."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import stat
import tempfile
import threading
from typing import BinaryIO

import numpy as np
import torch

PROTOCOL_VERSION = "PUB-R1.1"
_PHASES = {
    "smoke": ((9001,), 1, "engineering-only"),
    "pilot": ((9001,), 800, "engineering-only"),
    "formal": ((101, 102, 103, 104, 105), 8000, "formal"),
}
_METHODS = ("G", "B", "M10", "Q-only")


class CLEVRRunArtifactError(ValueError):
    """A contract failure; committed=None means commit status is unknown."""

    def __init__(self, message: str, *, committed: bool | None = False) -> None:
        super().__init__(message)
        self.committed = committed


class _TargetExists(CLEVRRunArtifactError):
    """A cleanly reconciled collision with another publisher."""


@dataclass(frozen=True, slots=True)
class CLEVRRunSpec:
    phase: str
    method: str
    seed: int

    def __post_init__(self) -> None:
        if type(self.phase) is not str or self.phase not in _PHASES:
            raise CLEVRRunArtifactError("phase must be smoke, pilot, or formal")
        if type(self.method) is not str or self.method not in _METHODS:
            raise CLEVRRunArtifactError("method must be G, B, M10, or Q-only")
        if type(self.seed) is not int or self.seed not in _PHASES[self.phase][0]:
            raise CLEVRRunArtifactError("seed disagrees with the frozen phase")

    @property
    def target_updates(self) -> int:
        return _PHASES[self.phase][1]

    @property
    def evidence_role(self) -> str:
        return _PHASES[self.phase][2]


@dataclass(frozen=True, slots=True)
class CLEVRPredictionRecord:
    question_index: int
    label: int
    logits: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.question_index) is not int or self.question_index < 0:
            raise CLEVRRunArtifactError("question_index must be an ordinary nonnegative integer")
        if type(self.label) is not int or self.label not in (0, 1):
            raise CLEVRRunArtifactError("label must be the ordinary integer 0 or 1")
        if (type(self.logits) is not tuple or len(self.logits) != 2
                or any(type(value) is not float or not math.isfinite(value) for value in self.logits)):
            raise CLEVRRunArtifactError("logits must be a tuple of two finite ordinary floats")


def _json_value(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise CLEVRRunArtifactError("JSON object keys must be ordinary strings")
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_json_value(item) for item in value]
    raise CLEVRRunArtifactError("value is not strict JSON")


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(_json_value(value), allow_nan=False, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise CLEVRRunArtifactError("strict JSON serialization failed") from exc


def _json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CLEVRRunArtifactError("duplicate JSON key")
        result[key] = value
    return result


def _directory(path: Path, *, create: bool) -> None:
    # Walk without resolving symlinks, including the caller's explicit root.
    if path == path.parent:
        return
    _directory(path.parent, create=create)
    if create:
        try:
            path.mkdir()
        except FileExistsError:
            pass
    entry = path.lstat()
    if not stat.S_ISDIR(entry.st_mode):
        raise CLEVRRunArtifactError(f"directory is substituted: {path}")


def _identity(path: Path) -> tuple[int, int]:
    entry = path.lstat()
    if not stat.S_ISREG(entry.st_mode):
        raise CLEVRRunArtifactError(f"expected regular file: {path}")
    return entry.st_dev, entry.st_ino


def _write_bytes(output: BinaryIO, content: bytes) -> None:
    if output.write(content) != len(content):
        raise CLEVRRunArtifactError("short write; complete evidence was not written")


class _CompleteTensorWriter:
    """torch.save's file-like seam must not silently accept short writes."""

    def __init__(self, output: BinaryIO) -> None:
        self.output = output

    def write(self, content: bytes) -> int:
        _write_bytes(self.output, content)
        return len(content)

    def flush(self) -> None:
        self.output.flush()


def _publish(path: Path, writer: Callable[[BinaryIO], object]) -> Path:
    """Commit complete bytes with link; never remove a final target on failure.

    A separate descriptor pins the inode through stream close and cleanup.
    A link that raises after committing is reconciled against that live inode,
    not exception wording.
    """
    descriptor = None
    output = None
    staging = None
    owned = None
    reference = None
    reference_release_started = False
    link_started = False
    committed: bool | None = False
    cleanup_started = False
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.staging-", dir=path.parent)
        staging = Path(name)
        entry = os.fstat(descriptor)
        owned = (entry.st_dev, entry.st_ino)
        reference = os.dup(descriptor)
        os.fchmod(descriptor, 0o600)
        output = os.fdopen(descriptor, "wb")
        descriptor = None
        writer(output)
        output.flush()
        output.close()
        if _identity(staging) != owned:
            raise CLEVRRunArtifactError("staging inode changed before publication")
        link_started = True
        os.link(staging, path)
        committed = True
        if _identity(staging) != owned:
            raise CLEVRRunArtifactError("staging inode changed before cleanup")
        cleanup_started = True
        os.unlink(staging)
        reference_release_started = True
        os.close(reference)
        reference = None
        return path
    except BaseException as primary:
        failures = []
        if owned is None and descriptor is not None:
            try:
                entry = os.fstat(descriptor)
                owned = (entry.st_dev, entry.st_ino)
            except BaseException as exc:
                failures.append(("identity recovery", exc))
        try:
            if output is not None and not output.closed:
                output.close()
        except BaseException as exc:
            failures.append(("close", exc))
        if link_started and committed is not True:
            try:
                committed = _identity(path) == owned
            except FileNotFoundError:
                committed = False
            except BaseException as exc:
                committed = None
                failures.append(("commit reconciliation", exc))
        if staging is not None and not cleanup_started:
            try:
                if _identity(staging) != owned:
                    raise CLEVRRunArtifactError("staging ownership lost; retained foreign file")
                os.unlink(staging)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                failures.append(("staging cleanup", exc))
        # If dup failed, the original descriptor must still pin the inode until
        # staging cleanup finishes. Otherwise the independent reference does.
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as exc:
                failures.append(("descriptor close", exc))
        if reference is not None:
            if reference_release_started:
                # close may have released the fd before raising. Retrying could
                # close a descriptor since reused by another thread.
                failures.append(("inode reference release", CLEVRRunArtifactError(
                    f"fd {reference} release was attempted; disposition unknown")))
            else:
                try:
                    os.close(reference)
                except BaseException as exc:
                    exc.add_note(f"inode reference fd {reference}; close will not be retried")
                    failures.append(("inode reference release", exc))
        error_type = (_TargetExists if isinstance(primary, FileExistsError)
                      and committed is False and not failures else CLEVRRunArtifactError)
        error = error_type(f"publication failed: {path} (committed={committed})",
                           committed=committed)
        for phase, failure in failures:
            error.add_note(f"{phase}: {type(failure).__name__}: {failure}")
        raise error from primary


def _environment() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": str(np.__version__),
            "torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda)}


def _snapshot_common(common_state: Mapping[str, torch.Tensor]) -> OrderedDict:
    if not isinstance(common_state, Mapping) or not common_state:
        raise CLEVRRunArtifactError("common_state must be a nonempty tensor mapping")
    snapshot = OrderedDict()
    for key, value in common_state.items():
        if type(key) is not str or not key or type(value) is not torch.Tensor:
            raise CLEVRRunArtifactError("common_state requires string keys and plain tensors")
        if value.device.type == "meta" or value.layout != torch.strided:
            raise CLEVRRunArtifactError("common_state requires materialized strided tensors")
        cloned = value.detach().cpu().clone()
        try:
            reflexive = torch.equal(cloned, cloned)
        except RuntimeError as exc:
            raise CLEVRRunArtifactError(f"common-state exact equality is unsupported: {key}") from exc
        if not reflexive:
            raise CLEVRRunArtifactError(f"common-state exact equality is not reflexive: {key}")
        snapshot[key] = cloned
    return snapshot


def _verify_common(path: Path, expected: dict[str, object]) -> None:
    try:
        _identity(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise CLEVRRunArtifactError("restricted common-state read failed") from exc
    if type(payload) is not dict or set(payload) != set(expected):
        raise CLEVRRunArtifactError("common-state payload fields disagree")
    if (type(payload["protocol_version"]) is not str
            or payload["protocol_version"] != PROTOCOL_VERSION
            or type(payload["run_seed"]) is not int
            or payload["run_seed"] != expected["run_seed"]):
        raise CLEVRRunArtifactError("common-state protocol or seed disagrees")
    if _json_bytes(payload["environment"]) != _json_bytes(expected["environment"]):
        raise CLEVRRunArtifactError("common-state environment disagrees")
    saved = payload["common_state"]
    wanted = expected["common_state"]
    if type(saved) not in (dict, OrderedDict) or list(saved) != list(wanted):
        raise CLEVRRunArtifactError("common-state ordered keys disagree")
    for key, value in saved.items():
        reference = wanted[key]
        if (type(value) is not torch.Tensor or value.device.type != "cpu"
                or value.layout != torch.strided or value.shape != reference.shape
                or value.dtype != reference.dtype or not torch.equal(value, reference)):
            raise CLEVRRunArtifactError(f"common-state tensor disagrees: {key}")


class CLEVRRunArtifacts:
    """One process attempt in an immutable phase/seed/method directory."""

    @classmethod
    def open(
        cls, *, output_root: Path, spec: CLEVRRunSpec,
        expanded_config: Mapping[str, object], resume: bool,
    ) -> CLEVRRunArtifacts:
        if not isinstance(output_root, Path) or type(spec) is not CLEVRRunSpec:
            raise CLEVRRunArtifactError("explicit Path and exact CLEVRRunSpec are required")
        if type(resume) is not bool or not isinstance(expanded_config, Mapping):
            raise CLEVRRunArtifactError("resume must be bool and expanded_config a mapping")
        frozen = {"protocol_version": PROTOCOL_VERSION, "phase": spec.phase,
                  "method": spec.method, "seed": spec.seed,
                  "target_updates": spec.target_updates, "evidence_role": spec.evidence_role}
        config = dict(expanded_config)
        for key, value in frozen.items():
            if key in config and _json_bytes(config[key]) != _json_bytes(value):
                raise CLEVRRunArtifactError(f"expanded config conflicts with {key}")
            config[key] = value
        serialized = _json_bytes(config) + b"\n"
        run_dir = output_root.absolute() / spec.phase / f"seed_{spec.seed}" / spec.method
        path = run_dir / "run_config.json"
        try:
            if resume:
                _directory(run_dir, create=False)
                _identity(path)
                document = json.loads(path.read_bytes(), object_pairs_hook=_json_pairs)
                if _json_bytes(document) + b"\n" != serialized:
                    raise CLEVRRunArtifactError("resume config disagrees")
            else:
                _directory(run_dir.parent, create=True)
                run_dir.mkdir()  # Exclusive claim: even an empty old run is evidence.
                _publish(path, lambda output: _write_bytes(output, serialized))
        except (OSError, ValueError) as exc:
            if isinstance(exc, CLEVRRunArtifactError):
                raise
            raise CLEVRRunArtifactError("run directory/config preflight failed") from exc
        result = cls()
        result.run_dir = run_dir
        result.spec = spec
        result._lock = threading.Lock()
        result._closed = False
        result._journal_failed = False
        attempts = run_dir / "attempts"
        try:
            _directory(attempts, create=True)
            existing = [int(match[1]) for path in attempts.iterdir()
                        if (match := re.fullmatch(r"attempt_([0-9]{4,})\.jsonl", path.name))]
            attempt = max(existing, default=0) + 1
            while True:
                journal_path = attempts / f"attempt_{attempt:04d}.jsonl"
                try:
                    descriptor = os.open(journal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                         0o600)
                    break
                except FileExistsError:
                    attempt += 1
            try:
                result._journal = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)
                raise
        except OSError as exc:
            raise CLEVRRunArtifactError("attempt allocation failed") from exc
        result.attempt = attempt
        result.journal_path = journal_path
        return result

    def _require_open(self) -> None:
        if self._closed or self._journal_failed:
            raise CLEVRRunArtifactError("attempt is closed or its journal failed")

    def _update(self, update: int) -> None:
        if type(update) is not int or not 0 <= update <= self.spec.target_updates:
            raise CLEVRRunArtifactError("update must be an ordinary integer in the run budget")

    def append_event(
        self, *, completed_updates: int, event: str, details: Mapping[str, object],
    ) -> None:
        with self._lock:
            self._require_open()
            self._update(completed_updates)
            if type(event) is not str or not event.strip() or not isinstance(details, Mapping):
                raise CLEVRRunArtifactError("event requires a nonempty string and details mapping")
            row = {"protocol_version": PROTOCOL_VERSION, "phase": self.spec.phase,
                   "method": self.spec.method, "seed": self.spec.seed, "attempt": self.attempt,
                   "completed_updates": completed_updates, "event": event, "details": details}
            line = _json_bytes(row) + b"\n"
            try:
                _write_bytes(self._journal, line)
                self._journal.flush()
            except BaseException:
                self._journal_failed = True
                raise

    def checkpoint_path(self, update: int) -> Path:
        self._require_open()
        self._update(update)
        parent = self.run_dir / "checkpoints"
        _directory(parent, create=True)
        return parent / f"update_{update:04d}.pt"

    def prediction_path(self, partition: str, update: int) -> Path:
        self._require_open()
        self._update(update)
        if type(partition) is not str or partition not in ("train", "dev"):
            raise CLEVRRunArtifactError("prediction_path requires train or dev")
        parent = self.run_dir / "predictions" / partition
        _directory(parent, create=True)
        path = parent / f"update_{update:04d}.jsonl"
        if path.exists() or path.is_symlink():
            raise CLEVRRunArtifactError("prediction target already exists")
        return path

    def ensure_common_state(self, common_state: Mapping[str, torch.Tensor]) -> Path:
        self._require_open()
        payload = {"protocol_version": PROTOCOL_VERSION, "run_seed": self.spec.seed,
                   "environment": _environment(), "common_state": _snapshot_common(common_state)}
        path = self.run_dir.parent / "common_state.pt"
        _directory(path.parent, create=False)
        committed = False
        if not path.exists() and not path.is_symlink():
            try:
                _publish(path, lambda output: torch.save(payload, _CompleteTensorWriter(output)))
            except _TargetExists:
                pass  # A concurrent winner must pass the same strict read below.
            else:
                committed = True
        try:
            _verify_common(path, payload)
        except CLEVRRunArtifactError as exc:
            exc.committed = committed
            exc.add_note(f"common-state verification failed (this invocation committed={committed})")
            raise
        return path

    def publish_predictions(
        self, *, partition: str, completed_updates: int,
        rows: Sequence[CLEVRPredictionRecord],
    ) -> Path:
        self._require_open()
        self._update(completed_updates)
        if type(partition) is not str:
            raise CLEVRRunArtifactError("partition must be an ordinary string")
        formal_final = self.spec.phase == "formal" and completed_updates == 8000
        allowed = ((partition in ("train", "base_test") and formal_final)
                   or (partition == "dev" and (
                       (self.spec.phase == "pilot" and completed_updates == 800)
                       or (self.spec.phase == "formal" and completed_updates > 0
                           and completed_updates % 1000 == 0))))
        if not allowed:
            raise CLEVRRunArtifactError("partition/update is not a protocol evaluation boundary")
        if (not isinstance(rows, Sequence) or not rows
                or any(type(row) is not CLEVRPredictionRecord for row in rows)):
            raise CLEVRRunArtifactError("rows must be a nonempty prediction-record sequence")
        snapshot = tuple(rows)
        if len({row.question_index for row in snapshot}) != len(snapshot):
            raise CLEVRRunArtifactError("duplicate prediction question_index")
        content = b"".join(_json_bytes({
            "protocol_version": PROTOCOL_VERSION, "method": self.spec.method,
            "seed": self.spec.seed, "question_index": row.question_index,
            "label": row.label, "logits": list(row.logits),
        }) + b"\n" for row in sorted(snapshot, key=lambda row: row.question_index))
        if partition == "base_test":
            parent = self.run_dir / "predictions" / "base_test"
            _directory(parent, create=True)
            path = parent / f"{self.spec.method}_seed_{self.spec.seed}.jsonl"
        else:
            path = self.prediction_path(partition, completed_updates)
        return _publish(path, lambda output: _write_bytes(output, content))

    def close(self) -> None:
        """Close this invocation's journal; repeated close is harmless."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._journal.close()
