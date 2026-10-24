"""PUB-R1.1 computation engine and explicit production orchestration.

A failed update is not rollback-safe. Recover from an explicit completed
checkpoint; only a returned result denotes a successfully committed update.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import stat

import numpy as np
import PIL
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import timm
import torchvision

from src.clevr_dataset import (
    CLEVRSampleIdentity,
    CLEVRQuestionDataset,
    CLEVRQuestionOnlyDataset,
    collate_clevr_question_only,
    collate_clevr_questions,
)
from src.clevr_checkpoint import load_clevr_checkpoint, save_clevr_checkpoint
from src.clevr_run_artifacts import (
    CLEVRPredictionRecord,
    CLEVRRunArtifacts,
    CLEVRRunSpec,
    PROTOCOL_VERSION,
)
from src.clevr_sampling import CLEVRIndexStream
from src.clevr_training import (
    CLEVRPairedInitialization,
    build_clevr_adamw,
    build_clevr_lr_scheduler,
)


class CLEVRRunnerError(RuntimeError):
    """The injected computation violates the frozen CLEVR contract."""


@dataclass(frozen=True, slots=True)
class CLEVRUpdateResult:
    dataset_indices: tuple[int, ...]
    question_indices: tuple[int, ...]
    loss_sum: float
    loss_mean: float
    gradient_norm: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class CLEVREvaluationResult:
    partition: str
    total_questions: int
    correct: int
    loss_sum: float
    cross_entropy: float
    accuracy: float
    rows: tuple[CLEVRPredictionRecord, ...]


@dataclass(frozen=True, slots=True)
class CLEVRRunPreflight:
    methods: tuple[str, ...]
    specs: Mapping[str, CLEVRRunSpec]
    train_datasets: Mapping[str, object]
    dev_datasets: Mapping[str, object]
    vocab_size: int


_METHOD_ORDER = ("G", "B", "M10", "Q-only")


def _normalize_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    if type(methods) is not tuple or not methods:
        raise CLEVRRunnerError("methods must be a nonempty tuple")
    if any(type(method) is not str or method not in _METHOD_ORDER for method in methods):
        raise CLEVRRunnerError("methods must contain only G, B, M10, or Q-only")
    if len(set(methods)) != len(methods):
        raise CLEVRRunnerError("methods must not contain duplicates")
    return tuple(method for method in _METHOD_ORDER if method in methods)


def _absolute_directory(path: Path, name: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or not path.is_dir():
        raise CLEVRRunnerError(f"{name} must be an explicit existing absolute directory")


def _explicit_output_root(path: Path) -> None:
    if (not isinstance(path, Path) or not path.is_absolute()
            or (path.exists() and not path.is_dir())):
        raise CLEVRRunnerError(
            "output_root must be an explicit absolute path that is absent or a directory"
        )


def _validate_request(
    *, spec_phase: str, seed: int, methods: tuple[str, ...], audit_dir: Path,
    data_root: Path | None, device_index: int,
) -> tuple[tuple[str, ...], dict[str, CLEVRRunSpec]]:
    ordered = _normalize_methods(methods)
    _absolute_directory(audit_dir, "audit_dir")
    needs_rgb = any(method != "Q-only" for method in ordered)
    if needs_rgb:
        if data_root is None:
            raise CLEVRRunnerError("data_root is required when an RGB method is requested")
        _absolute_directory(data_root, "data_root")
    elif data_root is not None:
        raise CLEVRRunnerError("data_root must be absent for a Q-only-only request")
    if type(device_index) is not int or device_index < 0:
        raise CLEVRRunnerError("device_index must be an ordinary nonnegative integer")
    try:
        specs = {method: CLEVRRunSpec(spec_phase, method, seed) for method in ordered}
    except ValueError as exc:
        raise CLEVRRunnerError("phase, seed, or method violates the frozen run matrix") from exc
    return ordered, specs


def _require_dataset_pair(rgb: object, qonly: object, partition: str) -> None:
    if len(rgb) != len(qonly):
        raise CLEVRRunnerError(f"RGB and Q-only {partition} lengths disagree")
    if getattr(rgb, "vocab_size", None) != getattr(qonly, "vocab_size", None):
        raise CLEVRRunnerError(f"RGB and Q-only {partition} vocabularies disagree")
    for index in range(len(rgb)):
        if rgb.sample_identity(index) != qonly.sample_identity(index):
            raise CLEVRRunnerError(f"RGB and Q-only {partition} identities disagree")


def preflight_clevr_run(
    *, spec_phase: str, seed: int, methods: tuple[str, ...], audit_dir: Path,
    data_root: Path | None, device_index: int,
) -> CLEVRRunPreflight:
    """Validate one seed request and build only its train/dev dataset views."""
    ordered, specs = _validate_request(
        spec_phase=spec_phase, seed=seed, methods=methods, audit_dir=audit_dir,
        data_root=data_root, device_index=device_index,
    )
    rgb_train = rgb_dev = q_train = q_dev = None
    if any(method != "Q-only" for method in ordered):
        assert data_root is not None
        rgb_train = CLEVRQuestionDataset(
            data_root=data_root, audit_dir=audit_dir, partition="train",
        )
        rgb_dev = CLEVRQuestionDataset(
            data_root=data_root, audit_dir=audit_dir, partition="dev",
        )
    if "Q-only" in ordered:
        q_train = CLEVRQuestionOnlyDataset(audit_dir=audit_dir, partition="train")
        q_dev = CLEVRQuestionOnlyDataset(audit_dir=audit_dir, partition="dev")
    if rgb_train is not None and q_train is not None:
        _require_dataset_pair(rgb_train, q_train, "train")
        _require_dataset_pair(rgb_dev, q_dev, "dev")
    all_datasets = [dataset for dataset in (rgb_train, rgb_dev, q_train, q_dev)
                    if dataset is not None]
    if not all_datasets:
        raise CLEVRRunnerError("preflight constructed no datasets")
    vocab_size = getattr(all_datasets[0], "vocab_size", None)
    if (type(vocab_size) is not int or vocab_size < 4
            or any(getattr(dataset, "vocab_size", None) != vocab_size
                   for dataset in all_datasets)):
        raise CLEVRRunnerError("train/dev dataset vocabularies disagree or are invalid")
    train_datasets = {
        method: q_train if method == "Q-only" else rgb_train for method in ordered
    }
    dev_datasets = {
        method: q_dev if method == "Q-only" else rgb_dev for method in ordered
    }
    return CLEVRRunPreflight(ordered, specs, train_datasets, dev_datasets, vocab_size)


def _setup_clevr_runtime(device_index: int) -> torch.device:
    if type(device_index) is not int or device_index < 0:
        raise CLEVRRunnerError("device_index must be an ordinary nonnegative integer")
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise CLEVRRunnerError("the explicitly requested CUDA device is unavailable")
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(f"cuda:{device_index}")


_AUDIT_ARTIFACT_NAMES = (
    "selected_questions.jsonl", "vocabulary.json", "audit_report.json",
)


def _read_regular_utf8(path: Path, description: str) -> str:
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise CLEVRRunnerError(f"{description} is not a regular file")
            content = source.read()
        current = path.lstat()
        if (not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
            raise CLEVRRunnerError(f"{description} changed while it was read")
        return content.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CLEVRRunnerError(f"{description} is unreadable UTF-8") from exc


def _audit_artifact_snapshot(audit_dir: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for name in _AUDIT_ARTIFACT_NAMES:
        snapshot[name] = _read_regular_utf8(
            audit_dir / name, f"audit artifact {name}",
        )
    return snapshot


def _expanded_clevr_config(
    *, device_index: int, audit_dir: Path, data_root: Path | None,
    vocab_size: int, train_size: int,
    audit_artifacts: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if audit_artifacts is None:
        audit_artifacts = _audit_artifact_snapshot(audit_dir)
    elif (not isinstance(audit_artifacts, Mapping)
          or tuple(audit_artifacts) != _AUDIT_ARTIFACT_NAMES
          or any(type(value) is not str for value in audit_artifacts.values())):
        raise CLEVRRunnerError("audit artifact snapshot is invalid")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "data": {
            "audit_dir": str(audit_dir.resolve()),
            "data_root": None if data_root is None else str(data_root.resolve()),
            "train_size": train_size,
            "vocab_size": vocab_size,
            "audit_artifacts": dict(audit_artifacts),
        },
        "packages": {
            "python": platform.python_version(), "numpy": str(np.__version__),
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "timm": str(timm.__version__), "pillow": str(PIL.__version__),
            "cuda_runtime": str(torch.version.cuda),
        },
        "model": {
            "image_shape": [3, 224, 224], "patch_size": 16, "grid_size": [14, 14],
            "embed_dim": 192, "depth": 6, "heads": 3, "head_dim": 64,
            "mlp_dim": 768, "num_classes": 2, "question_embedding_dim": 64,
            "question_hidden_dim": 192, "question_layers": 1,
            "question_bidirectional": False, "attention_backend": "manual",
            "layer_norm_epsilon": 1e-6, "layer_scale_initial": 1e-4,
            "dropout": 0.0, "attention_dropout": 0.0, "drop_path": 0.0,
            "fixed_lambda": 1.0, "global_theta": 100.0, "mixed_theta": 10.0,
            "m10_seed_offset": 20_000,
        },
        "training": {
            "effective_batch_size": 128, "micro_batch_size": 16,
            "accumulation_steps": 8, "optimizer": "AdamW", "betas": [0.9, 0.999],
            "epsilon": 1e-8, "weight_decay": 0.05, "base_learning_rate": 3e-4,
            "minimum_learning_rate": 1e-6, "warmup_updates": 400,
            "scheduler_total_updates": 8_000, "loss": "cross_entropy",
            "label_smoothing": 0.0, "gradient_clip_norm": 1.0,
            "sampler": "numpy.random.PCG64", "sampler_seed_offset": 10_000,
            "checkpoint_interval_updates": 1_000,
        },
        "runtime": {
            "device": f"cuda:{device_index}", "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device_index),
            "precision": "float32", "amp": False,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }


def _validate_resume_checkpoints(
    resume_checkpoints: Mapping[str, Path], methods: tuple[str, ...],
) -> dict[str, Path]:
    if not isinstance(resume_checkpoints, Mapping):
        raise CLEVRRunnerError("resume_checkpoints must be an explicit mapping")
    result: dict[str, Path] = {}
    for method, path in resume_checkpoints.items():
        if type(method) is not str or method not in methods or method in result:
            raise CLEVRRunnerError("resume checkpoint methods must be unique requested methods")
        if (not isinstance(path, Path) or not path.is_absolute()
                or not path.is_file()):
            raise CLEVRRunnerError("resume checkpoints must be explicit existing absolute files")
        result[method] = path
    return result


def _checkpoint_update_for_run(
    path: Path, *, artifacts: CLEVRRunArtifacts, target_updates: int,
) -> int:
    match = re.fullmatch(r"update_([0-9]+)\.pt", path.name)
    if match is None:
        raise CLEVRRunnerError("checkpoint filename does not encode its update")
    update = int(match.group(1))
    expected = artifacts.run_dir / "checkpoints" / f"update_{update:04d}.pt"
    try:
        regular = stat.S_ISREG(path.lstat().st_mode)
    except OSError as exc:
        raise CLEVRRunnerError("checkpoint is not an accessible regular file") from exc
    if (not regular or path != expected or not 0 <= update <= target_updates):
        raise CLEVRRunnerError(
            "checkpoint is not bound to the current phase/seed/method/target"
        )
    return update


def _save_boundary(
    *, artifacts: CLEVRRunArtifacts, completed_updates: int, method: str,
    seed: int, model: nn.Module, optimizer: AdamW, scheduler: LambdaLR,
    sampler: CLEVRIndexStream, device_index: int,
) -> None:
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise CLEVRRunnerError("checkpoint boundary requires all gradients to be None")
    save_clevr_checkpoint(
        artifacts.checkpoint_path(completed_updates), method=method,
        run_seed=seed, completed_updates=completed_updates, model=model,
        optimizer=optimizer, scheduler=scheduler, sampler=sampler,
        cuda_devices=(device_index,),
    )


def _publish_evaluation(
    *, artifacts: CLEVRRunArtifacts, result: CLEVREvaluationResult,
    completed_updates: int,
) -> Path:
    path = artifacts.publish_predictions(
        partition=result.partition, completed_updates=completed_updates,
        rows=result.rows,
    )
    _append_evaluation_event(
        artifacts=artifacts, result=result, completed_updates=completed_updates,
        path=path,
    )
    return path


def _append_evaluation_event(
    *, artifacts: CLEVRRunArtifacts, result: CLEVREvaluationResult,
    completed_updates: int, path: Path,
) -> None:
    artifacts.append_event(
        completed_updates=completed_updates, event="evaluation",
        details={
            "partition": result.partition,
            "total_questions": result.total_questions,
            "correct": result.correct,
            "loss_sum": result.loss_sum,
            "cross_entropy": result.cross_entropy,
            "accuracy": result.accuracy,
            "predictions": str(path),
        },
    )


_PREDICTION_FIELDS = {
    "protocol_version", "method", "seed", "question_index", "label", "logits",
}


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CLEVRRunnerError("prediction JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _prediction_path(
    artifacts: CLEVRRunArtifacts, *, partition: str, completed_updates: int,
) -> Path:
    return (artifacts.run_dir / "predictions" / partition /
            f"update_{completed_updates:04d}.jsonl")


def _dataset_label(dataset: object, index: int) -> int:
    sample = dataset[index]
    if not isinstance(sample, Mapping) or "answer" not in sample:
        raise CLEVRRunnerError("evaluation sample does not expose its answer")
    answer = sample["answer"]
    if type(answer) is torch.Tensor and answer.dtype == torch.int64 and answer.shape == ():
        label = int(answer.item())
    elif type(answer) is int:
        label = answer
    else:
        raise CLEVRRunnerError("evaluation answer must be an int64 scalar")
    if label not in (0, 1):
        raise CLEVRRunnerError("evaluation answer must be binary")
    return label


def _read_boundary_evaluation(
    *, path: Path, dataset: object, partition: str, method: str, seed: int,
) -> CLEVREvaluationResult:
    content = _read_regular_utf8(path, "existing prediction artifact")
    lines = content.splitlines()
    if not lines or any(not line for line in lines):
        raise CLEVRRunnerError("existing prediction artifact is empty or has blank rows")
    rows: list[CLEVRPredictionRecord] = []
    try:
        for line in lines:
            document = json.loads(
                line, object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
            if type(document) is not dict or set(document) != _PREDICTION_FIELDS:
                raise CLEVRRunnerError("prediction row fields disagree with the protocol")
            if (document["protocol_version"] != PROTOCOL_VERSION
                    or document["method"] != method or document["seed"] != seed):
                raise CLEVRRunnerError("prediction row protocol, method, or seed disagrees")
            logits = document["logits"]
            if type(logits) is not list:
                raise CLEVRRunnerError("prediction logits must be a JSON list")
            rows.append(CLEVRPredictionRecord(
                document["question_index"], document["label"], tuple(logits),
            ))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, CLEVRRunnerError):
            raise
        raise CLEVRRunnerError("existing prediction artifact is invalid") from exc
    if len(rows) != len(dataset):
        raise CLEVRRunnerError("prediction coverage disagrees with the evaluation dataset")
    question_indices = [row.question_index for row in rows]
    if question_indices != sorted(question_indices) or len(set(question_indices)) != len(rows):
        raise CLEVRRunnerError("prediction question indices must be sorted and unique")
    expected = sorted(
        (_question_index(dataset, index, partition), _dataset_label(dataset, index))
        for index in range(len(dataset))
    )
    if [(row.question_index, row.label) for row in rows] != expected:
        raise CLEVRRunnerError("prediction identities or labels disagree with the dataset")
    loss_sum = 0.0
    correct = 0
    for row in rows:
        high = max(row.logits)
        loss_sum += (high + math.log(sum(math.exp(value - high) for value in row.logits))
                     - row.logits[row.label])
        prediction = 0 if row.logits[0] >= row.logits[1] else 1
        correct += int(prediction == row.label)
    if not math.isfinite(loss_sum):
        raise CLEVRRunnerError("reconstructed prediction loss is not finite")
    total = len(rows)
    return CLEVREvaluationResult(
        partition, total, correct, loss_sum, loss_sum / total, correct / total,
        tuple(rows),
    )


def _record_boundary_evaluation(
    *, artifacts: CLEVRRunArtifacts, model: nn.Module, dataset: object,
    collate_fn: Callable[..., Mapping[str, Tensor]], method: str, partition: str,
    completed_updates: int, seed: int, device: torch.device,
) -> None:
    path = _prediction_path(
        artifacts, partition=partition, completed_updates=completed_updates,
    )
    if path.exists() or path.is_symlink():
        result = _read_boundary_evaluation(
            path=path, dataset=dataset, partition=partition, method=method, seed=seed,
        )
        _append_evaluation_event(
            artifacts=artifacts, result=result, completed_updates=completed_updates,
            path=path,
        )
        return
    result = evaluate_clevr_model(
        model=model, dataset=dataset, collate_fn=collate_fn,
        method=method, partition=partition, device=device,
    )
    _publish_evaluation(
        artifacts=artifacts, result=result, completed_updates=completed_updates,
    )


def _complete_training_boundary(
    *, artifacts: CLEVRRunArtifacts, spec: CLEVRRunSpec, model: nn.Module,
    train_dataset: object, dev_dataset: object,
    collate_fn: Callable[..., Mapping[str, Tensor]], method: str,
    completed_updates: int, seed: int, device: torch.device,
) -> None:
    dev_boundary = (
        (spec.phase == "pilot" and completed_updates == spec.target_updates)
        or (spec.phase == "formal" and completed_updates > 0
            and completed_updates % 1_000 == 0)
    )
    if dev_boundary:
        _record_boundary_evaluation(
            artifacts=artifacts, model=model, dataset=dev_dataset,
            collate_fn=collate_fn, method=method, partition="dev",
            completed_updates=completed_updates, seed=seed, device=device,
        )
    if spec.phase == "formal" and completed_updates == spec.target_updates:
        _record_boundary_evaluation(
            artifacts=artifacts, model=model, dataset=train_dataset,
            collate_fn=collate_fn, method=method, partition="train",
            completed_updates=completed_updates, seed=seed, device=device,
        )


def _append_failure(
    artifacts: CLEVRRunArtifacts, completed_updates: int, primary: BaseException,
) -> None:
    try:
        artifacts.append_event(
            completed_updates=completed_updates, event="failure",
            details={"type": type(primary).__name__, "message": str(primary)},
        )
    except BaseException as logging_failure:
        primary.add_note(
            f"failure journal append failed: {type(logging_failure).__name__}: "
            f"{logging_failure}"
        )


def train_clevr_seed(
    *,
    spec_phase: str,
    seed: int,
    methods: tuple[str, ...],
    audit_dir: Path,
    data_root: Path | None,
    output_root: Path,
    device_index: int,
    resume_checkpoints: Mapping[str, Path],
) -> None:
    """Run one explicitly authorized seed, one sequential method arm at a time."""
    _explicit_output_root(output_root)
    _absolute_directory(audit_dir, "audit_dir")
    audit_artifacts = _audit_artifact_snapshot(audit_dir)
    preflight = preflight_clevr_run(
        spec_phase=spec_phase, seed=seed, methods=methods, audit_dir=audit_dir,
        data_root=data_root, device_index=device_index,
    )
    if _audit_artifact_snapshot(audit_dir) != audit_artifacts:
        raise CLEVRRunnerError("audit artifacts changed while datasets were constructed")
    resumes = _validate_resume_checkpoints(resume_checkpoints, preflight.methods)
    device = _setup_clevr_runtime(device_index)
    factory = CLEVRPairedInitialization(vocab_size=preflight.vocab_size, run_seed=seed)
    common_state = factory.common_state_dict()
    for method in preflight.methods:
        spec = preflight.specs[method]
        resume_path = resumes.get(method)
        train_dataset = preflight.train_datasets[method]
        dev_dataset = preflight.dev_datasets[method]
        expanded_config = _expanded_clevr_config(
            device_index=device_index, audit_dir=audit_dir,
            data_root=None if method == "Q-only" else data_root,
            vocab_size=preflight.vocab_size, train_size=len(train_dataset),
            audit_artifacts=audit_artifacts,
        )
        artifacts = CLEVRRunArtifacts.open(
            output_root=output_root, spec=spec, expanded_config=expanded_config,
            resume=resume_path is not None,
        )
        completed_updates = 0
        primary: BaseException | None = None
        try:
            checkpoint_update = (None if resume_path is None else
                                 _checkpoint_update_for_run(
                                     resume_path, artifacts=artifacts,
                                     target_updates=spec.target_updates,
                                 ))
            artifacts.ensure_common_state(common_state)
            if resume_path is None:
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            model = factory.build(method).to(device)
            optimizer = build_clevr_adamw(model)
            scheduler = build_clevr_lr_scheduler(optimizer)
            sampler = CLEVRIndexStream(dataset_size=len(train_dataset), run_seed=seed)
            collate_fn = (collate_clevr_question_only
                          if method == "Q-only" else collate_clevr_questions)
            if resume_path is None:
                artifacts.append_event(
                    completed_updates=0, event="start",
                    details={"target_updates": spec.target_updates},
                )
                _save_boundary(
                    artifacts=artifacts, completed_updates=0, method=method,
                    seed=seed, model=model, optimizer=optimizer,
                    scheduler=scheduler, sampler=sampler, device_index=device_index,
                )
            else:
                metadata = load_clevr_checkpoint(
                    resume_path, expected_method=method, expected_run_seed=seed,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    sampler=sampler, cuda_devices=(device_index,),
                )
                if (metadata.method != method or metadata.run_seed != seed
                        or metadata.dataset_size != len(train_dataset)
                        or metadata.cuda_devices != (device_index,)
                        or type(metadata.completed_updates) is not int
                        or metadata.completed_updates != checkpoint_update
                        or not 0 <= metadata.completed_updates <= spec.target_updates):
                    raise CLEVRRunnerError(
                        "resume checkpoint metadata disagrees with method/seed/dataset/target"
                    )
                completed_updates = metadata.completed_updates
                artifacts.append_event(
                    completed_updates=completed_updates, event="resume",
                    details={"checkpoint": str(resume_path),
                             "target_updates": spec.target_updates},
                )
            model.train(True)
            if resume_path is not None:
                _complete_training_boundary(
                    artifacts=artifacts, spec=spec, model=model,
                    train_dataset=train_dataset, dev_dataset=dev_dataset,
                    collate_fn=collate_fn, method=method,
                    completed_updates=completed_updates, seed=seed, device=device,
                )
            for update in range(completed_updates + 1, spec.target_updates + 1):
                result = run_clevr_update(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    sampler=sampler, dataset=train_dataset,
                    collate_fn=collate_fn, method=method, device=device,
                )
                completed_updates = update
                artifacts.append_event(
                    completed_updates=update, event="update",
                    details={
                        "dataset_indices": list(result.dataset_indices),
                        "question_indices": list(result.question_indices),
                        "loss_sum": result.loss_sum,
                        "loss_mean": result.loss_mean,
                        "gradient_norm": result.gradient_norm,
                        "learning_rate": result.learning_rate,
                    },
                )
                checkpoint_boundary = update % 1_000 == 0 or update == spec.target_updates
                if checkpoint_boundary:
                    _save_boundary(
                        artifacts=artifacts, completed_updates=update,
                        method=method, seed=seed, model=model,
                        optimizer=optimizer, scheduler=scheduler,
                        sampler=sampler, device_index=device_index,
                    )
                    _complete_training_boundary(
                        artifacts=artifacts, spec=spec, model=model,
                        train_dataset=train_dataset, dev_dataset=dev_dataset,
                        collate_fn=collate_fn, method=method,
                        completed_updates=completed_updates, seed=seed, device=device,
                    )
            artifacts.append_event(
                completed_updates=completed_updates, event="complete",
                details={"target_updates": spec.target_updates},
            )
        except BaseException as exc:
            primary = exc
            _append_failure(artifacts, completed_updates, exc)
            raise
        finally:
            try:
                artifacts.close()
            except BaseException as close_failure:
                if primary is None:
                    raise
                primary.add_note(
                    f"attempt close failed: {type(close_failure).__name__}: {close_failure}"
                )
        del model, optimizer, scheduler, sampler


def evaluate_clevr_checkpoint(
    *,
    spec_phase: str,
    seed: int,
    method: str,
    audit_dir: Path,
    data_root: Path | None,
    output_root: Path,
    device_index: int,
    checkpoint: Path,
    partition: str = "base_test",
) -> Path:
    """Evaluate one explicit formal update-8000 checkpoint on base-test once."""
    _absolute_directory(output_root, "output_root")
    if spec_phase != "formal":
        raise CLEVRRunnerError("base-test evaluation is restricted to the formal phase")
    if partition != "base_test":
        raise CLEVRRunnerError("only the base_test partition is supported")
    _absolute_directory(audit_dir, "audit_dir")
    audit_artifacts = _audit_artifact_snapshot(audit_dir)
    ordered, specs = _validate_request(
        spec_phase=spec_phase, seed=seed, methods=(method,),
        audit_dir=audit_dir, data_root=data_root, device_index=device_index,
    )
    method = ordered[0]
    if (not isinstance(checkpoint, Path) or not checkpoint.is_absolute()
            or not checkpoint.is_file()):
        raise CLEVRRunnerError("checkpoint must be an explicit existing absolute file")
    canonical = (output_root / "formal" / f"seed_{seed}" / method /
                 "predictions" / "base_test" / f"{method}_seed_{seed}.jsonl")
    if canonical.exists() or canonical.is_symlink():
        raise CLEVRRunnerError("canonical base-test prediction output already exists")
    if method == "Q-only":
        train_dataset = CLEVRQuestionOnlyDataset(
            audit_dir=audit_dir, partition="train",
        )
        collate_fn = collate_clevr_question_only
    else:
        assert data_root is not None
        train_dataset = CLEVRQuestionDataset(
            data_root=data_root, audit_dir=audit_dir, partition="train",
        )
        collate_fn = collate_clevr_questions
    if _audit_artifact_snapshot(audit_dir) != audit_artifacts:
        raise CLEVRRunnerError("audit artifacts changed while datasets were constructed")
    device = _setup_clevr_runtime(device_index)
    expanded_config = _expanded_clevr_config(
        device_index=device_index, audit_dir=audit_dir,
        data_root=None if method == "Q-only" else data_root,
        vocab_size=train_dataset.vocab_size, train_size=len(train_dataset),
        audit_artifacts=audit_artifacts,
    )
    factory = CLEVRPairedInitialization(
        vocab_size=train_dataset.vocab_size, run_seed=seed,
    )
    spec = specs[method]
    artifacts = CLEVRRunArtifacts.open(
        output_root=output_root, spec=spec, expanded_config=expanded_config,
        resume=True,
    )
    completed_updates = 0
    primary: BaseException | None = None
    try:
        checkpoint_update = _checkpoint_update_for_run(
            checkpoint, artifacts=artifacts, target_updates=spec.target_updates,
        )
        if checkpoint_update != spec.target_updates:
            raise CLEVRRunnerError("formal evaluation requires the final checkpoint")
        artifacts.ensure_common_state(factory.common_state_dict())
        model = factory.build(method).to(device)
        optimizer = build_clevr_adamw(model)
        scheduler = build_clevr_lr_scheduler(optimizer)
        sampler = CLEVRIndexStream(dataset_size=len(train_dataset), run_seed=seed)
        metadata = load_clevr_checkpoint(
            checkpoint, expected_method=method, expected_run_seed=seed,
            model=model, optimizer=optimizer, scheduler=scheduler,
            sampler=sampler, cuda_devices=(device_index,),
        )
        if (metadata.method != method or metadata.run_seed != seed
                or metadata.completed_updates != checkpoint_update
                or metadata.dataset_size != len(train_dataset)
                or metadata.cuda_devices != (device_index,)):
            raise CLEVRRunnerError(
                "formal evaluation requires the matching update-8000 checkpoint"
            )
        completed_updates = checkpoint_update
        if method == "Q-only":
            base_test = CLEVRQuestionOnlyDataset(
                audit_dir=audit_dir, partition="base_test",
            )
        else:
            assert data_root is not None
            base_test = CLEVRQuestionDataset(
                data_root=data_root, audit_dir=audit_dir, partition="base_test",
            )
        if _audit_artifact_snapshot(audit_dir) != audit_artifacts:
            raise CLEVRRunnerError("audit artifacts changed before base-test evaluation")
        if (base_test.vocab_size != train_dataset.vocab_size or len(base_test) == 0):
            raise CLEVRRunnerError("base-test vocabulary or size is invalid")
        result = evaluate_clevr_model(
            model=model, dataset=base_test, collate_fn=collate_fn,
            method=method, partition="base_test", device=device,
        )
        path = _publish_evaluation(
            artifacts=artifacts, result=result, completed_updates=8_000,
        )
        artifacts.append_event(
            completed_updates=8_000, event="complete",
            details={"partition": "base_test", "checkpoint": str(checkpoint)},
        )
        return path
    except BaseException as exc:
        primary = exc
        _append_failure(artifacts, completed_updates, exc)
        raise
    finally:
        try:
            artifacts.close()
        except BaseException as close_failure:
            if primary is None:
                raise
            primary.add_note(
                f"attempt close failed: {type(close_failure).__name__}: {close_failure}"
            )


def _require_method(method: str) -> None:
    if type(method) is not str or method not in ("G", "B", "M10", "Q-only"):
        raise CLEVRRunnerError("method must be G, B, M10, or Q-only")


def _question_index(dataset: object, index: int, partition: str) -> int:
    identity = dataset.sample_identity(index)
    if (not isinstance(identity, CLEVRSampleIdentity)
            or identity.partition != partition
            or type(identity.question_index) is not int or identity.question_index < 0):
        raise CLEVRRunnerError("sample identity has an invalid partition or question_index")
    return identity.question_index


def _model_batch(
    batch: Mapping[str, Tensor], *, size: int, method: str, device: torch.device,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    fields = {"question_tokens", "question_length", "answer"}
    if method != "Q-only":
        fields.add("image")
    if not isinstance(batch, Mapping) or set(batch) != fields:
        raise CLEVRRunnerError("batch keys disagree with the method's model contract")
    if any(not isinstance(value, Tensor) for value in batch.values()):
        raise CLEVRRunnerError("batch values must be tensors")
    tokens, lengths, labels = (batch[key] for key in ("question_tokens", "question_length", "answer"))
    if (tokens.dtype != torch.int64 or tokens.ndim != 2
            or tokens.shape[0] != size or tokens.shape[1] < 2):
        raise CLEVRRunnerError("question_tokens must be int64 [N,T], T >= 2")
    if (lengths.dtype != torch.int64 or lengths.shape != (size,)
            or bool((lengths < 2).any()) or bool((lengths > tokens.shape[1]).any())):
        raise CLEVRRunnerError("question_length must be int64 [N] within token bounds")
    if (labels.dtype != torch.int64 or labels.shape != (size,)
            or bool(((labels != 0) & (labels != 1)).any())):
        raise CLEVRRunnerError("answer must be int64 [N] with binary labels")
    image = None
    if method != "Q-only":
        image = batch["image"]
        if (image.dtype != torch.float32 or image.shape != (size, 3, 224, 224)
                or not bool(torch.isfinite(image).all())):
            raise CLEVRRunnerError("image must be finite float32 [N,3,224,224]")
        image = image.to(device)
    return image, tokens.to(device), lengths.to(device), labels.to(device)


def _logits(model: nn.Module, image: Tensor | None, tokens: Tensor, lengths: Tensor) -> Tensor:
    logits = model(image, tokens, lengths)
    if (not isinstance(logits, Tensor) or logits.dtype != torch.float32
            or logits.shape != (len(tokens), 2) or not bool(torch.isfinite(logits).all())):
        raise CLEVRRunnerError("logits must be finite float32 [N,2]")
    return logits


def _finite_scalar(value: Tensor, name: str) -> float:
    result = float(value.detach().item())
    if not math.isfinite(result):
        raise CLEVRRunnerError(f"{name} must be finite")
    return result


def run_clevr_update(
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    dataset: object,
    collate_fn: Callable[[Sequence[Mapping[str, Tensor]]], Mapping[str, Tensor]],
    method: str,
    device: torch.device,
) -> CLEVRUpdateResult:
    """Accumulate eight contiguous micro-batches, then commit exactly 128 indices.

    The caller owns model mode and device placement. No checkpoint, journal,
    retry, or rollback is attempted, even if a dependency raises after mutation.
    """
    optimizer.zero_grad(set_to_none=True)
    _require_method(method)
    if len(optimizer.param_groups) != 2:
        raise CLEVRRunnerError("optimizer must expose exactly two parameter groups")
    rates = [group["lr"] for group in optimizer.param_groups]
    if (any(type(rate) not in (int, float) or not math.isfinite(rate) for rate in rates)
            or rates[0] != rates[1]):
        raise CLEVRRunnerError("optimizer groups must share one finite learning rate")
    learning_rate = float(rates[0])
    if len(dataset) != sampler.dataset_size:
        raise CLEVRRunnerError("dataset size disagrees with sampler")
    indices = sampler.preview(128)
    if (type(indices) is not tuple or len(indices) != 128
            or any(type(index) is not int or not 0 <= index < len(dataset) for index in indices)):
        raise CLEVRRunnerError("sampler preview must contain 128 valid dataset indices")
    question_indices = tuple(_question_index(dataset, index, "train") for index in indices)
    loss_sum = 0.0
    for offset in range(0, 128, 16):
        samples = [dataset[index] for index in indices[offset:offset + 16]]
        image, tokens, lengths, labels = _model_batch(
            collate_fn(samples), size=16, method=method, device=device,
        )
        logits = _logits(model, image, tokens, lengths)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        loss_sum += _finite_scalar(loss, "loss") * 16
        (loss / 8).backward()
    if not math.isfinite(loss_sum):
        raise CLEVRRunnerError("accumulated loss must be finite")
    try:
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise CLEVRRunnerError("gradient clipping failed") from exc
    gradient_norm = _finite_scalar(norm, "gradient norm")
    # Validate and construct all metadata before the irreversible step sequence.
    result = CLEVRUpdateResult(indices, question_indices, loss_sum, loss_sum / 128,
                               gradient_norm, learning_rate)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    sampler.commit(indices)
    return result


def evaluate_clevr_model(
    *,
    model: nn.Module,
    dataset: object,
    collate_fn: Callable[..., Mapping[str, Tensor]],
    method: str,
    partition: str,
    device: torch.device,
) -> CLEVREvaluationResult:
    """Return complete question-weighted metrics and sorted CPU prediction rows.

    This seam never publishes files or chooses checkpoints. The caller must
    separately authorize and assemble any production evaluation.
    """
    was_training = model.training
    try:
        model.eval()
        _require_method(method)
        if type(partition) is not str or partition not in ("train", "dev", "base_test"):
            raise CLEVRRunnerError("evaluation partition must be train, dev, or base_test")
        size = len(dataset)
        if size == 0:
            raise CLEVRRunnerError("evaluation dataset must not be empty")
        seen: set[int] = set()
        rows: list[CLEVRPredictionRecord] = []
        loss_sum = 0.0
        correct = 0
        with torch.inference_mode():
            for offset in range(0, size, 16):
                indices = range(offset, min(offset + 16, size))
                question_indices = []
                for index in indices:
                    question_index = _question_index(dataset, index, partition)
                    if question_index in seen:
                        raise CLEVRRunnerError("duplicate evaluation question_index")
                    seen.add(question_index)
                    question_indices.append(question_index)
                samples = [dataset[index] for index in indices]
                image, tokens, lengths, labels = _model_batch(
                    collate_fn(samples), size=len(indices), method=method, device=device,
                )
                logits = _logits(model, image, tokens, lengths)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                loss_sum += _finite_scalar(loss, "evaluation loss")
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                cpu_logits = logits.detach().cpu().tolist()
                cpu_labels = labels.detach().cpu().tolist()
                rows.extend(CLEVRPredictionRecord(question_index, label, tuple(values))
                            for question_index, label, values in zip(question_indices, cpu_labels, cpu_logits))
        if not math.isfinite(loss_sum) or len(rows) != size:
            raise CLEVRRunnerError("evaluation metrics or row coverage are incomplete")
        return CLEVREvaluationResult(
            partition, size, correct, loss_sum, loss_sum / size, correct / size,
            tuple(sorted(rows, key=lambda row: row.question_index)),
        )
    finally:
        model.train(was_training)
