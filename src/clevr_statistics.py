"""Pure statistics primitives for the frozen CLEVR PUB-R1.1 protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np


METHODS = ("G", "B", "M10", "Q-only")
SEEDS = (101, 102, 103, 104, 105)
RELATIONS = ("left", "right", "front", "behind")


def _require_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_label(value: object) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError("label must use no=0 / yes=1")
    return value


def _require_finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite real number")
    return normalized


def _require_method_seed(method: object, seed: object) -> tuple[str, int]:
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("method is not part of the frozen formal comparison")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed is not part of the frozen formal comparison")
    return method, seed


@dataclass(frozen=True, slots=True)
class TargetRow:
    partition: str
    official_split: str
    image_index: int
    image_filename: str
    question_index: int
    relation: str
    label: int

    def __post_init__(self) -> None:
        allowed_pairs = {("train", "train"), ("base_test", "val")}
        if (self.partition, self.official_split) not in allowed_pairs:
            raise ValueError(
                "partition and official_split must be train/train or base_test/val"
            )
        _require_nonnegative_int(self.image_index, "image_index")
        _require_nonnegative_int(self.question_index, "question_index")
        expected_filename = (
            f"CLEVR_{self.official_split}_{self.image_index:06d}.png"
        )
        if self.image_filename != expected_filename:
            raise ValueError(
                "image_filename must match official_split and image_index"
            )
        if self.relation not in RELATIONS:
            raise ValueError("relation is not part of the frozen task")
        _require_label(self.label)


@dataclass(frozen=True, slots=True)
class PredictionRow:
    question_index: int
    label: int
    logits: tuple[float, float]

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.question_index, "question_index")
        _require_label(self.label)
        if not isinstance(self.logits, tuple):
            raise ValueError("logits must be an immutable tuple")
        if len(self.logits) != 2:
            raise ValueError("logits must contain exactly two values")
        normalized = tuple(
            _require_finite_real(value, f"logits[{index}]")
            for index, value in enumerate(self.logits)
        )
        object.__setattr__(self, "logits", normalized)


@dataclass(frozen=True, slots=True)
class RunPredictions:
    method: str
    seed: int
    rows: tuple[PredictionRow, ...]

    def __post_init__(self) -> None:
        _require_method_seed(self.method, self.seed)
        if not isinstance(self.rows, tuple):
            raise ValueError("rows must be an immutable tuple")
        if not self.rows:
            raise ValueError("rows must not be empty")
        if any(not isinstance(row, PredictionRow) for row in self.rows):
            raise ValueError("rows must contain PredictionRow records")


@dataclass(frozen=True, slots=True)
class CellMetrics:
    relation: str
    label: int
    count: int
    correct_count: int
    accuracy: float | None

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError("cell relation is not part of the frozen task")
        _require_label(self.label)
        _require_nonnegative_int(self.count, "cell count")
        _require_nonnegative_int(self.correct_count, "cell correct_count")
        if self.correct_count > self.count:
            raise ValueError("cell correct_count cannot exceed count")
        expected = None if self.count == 0 else self.correct_count / self.count
        if expected is None:
            if self.accuracy is not None:
                raise ValueError("empty cell accuracy must be None")
            return
        accuracy = _require_finite_real(self.accuracy, "cell accuracy")
        if accuracy != expected:
            raise ValueError("cell accuracy does not match its counts")
        object.__setattr__(self, "accuracy", accuracy)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    question_count: int
    correct_count: int
    accuracy: float
    cross_entropy: float | None
    cells: tuple[CellMetrics, ...]
    macro_accuracy: float | None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.question_count, "question_count")
        _require_nonnegative_int(self.correct_count, "correct_count")
        if self.question_count == 0:
            raise ValueError("question_count must be positive")
        if self.correct_count > self.question_count:
            raise ValueError("correct_count cannot exceed question_count")
        accuracy = _require_finite_real(self.accuracy, "accuracy")
        if accuracy != self.correct_count / self.question_count:
            raise ValueError("accuracy does not match its counts")
        object.__setattr__(self, "accuracy", accuracy)

        if self.cross_entropy is not None:
            cross_entropy = _require_finite_real(
                self.cross_entropy, "cross_entropy"
            )
            if cross_entropy < 0:
                raise ValueError("cross_entropy must be non-negative")
            object.__setattr__(self, "cross_entropy", cross_entropy)

        if not isinstance(self.cells, tuple):
            raise ValueError("cells must be an immutable tuple")
        expected_keys = tuple(
            (relation, label) for relation in RELATIONS for label in (0, 1)
        )
        observed_keys = tuple(
            (cell.relation, cell.label)
            for cell in self.cells
            if isinstance(cell, CellMetrics)
        )
        if len(observed_keys) != len(self.cells) or observed_keys != expected_keys:
            raise ValueError("cells must contain the fixed eight relation-answer cells")
        if sum(cell.count for cell in self.cells) != self.question_count:
            raise ValueError("cell counts do not sum to question_count")
        if sum(cell.correct_count for cell in self.cells) != self.correct_count:
            raise ValueError("cell correct counts do not sum to correct_count")

        cell_accuracies = tuple(cell.accuracy for cell in self.cells)
        expected_macro = (
            None
            if any(value is None for value in cell_accuracies)
            else sum(value for value in cell_accuracies if value is not None) / 8
        )
        if expected_macro is None:
            if self.macro_accuracy is not None:
                raise ValueError("macro_accuracy must be None when a cell is empty")
        else:
            macro_accuracy = _require_finite_real(
                self.macro_accuracy, "macro_accuracy"
            )
            if macro_accuracy != expected_macro:
                raise ValueError("macro_accuracy does not equal the eight-cell mean")
            object.__setattr__(self, "macro_accuracy", macro_accuracy)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    method: str
    seed: int
    metrics: ClassificationMetrics

    def __post_init__(self) -> None:
        _require_method_seed(self.method, self.seed)
        if not isinstance(self.metrics, ClassificationMetrics):
            raise ValueError("metrics must be ClassificationMetrics")


def _validated_targets(
    targets: Sequence[TargetRow],
) -> tuple[tuple[TargetRow, ...], tuple[TargetRow, ...]]:
    if not isinstance(targets, Sequence):
        raise ValueError("targets must be a sequence of TargetRow records")
    seen: set[tuple[str, int]] = set()
    train: list[TargetRow] = []
    base_test: list[TargetRow] = []
    for target in targets:
        if not isinstance(target, TargetRow):
            raise ValueError("targets must contain TargetRow records")
        key = (target.partition, target.question_index)
        if key in seen:
            raise ValueError(f"duplicate target ID in {target.partition}: {key[1]}")
        seen.add(key)
        (train if target.partition == "train" else base_test).append(target)
    if not train:
        raise ValueError("authoritative train targets must not be empty")
    if not base_test:
        raise ValueError("authoritative base_test targets must not be empty")
    return (
        tuple(sorted(train, key=lambda row: row.question_index)),
        tuple(sorted(base_test, key=lambda row: row.question_index)),
    )


def _classification_metrics(
    targets: tuple[TargetRow, ...],
    predictions: Mapping[int, int],
    losses: Mapping[int, float] | None,
) -> ClassificationMetrics:
    cell_counts = {
        (relation, label): [0, 0]
        for relation in RELATIONS
        for label in (0, 1)
    }
    correct_count = 0
    for target in targets:
        predicted = predictions[target.question_index]
        is_correct = int(predicted == target.label)
        correct_count += is_correct
        counts = cell_counts[(target.relation, target.label)]
        counts[0] += 1
        counts[1] += is_correct

    cells = tuple(
        CellMetrics(
            relation=relation,
            label=label,
            count=cell_counts[(relation, label)][0],
            correct_count=cell_counts[(relation, label)][1],
            accuracy=(
                None
                if cell_counts[(relation, label)][0] == 0
                else cell_counts[(relation, label)][1]
                / cell_counts[(relation, label)][0]
            ),
        )
        for relation in RELATIONS
        for label in (0, 1)
    )
    cell_accuracies = tuple(cell.accuracy for cell in cells)
    macro_accuracy = (
        None
        if any(value is None for value in cell_accuracies)
        else sum(value for value in cell_accuracies if value is not None) / 8
    )
    cross_entropy = (
        None
        if losses is None
        else math.fsum(losses[target.question_index] for target in targets)
        / len(targets)
    )
    return ClassificationMetrics(
        question_count=len(targets),
        correct_count=correct_count,
        accuracy=correct_count / len(targets),
        cross_entropy=cross_entropy,
        cells=cells,
        macro_accuracy=macro_accuracy,
    )


def evaluate_run(
    targets: Sequence[TargetRow], run: RunPredictions
) -> RunMetrics:
    """Recompute one formal run against authoritative manifest targets."""
    if not isinstance(run, RunPredictions):
        raise ValueError("run must be RunPredictions")
    _, base_test = _validated_targets(targets)
    target_by_id = {target.question_index: target for target in base_test}

    prediction_by_id: dict[int, PredictionRow] = {}
    for prediction in run.rows:
        if prediction.question_index in prediction_by_id:
            raise ValueError(
                f"duplicate prediction ID: {prediction.question_index}"
            )
        prediction_by_id[prediction.question_index] = prediction

    target_ids = set(target_by_id)
    prediction_ids = set(prediction_by_id)
    missing = sorted(target_ids - prediction_ids)
    extra = sorted(prediction_ids - target_ids)
    if missing:
        raise ValueError(f"missing prediction IDs: {missing}")
    if extra:
        raise ValueError(f"extra prediction IDs: {extra}")

    predictions: dict[int, int] = {}
    losses: dict[int, float] = {}
    for question_index in sorted(target_by_id):
        target = target_by_id[question_index]
        prediction = prediction_by_id[question_index]
        if prediction.label != target.label:
            raise ValueError(
                f"prediction label mismatch for question {question_index}"
            )
        logit_0, logit_1 = prediction.logits
        predictions[question_index] = int(logit_1 > logit_0)
        target_logit = prediction.logits[target.label]
        other_logit = prediction.logits[1 - target.label]
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(np.logaddexp(0.0, other_logit - target_logit))
        if not math.isfinite(loss):
            raise ValueError(
                f"cross_entropy is not finite for question {question_index}"
            )
        losses[question_index] = loss

    return RunMetrics(
        method=run.method,
        seed=run.seed,
        metrics=_classification_metrics(base_test, predictions, losses),
    )
