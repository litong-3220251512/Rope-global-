"""Pure statistics primitives for the frozen CLEVR PUB-R1.1 protocol."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.random import Generator, PCG64


METHODS = ("G", "B", "M10", "Q-only")
SEEDS = (101, 102, 103, 104, 105)
RELATIONS = ("left", "right", "front", "behind")
T4_975 = 2.7764451051977987
BOOTSTRAP_SEED = 31001
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_QUANTILE_METHOD = "linear"


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


def _validate_summary_fields(
    name: object,
    seeds: object,
    values: object,
    mean: object,
    sample_sd: object,
    interval: object,
) -> tuple[str, tuple[int, ...], tuple[float, ...], float, float, tuple[float, float]]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("summary name must be a non-empty string")
    if not isinstance(seeds, tuple) or seeds != SEEDS:
        raise ValueError("summary seeds must be the formal seed tuple")
    if not isinstance(values, tuple) or len(values) != len(SEEDS):
        raise ValueError("summary values must be a five-value tuple")
    normalized_values = tuple(
        _require_finite_real(value, f"values[{index}]")
        for index, value in enumerate(values)
    )
    normalized_mean = _require_finite_real(mean, "summary mean")
    normalized_sd = _require_finite_real(sample_sd, "sample_sd")
    if normalized_sd < 0:
        raise ValueError("sample_sd must be non-negative")
    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ValueError("interval must be a two-value tuple")
    normalized_interval = tuple(
        _require_finite_real(value, f"interval[{index}]")
        for index, value in enumerate(interval)
    )
    expected_mean = statistics.fmean(normalized_values)
    expected_sd = statistics.stdev(normalized_values)
    half_width = T4_975 * expected_sd / math.sqrt(len(SEEDS))
    expected_interval = (expected_mean - half_width, expected_mean + half_width)
    if normalized_mean != expected_mean:
        raise ValueError("summary mean does not match values")
    if normalized_sd != expected_sd:
        raise ValueError("sample_sd does not match values")
    if normalized_interval != expected_interval:
        raise ValueError("interval does not match values")
    return (
        name,
        seeds,
        normalized_values,
        normalized_mean,
        normalized_sd,
        normalized_interval,
    )


@dataclass(frozen=True, slots=True)
class SeedSummary:
    name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    sample_sd: float
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        validated = _validate_summary_fields(
            self.name,
            self.seeds,
            self.values,
            self.mean,
            self.sample_sd,
            self.interval,
        )
        for field, value in zip(
            ("name", "seeds", "values", "mean", "sample_sd", "interval"),
            validated,
            strict=True,
        ):
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class PairedSummary:
    name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    sample_sd: float
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        validated = _validate_summary_fields(
            self.name,
            self.seeds,
            self.values,
            self.mean,
            self.sample_sd,
            self.interval,
        )
        for field, value in zip(
            ("name", "seeds", "values", "mean", "sample_sd", "interval"),
            validated,
            strict=True,
        ):
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class ClusterBootstrapSummary:
    seed: int
    draws: int
    quantile_method: str
    replicates: tuple[float, ...]
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed != BOOTSTRAP_SEED:
            raise ValueError("bootstrap seed must be 31001")
        if type(self.draws) is not int or self.draws != BOOTSTRAP_DRAWS:
            raise ValueError("bootstrap draws must be 2000")
        if self.quantile_method != BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("bootstrap quantile method must be linear")
        if not isinstance(self.replicates, tuple) or len(self.replicates) != self.draws:
            raise ValueError("bootstrap replicates must be a 2000-value tuple")
        normalized_replicates = tuple(
            _require_finite_real(value, f"replicates[{index}]")
            for index, value in enumerate(self.replicates)
        )
        if any(value < -1 or value > 1 for value in normalized_replicates):
            raise ValueError("bootstrap B-G differences must lie in [-1, 1]")
        if not isinstance(self.interval, tuple) or len(self.interval) != 2:
            raise ValueError("bootstrap interval must be a two-value tuple")
        normalized_interval = tuple(
            _require_finite_real(value, f"bootstrap interval[{index}]")
            for index, value in enumerate(self.interval)
        )
        expected = np.quantile(
            np.asarray(normalized_replicates, dtype=np.float64),
            (0.025, 0.975),
            method=BOOTSTRAP_QUANTILE_METHOD,
        )
        expected_interval = (float(expected[0]), float(expected[1]))
        if normalized_interval != expected_interval:
            raise ValueError("bootstrap interval does not match its replicates")
        object.__setattr__(self, "replicates", normalized_replicates)
        object.__setattr__(self, "interval", normalized_interval)


@dataclass(frozen=True, slots=True)
class CLEVRStatistics:
    run_metrics: tuple[RunMetrics, ...]
    method_summaries: tuple[SeedSummary, ...]
    b_minus_g: PairedSummary
    m10_minus_g: PairedSummary
    train_majority: ClassificationMetrics
    bootstrap: ClusterBootstrapSummary

    def __post_init__(self) -> None:
        expected_keys = tuple((method, seed) for method in METHODS for seed in SEEDS)
        if not isinstance(self.run_metrics, tuple) or tuple(
            (run.method, run.seed)
            for run in self.run_metrics
            if isinstance(run, RunMetrics)
        ) != expected_keys:
            raise ValueError("run_metrics must contain the ordered formal run grid")
        expected_names = tuple(f"{method} accuracy" for method in METHODS)
        if not isinstance(self.method_summaries, tuple) or tuple(
            summary.name
            for summary in self.method_summaries
            if isinstance(summary, SeedSummary)
        ) != expected_names:
            raise ValueError("method_summaries must contain the four ordered methods")
        accuracy_by_key = {
            (run.method, run.seed): run.metrics.accuracy for run in self.run_metrics
        }
        for method, summary in zip(
            METHODS, self.method_summaries, strict=True
        ):
            expected_values = tuple(
                accuracy_by_key[(method, seed)] for seed in SEEDS
            )
            if summary.values != expected_values:
                raise ValueError("method summary values do not match run metrics")
        if not isinstance(self.b_minus_g, PairedSummary) or self.b_minus_g.name != "B-G":
            raise ValueError("b_minus_g must be the B-G paired summary")
        if not isinstance(self.m10_minus_g, PairedSummary) or self.m10_minus_g.name != "M10-G":
            raise ValueError("m10_minus_g must be the M10-G paired summary")
        expected_b_g = tuple(
            accuracy_by_key[("B", seed)] - accuracy_by_key[("G", seed)]
            for seed in SEEDS
        )
        expected_m10_g = tuple(
            accuracy_by_key[("M10", seed)] - accuracy_by_key[("G", seed)]
            for seed in SEEDS
        )
        if self.b_minus_g.values != expected_b_g:
            raise ValueError("B-G values do not match run metrics")
        if self.m10_minus_g.values != expected_m10_g:
            raise ValueError("M10-G values do not match run metrics")
        if not isinstance(self.train_majority, ClassificationMetrics):
            raise ValueError("train_majority must be ClassificationMetrics")
        if self.train_majority.cross_entropy is not None:
            raise ValueError("train_majority cross_entropy must be None")
        if not isinstance(self.bootstrap, ClusterBootstrapSummary):
            raise ValueError("bootstrap must be ClusterBootstrapSummary")


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


def evaluate_train_majority(
    targets: Sequence[TargetRow],
) -> ClassificationMetrics:
    """Evaluate the no-training baseline chosen only from train labels."""
    train, base_test = _validated_targets(targets)
    predicted_label = int(sum(target.label for target in train) > len(train) / 2)
    predictions = {
        target.question_index: predicted_label for target in base_test
    }
    return _classification_metrics(base_test, predictions, losses=None)


def _ordered_seed_values(values: Mapping[int, float]) -> tuple[float, ...]:
    if not isinstance(values, Mapping):
        raise ValueError("seed values must be a mapping")
    if any(type(seed) is not int for seed in values) or set(values) != set(SEEDS):
        raise ValueError("seed values must contain exactly the five formal seeds")
    return tuple(
        _require_finite_real(values[seed], f"value for seed {seed}")
        for seed in SEEDS
    )


def _summary_numbers(
    values: tuple[float, ...],
) -> tuple[float, float, tuple[float, float]]:
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    half_width = T4_975 * sample_sd / math.sqrt(len(SEEDS))
    return mean, sample_sd, (mean - half_width, mean + half_width)


def summarize_seed_values(
    name: str, values: Mapping[int, float]
) -> SeedSummary:
    """Summarize exactly the five formal seeds with a sample-SD t interval."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("summary name must be a non-empty string")
    ordered = _ordered_seed_values(values)
    mean, sample_sd, interval = _summary_numbers(ordered)
    return SeedSummary(name, SEEDS, ordered, mean, sample_sd, interval)


def summarize_paired_differences(
    name: str,
    minuend: Mapping[int, float],
    subtrahend: Mapping[int, float],
) -> PairedSummary:
    """Subtract corresponding formal seeds before summarizing the contrast."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("summary name must be a non-empty string")
    minuend_values = _ordered_seed_values(minuend)
    subtrahend_values = _ordered_seed_values(subtrahend)
    differences = tuple(
        _require_finite_real(
            first - second, f"paired difference for seed {seed}"
        )
        for seed, first, second in zip(
            SEEDS, minuend_values, subtrahend_values, strict=True
        )
    )
    mean, sample_sd, interval = _summary_numbers(differences)
    return PairedSummary(name, SEEDS, differences, mean, sample_sd, interval)


def _validated_formal_runs(
    runs: Sequence[RunPredictions],
) -> tuple[RunPredictions, ...]:
    if not isinstance(runs, Sequence):
        raise ValueError("runs must be a sequence of RunPredictions")
    by_key: dict[tuple[str, int], RunPredictions] = {}
    for run in runs:
        if not isinstance(run, RunPredictions):
            raise ValueError("runs must contain RunPredictions records")
        key = (run.method, run.seed)
        if key in by_key:
            raise ValueError(f"duplicate formal run: {key}")
        by_key[key] = run
    expected_keys = tuple((method, seed) for method in METHODS for seed in SEEDS)
    missing = tuple(key for key in expected_keys if key not in by_key)
    if missing:
        raise ValueError(f"missing formal runs: {missing}")
    if len(by_key) != len(expected_keys):
        extra = tuple(key for key in by_key if key not in set(expected_keys))
        raise ValueError(f"extra formal runs: {extra}")
    return tuple(by_key[key] for key in expected_keys)


def _cluster_bootstrap(
    base_test: tuple[TargetRow, ...],
    runs_by_key: Mapping[tuple[str, int], RunPredictions],
) -> ClusterBootstrapSummary:
    image_keys = tuple(
        sorted({(target.image_index, target.image_filename) for target in base_test})
    )
    image_positions = {key: index for index, key in enumerate(image_keys)}
    question_counts = np.zeros(len(image_keys), dtype=np.int64)
    target_by_id = {target.question_index: target for target in base_test}
    for target in base_test:
        question_counts[
            image_positions[(target.image_index, target.image_filename)]
        ] += 1

    correct_by_method = {
        method: np.zeros((len(SEEDS), len(image_keys)), dtype=np.int64)
        for method in ("G", "B")
    }
    for method in ("G", "B"):
        for seed_position, seed in enumerate(SEEDS):
            run = runs_by_key[(method, seed)]
            for prediction in run.rows:
                target = target_by_id[prediction.question_index]
                predicted = int(prediction.logits[1] > prediction.logits[0])
                correct_by_method[method][
                    seed_position,
                    image_positions[(target.image_index, target.image_filename)],
                ] += int(predicted == target.label)

    image_count = len(image_keys)
    probabilities = np.full(image_count, 1.0 / image_count, dtype=np.float64)
    weights = Generator(PCG64(BOOTSTRAP_SEED)).multinomial(
        image_count,
        probabilities,
        size=BOOTSTRAP_DRAWS,
    )
    denominators = weights @ question_counts
    g_accuracy = (weights @ correct_by_method["G"].T) / denominators[:, None]
    b_accuracy = (weights @ correct_by_method["B"].T) / denominators[:, None]
    replicate_array = np.mean(b_accuracy - g_accuracy, axis=1)
    replicates = tuple(float(value) for value in replicate_array)
    quantiles = np.quantile(
        replicate_array,
        (0.025, 0.975),
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    return ClusterBootstrapSummary(
        seed=BOOTSTRAP_SEED,
        draws=BOOTSTRAP_DRAWS,
        quantile_method=BOOTSTRAP_QUANTILE_METHOD,
        replicates=replicates,
        interval=(float(quantiles[0]), float(quantiles[1])),
    )


def compute_clevr_statistics(
    targets: Sequence[TargetRow],
    runs: Sequence[RunPredictions],
) -> CLEVRStatistics:
    """Compute the complete frozen PUB-R1.1 formal statistics object."""
    _, base_test = _validated_targets(targets)
    ordered_runs = _validated_formal_runs(runs)
    run_metrics = tuple(evaluate_run(targets, run) for run in ordered_runs)
    accuracy_by_method = {
        method: {
            run.seed: metric.metrics.accuracy
            for run, metric in zip(ordered_runs, run_metrics, strict=True)
            if run.method == method
        }
        for method in METHODS
    }
    method_summaries = tuple(
        summarize_seed_values(
            f"{method} accuracy", accuracy_by_method[method]
        )
        for method in METHODS
    )
    runs_by_key = {(run.method, run.seed): run for run in ordered_runs}
    return CLEVRStatistics(
        run_metrics=run_metrics,
        method_summaries=method_summaries,
        b_minus_g=summarize_paired_differences(
            "B-G", accuracy_by_method["B"], accuracy_by_method["G"]
        ),
        m10_minus_g=summarize_paired_differences(
            "M10-G", accuracy_by_method["M10"], accuracy_by_method["G"]
        ),
        train_majority=evaluate_train_majority(targets),
        bootstrap=_cluster_bootstrap(base_test, runs_by_key),
    )
