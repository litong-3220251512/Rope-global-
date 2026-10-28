"""Strict file boundary for CLEVR PUB-R1.1 statistics reporting."""

from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from src.clevr_partitions import ImageIdentity
from src.clevr_data_contract import AUDIT_ARTIFACT_NAMES, run_data_contract
from src.clevr_statistics import (
    CLEVRStatistics,
    METHODS,
    SEEDS,
    CellMetrics,
    ClassificationMetrics,
    ClusterBootstrapSummary,
    PairedSummary,
    PredictionRow,
    RunPredictions,
    RunMetrics,
    SeedSummary,
    TargetRow,
    compute_clevr_statistics,
)


PROTOCOL_VERSION = "PUB-R1.1"

_PARTITIONS = ("train", "dev", "base_test", "confirmation")
_TARGET_PARTITIONS = ("train", "base_test")
_PARTITION_SPLITS = {
    "train": "train",
    "dev": "train",
    "base_test": "val",
    "confirmation": "val",
}
_RELATIONS = ("left", "right", "front", "behind")
_IMAGE_DOCUMENT_KEYS = ("protocol_version", "partitions")
_IMAGE_KEYS = ("official_split", "image_index", "image_filename")
_REPORT_KEYS = (
    "protocol_version",
    "environment",
    "partition_summaries",
    "cell_counts",
    "exclusion_counts",
    "source_summaries",
    "file_summaries",
)
_SUMMARY_KEYS = (
    "partition",
    "mother_image_count",
    "selected_image_count",
    "selected_question_count",
)
_MANIFEST_KEYS = (
    "partition",
    "official_split",
    "image_index",
    "image_filename",
    "image_path",
    "question_index",
    "source_position",
    "question",
    "encoded_tokens",
    "answer",
    "relation",
)
_PREDICTION_KEYS = (
    "protocol_version",
    "method",
    "seed",
    "question_index",
    "label",
    "logits",
)


class CLEVRReportError(ValueError):
    """A reporting input violates the frozen file contract."""


class ReportPublicationError(CLEVRReportError):
    """A publication cleanup failed, with explicit rename commit status.

    The lock protects cooperating publishers. A noncooperating process that
    creates the target between the final check and POSIX rename is outside
    this guarantee.
    """

    def __init__(self, message: str, *, committed: bool) -> None:
        if type(committed) is not bool:
            raise ValueError("committed must be a bool")
        super().__init__(message)
        self.committed = committed


def _require_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    path: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or not self.path.strip()
            or not Path(self.path).is_absolute()
        ):
            raise ValueError(
                "artifact evidence path must be a non-empty absolute string"
            )
        _require_nonnegative_int(self.size_bytes, "artifact evidence size_bytes")
        _require_nonnegative_int(self.record_count, "artifact evidence record_count")


@dataclass(frozen=True, slots=True)
class PartitionCounts:
    train: int
    dev: int
    base_test: int
    confirmation: int

    def __post_init__(self) -> None:
        for partition in _PARTITIONS:
            _require_nonnegative_int(
                getattr(self, partition), f"{partition} manifest count"
            )


@dataclass(frozen=True, slots=True)
class PredictionSource:
    path: Path
    run_config_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError("prediction source path must be a pathlib.Path")
        if self.run_config_path is not None and not isinstance(self.run_config_path, Path):
            raise ValueError("run_config_path must be a pathlib.Path or None")


@dataclass(frozen=True, slots=True)
class PredictionEvidence:
    method: str
    seed: int
    source_path: str
    size_bytes: int
    row_count: int
    canonical_path: str
    run_config_source_path: str | None = None
    run_config_json: str | None = None

    @property
    def canonical_run_config_path(self) -> str:
        return f"predictions/{self.method}_seed_{self.seed}.run_config.json"

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in METHODS:
            raise ValueError("prediction evidence method is not formal")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise ValueError("prediction evidence seed is not formal")
        if (
            type(self.source_path) is not str
            or not self.source_path.strip()
            or not Path(self.source_path).is_absolute()
        ):
            raise ValueError(
                "prediction evidence source_path must be a non-empty absolute string"
            )
        _require_nonnegative_int(
            self.size_bytes, "prediction evidence size_bytes"
        )
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("prediction evidence row_count must be a positive integer")
        expected_path = f"predictions/{self.method}_seed_{self.seed}.jsonl"
        if self.canonical_path != expected_path:
            raise ValueError(
                "prediction evidence canonical_path does not match method and seed"
            )
        if self.run_config_json is not None:
            if (type(self.run_config_json) is not str
                    or type(self.run_config_source_path) is not str
                    or not Path(self.run_config_source_path).is_absolute()):
                raise ValueError("run configuration evidence requires text and absolute source path")


@dataclass(frozen=True, slots=True)
class EnvironmentEvidence:
    python: str
    numpy: str

    def __post_init__(self) -> None:
        if type(self.python) is not str or not self.python.strip():
            raise ValueError("environment python version must be a non-empty string")
        if type(self.numpy) is not str or not self.numpy.strip():
            raise ValueError("environment numpy version must be a non-empty string")


@dataclass(frozen=True, slots=True)
class StatisticsReport:
    protocol_version: str
    report_schema_version: int
    audit_sources: tuple[ArtifactEvidence, ...]
    targets_path: str
    manifest_partition_counts: PartitionCounts
    predictions: tuple[PredictionEvidence, ...]
    statistics: CLEVRStatistics
    environment: EnvironmentEvidence

    def __post_init__(self) -> None:
        if (
            type(self.protocol_version) is not str
            or self.protocol_version != PROTOCOL_VERSION
        ):
            raise ValueError(
                f"report protocol_version must be exactly {PROTOCOL_VERSION!r}"
            )
        if (
            type(self.report_schema_version) is not int
            or self.report_schema_version != 2
        ):
            raise ValueError("report_schema_version must be exactly 2")
        if (
            not isinstance(self.audit_sources, tuple)
            or len(self.audit_sources) != 3
            or any(
                not isinstance(item, ArtifactEvidence)
                for item in self.audit_sources
            )
        ):
            raise ValueError(
                "audit_sources must be the three immutable artifact records"
            )
        if self.targets_path != "targets.jsonl":
            raise ValueError("targets_path must be exactly 'targets.jsonl'")
        if not isinstance(self.manifest_partition_counts, PartitionCounts):
            raise ValueError(
                "manifest_partition_counts must be PartitionCounts"
            )
        expected_keys = tuple(
            (method, seed) for method in METHODS for seed in SEEDS
        )
        if (
            not isinstance(self.predictions, tuple)
            or len(self.predictions) != len(expected_keys)
            or any(
                not isinstance(item, PredictionEvidence)
                for item in self.predictions
            )
            or tuple((item.method, item.seed) for item in self.predictions)
            != expected_keys
        ):
            raise ValueError(
                "predictions must contain the ordered formal evidence grid"
            )
        if not isinstance(self.statistics, CLEVRStatistics):
            raise ValueError("statistics must be CLEVRStatistics")
        if any(item.run_config_json is None for item in self.predictions):
            raise ValueError("paired report requires original run configuration evidence")
        if not isinstance(self.environment, EnvironmentEvidence):
            raise ValueError("environment must be EnvironmentEvidence")
        base_test_count = self.manifest_partition_counts.base_test
        if self.statistics.train_majority.question_count != base_test_count:
            raise ValueError(
                "manifest base_test count does not match train-majority statistics"
            )
        for evidence, run_metrics in zip(
            self.predictions, self.statistics.run_metrics, strict=True
        ):
            if run_metrics.metrics.question_count != base_test_count:
                raise ValueError(
                    "manifest base_test count does not match run statistics"
                )
            if evidence.row_count != base_test_count:
                raise ValueError(
                    "prediction evidence row_count does not match base_test count"
                )


@dataclass(frozen=True, slots=True)
class StatisticsBundle:
    report: StatisticsReport
    targets: tuple[TargetRow, ...]
    runs: tuple[RunPredictions, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.report, StatisticsReport):
            raise ValueError("report must be StatisticsReport")
        if (
            not isinstance(self.targets, tuple)
            or not self.targets
            or any(not isinstance(item, TargetRow) for item in self.targets)
        ):
            raise ValueError("targets must be a non-empty TargetRow tuple")
        partition_rank = {"train": 0, "base_test": 1}
        expected_targets = tuple(
            sorted(
                self.targets,
                key=lambda row: (
                    partition_rank[row.partition],
                    row.question_index,
                ),
            )
        )
        if self.targets != expected_targets:
            raise ValueError("targets must use canonical partition/question order")
        expected_keys = tuple(
            (method, seed) for method in METHODS for seed in SEEDS
        )
        if (
            not isinstance(self.runs, tuple)
            or len(self.runs) != len(expected_keys)
            or any(not isinstance(run, RunPredictions) for run in self.runs)
            or tuple((run.method, run.seed) for run in self.runs)
            != expected_keys
        ):
            raise ValueError("runs must contain the ordered formal run grid")
        evidence_keys = tuple(
            (item.method, item.seed) for item in self.report.predictions
        )
        if evidence_keys != expected_keys:
            raise ValueError("report prediction evidence does not match runs")
        base_test_count = sum(
            target.partition == "base_test" for target in self.targets
        )
        if self.report.manifest_partition_counts.base_test != base_test_count:
            raise ValueError(
                "report base_test count does not match bundle targets"
            )
        for run, evidence in zip(
            self.runs, self.report.predictions, strict=True
        ):
            if len(run.rows) != base_test_count:
                raise ValueError(
                    "run row count does not match bundle base_test targets"
                )
            if evidence.row_count != len(run.rows):
                raise ValueError(
                    "prediction evidence row_count does not match run rows"
                )
        try:
            recomputed_statistics = compute_clevr_statistics(
                self.targets, self.runs
            )
        except ValueError as exc:
            raise ValueError(
                f"bundle targets and runs are inconsistent: {exc}"
            ) from exc
        if recomputed_statistics != self.report.statistics:
            raise ValueError(
                "report statistics do not match bundle targets and runs"
            )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _context_error(
    path: Path, message: str, *, line: int | None = None
) -> CLEVRReportError:
    location = str(path)
    if line is not None:
        location += f", line {line}"
    return CLEVRReportError(f"{location}: {message}")


def _resolve_regular_file(path: Path) -> Path:
    try:
        resolved = path.resolve()
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError) as exc:
        raise _context_error(path.absolute(), str(exc)) from exc
    if not stat.S_ISREG(mode):
        raise _context_error(resolved, "audit input must be a regular file")
    return resolved


def _read_json_document(path: Path) -> tuple[object, int, Path]:
    resolved = _resolve_regular_file(path)
    try:
        with resolved.open("r", encoding="utf-8") as input_file:
            size_bytes = os.fstat(input_file.fileno()).st_size
            value = json.load(
                input_file,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
            )
    except json.JSONDecodeError as exc:
        raise _context_error(resolved, str(exc), line=exc.lineno) from exc
    except (UnicodeError, OSError, ValueError) as exc:
        raise _context_error(resolved, str(exc)) from exc
    return value, size_bytes, resolved


def _read_json_lines(
    path: Path,
) -> tuple[tuple[tuple[int, object], ...], int, Path]:
    resolved = _resolve_regular_file(path)
    rows: list[tuple[int, object]] = []
    try:
        with resolved.open("r", encoding="utf-8") as input_file:
            size_bytes = os.fstat(input_file.fileno()).st_size
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise _context_error(
                        resolved,
                        "blank JSONL records are not allowed",
                        line=line_number,
                    )
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_nonstandard_constant,
                    )
                except json.JSONDecodeError as exc:
                    raise _context_error(
                        resolved, str(exc), line=line_number
                    ) from exc
                except ValueError as exc:
                    raise _context_error(
                        resolved, str(exc), line=line_number
                    ) from exc
                rows.append((line_number, value))
    except CLEVRReportError:
        raise
    except (UnicodeError, OSError) as exc:
        raise _context_error(resolved, str(exc)) from exc
    if not rows:
        raise _context_error(resolved, "JSONL input must not be empty")
    return tuple(rows), size_bytes, resolved


def _require_exact_keys(
    value: object, expected: tuple[str, ...], context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be a JSON object")
    observed = set(value)
    required = set(expected)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise ValueError(
            f"{context} keys must be exact; missing={missing}, extra={extra}"
        )
    return value


def _require_protocol(value: object) -> None:
    if type(value) is not str or value != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be exactly {PROTOCOL_VERSION!r}")


def _parse_image_partitions(
    value: object, path: Path
) -> tuple[dict[str, tuple[ImageIdentity, ...]], dict[ImageIdentity, str], int]:
    try:
        document = _require_exact_keys(
            value, _IMAGE_DOCUMENT_KEYS, "image partition document"
        )
        _require_protocol(document["protocol_version"])
        partitions = _require_exact_keys(
            document["partitions"], _PARTITIONS, "image partitions"
        )

        parsed: dict[str, tuple[ImageIdentity, ...]] = {}
        membership: dict[ImageIdentity, str] = {}
        seen_keys: set[tuple[str, int]] = set()
        seen_filenames: set[str] = set()
        record_count = 0
        for partition in _PARTITIONS:
            raw_images = partitions[partition]
            if type(raw_images) is not list:
                raise ValueError(f"image partition {partition!r} must be a JSON array")
            images: list[ImageIdentity] = []
            for index, raw_image in enumerate(raw_images):
                image_record = _require_exact_keys(
                    raw_image, _IMAGE_KEYS, f"{partition} image record {index}"
                )
                official_split = image_record["official_split"]
                image_index = image_record["image_index"]
                image_filename = image_record["image_filename"]
                if type(official_split) is not str:
                    raise ValueError("official_split must be a string")
                _require_nonnegative_int(image_index, "image_index")
                if type(image_filename) is not str:
                    raise ValueError("image_filename must be a string")
                image = ImageIdentity(
                    official_split=official_split,
                    image_index=image_index,
                    image_filename=image_filename,
                )
                expected_split = _PARTITION_SPLITS[partition]
                if image.official_split != expected_split:
                    raise ValueError(
                        f"partition {partition!r} must contain official "
                        f"{expected_split!r} images"
                    )
                image_key = (image.official_split, image.image_index)
                if image_key in seen_keys or image.image_filename in seen_filenames:
                    raise ValueError(
                        "duplicate image identity across partitions: "
                        f"{image.image_filename}"
                    )
                seen_keys.add(image_key)
                seen_filenames.add(image.image_filename)
                membership[image] = partition
                images.append(image)
                record_count += 1
            parsed[partition] = tuple(images)
    except (TypeError, ValueError) as exc:
        raise _context_error(path, str(exc)) from exc
    return parsed, membership, record_count


def _parse_partition_summaries(
    value: object, path: Path
) -> dict[str, dict[str, Any]]:
    try:
        report = _require_exact_keys(value, _REPORT_KEYS, "audit report")
        _require_protocol(report["protocol_version"])
        if type(report["environment"]) is not dict:
            raise ValueError("audit report environment must be a JSON object")
        for field in (
            "cell_counts",
            "exclusion_counts",
            "source_summaries",
            "file_summaries",
        ):
            if type(report[field]) is not list:
                raise ValueError(f"audit report {field} must be a JSON array")

        raw_summaries = report["partition_summaries"]
        if type(raw_summaries) is not list:
            raise ValueError("partition_summaries must be a JSON array")
        if len(raw_summaries) != len(_PARTITIONS):
            raise ValueError("partition_summaries must contain exactly four records")

        summaries: dict[str, dict[str, Any]] = {}
        for index, expected_partition in enumerate(_PARTITIONS):
            summary = _require_exact_keys(
                raw_summaries[index],
                _SUMMARY_KEYS,
                f"partition summary {index}",
            )
            if (
                type(summary["partition"]) is not str
                or summary["partition"] != expected_partition
            ):
                raise ValueError(
                    "partition_summaries must follow train, dev, base_test, "
                    "confirmation order"
                )
            for field in (
                "mother_image_count",
                "selected_image_count",
                "selected_question_count",
            ):
                _require_nonnegative_int(
                    summary[field], f"{expected_partition} {field}"
                )
            summaries[expected_partition] = summary
    except (TypeError, ValueError) as exc:
        raise _context_error(path, str(exc)) from exc
    return summaries


def _parse_manifest(
    rows: tuple[tuple[int, object], ...],
    path: Path,
    image_membership: dict[ImageIdentity, str],
) -> tuple[
    tuple[TargetRow, ...],
    PartitionCounts,
    dict[str, set[ImageIdentity]],
]:
    targets: list[TargetRow] = []
    counts = {partition: 0 for partition in _PARTITIONS}
    selected_images = {partition: set() for partition in _PARTITIONS}
    seen_questions: set[tuple[str, int]] = set()
    seen_source_positions: set[tuple[str, int]] = set()

    for line_number, value in rows:
        try:
            row = _require_exact_keys(value, _MANIFEST_KEYS, "manifest row")
            partition = row["partition"]
            if type(partition) is not str or partition not in _PARTITIONS:
                raise ValueError("partition is not part of the frozen protocol")
            official_split = row["official_split"]
            if type(official_split) is not str:
                raise ValueError("official_split must be a string")
            if official_split != _PARTITION_SPLITS[partition]:
                raise ValueError(
                    "official_split does not match the manifest partition"
                )
            image_index = _require_nonnegative_int(
                row["image_index"], "image_index"
            )
            image_filename = row["image_filename"]
            if type(image_filename) is not str:
                raise ValueError("image_filename must be a string")
            image = ImageIdentity(
                official_split=official_split,
                image_index=image_index,
                image_filename=image_filename,
            )
            image_path = row["image_path"]
            expected_image_path = (
                f"images/{official_split}/{image.image_filename}"
            )
            if type(image_path) is not str or image_path != expected_image_path:
                raise ValueError(
                    "image_path must be the canonical image path for the row identity"
                )

            question_index = _require_nonnegative_int(
                row["question_index"], "question_index"
            )
            source_position = _require_nonnegative_int(
                row["source_position"], "source_position"
            )
            if type(row["question"]) is not str:
                raise ValueError("question must be a string")
            encoded_tokens = row["encoded_tokens"]
            if type(encoded_tokens) is not list or any(
                type(token) is not int or token < 0 for token in encoded_tokens
            ):
                raise ValueError(
                    "encoded_tokens must be an array of non-negative integers"
                )
            answer = row["answer"]
            if type(answer) is not int or answer not in {0, 1}:
                raise ValueError("answer must use no=0 / yes=1")
            relation = row["relation"]
            if type(relation) is not str or relation not in _RELATIONS:
                raise ValueError("relation is not part of the frozen task")

            question_key = (official_split, question_index)
            if question_key in seen_questions:
                raise ValueError(
                    f"duplicate question identity: {question_key!r}"
                )
            source_key = (official_split, source_position)
            if source_key in seen_source_positions:
                raise ValueError(
                    f"duplicate source position: {source_key!r}"
                )
            seen_questions.add(question_key)
            seen_source_positions.add(source_key)

            assigned_partition = image_membership.get(image)
            if assigned_partition is None:
                raise ValueError(
                    "selected image is absent from image partitions: "
                    f"{image.image_filename}"
                )
            if assigned_partition != partition:
                raise ValueError(
                    f"selected image belongs to {assigned_partition!r}, "
                    f"not {partition!r}"
                )

            counts[partition] += 1
            selected_images[partition].add(image)
            if partition in _TARGET_PARTITIONS:
                targets.append(
                    TargetRow(
                        partition=partition,
                        official_split=official_split,
                        image_index=image_index,
                        image_filename=image_filename,
                        question_index=question_index,
                        relation=relation,
                        label=answer,
                    )
                )
        except (TypeError, ValueError) as exc:
            raise _context_error(path, str(exc), line=line_number) from exc

    partition_rank = {
        partition: index for index, partition in enumerate(_TARGET_PARTITIONS)
    }
    targets.sort(key=lambda row: (partition_rank[row.partition], row.question_index))
    for partition in _TARGET_PARTITIONS:
        if not any(row.partition == partition for row in targets):
            raise _context_error(path, f"manifest has no {partition} targets")
    return (
        tuple(targets),
        PartitionCounts(**counts),
        selected_images,
    )


def _cross_check_summaries(
    summaries: dict[str, dict[str, Any]],
    image_partitions: dict[str, tuple[ImageIdentity, ...]],
    selected_images: dict[str, set[ImageIdentity]],
    partition_counts: PartitionCounts,
    report_path: Path,
) -> None:
    for partition in _PARTITIONS:
        summary = summaries[partition]
        observed = {
            "mother_image_count": len(image_partitions[partition]),
            "selected_image_count": len(selected_images[partition]),
            "selected_question_count": getattr(partition_counts, partition),
        }
        for field, actual in observed.items():
            if summary[field] != actual:
                raise _context_error(
                    report_path,
                    f"{partition} {field}={summary[field]} does not match "
                    f"observed {actual}",
                )


def load_audit_targets(
    audit_dir: Path,
) -> tuple[
    tuple[TargetRow, ...],
    tuple[ArtifactEvidence, ...],
    PartitionCounts,
]:
    """Load and cross-check one explicit PUB-R1.1 audit artifact directory."""
    if not isinstance(audit_dir, Path):
        raise CLEVRReportError("audit_dir must be a pathlib.Path")
    try:
        resolved_dir = audit_dir.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise CLEVRReportError(
            f"cannot resolve audit directory {audit_dir}: {exc}"
        ) from exc
    if not resolved_dir.is_dir():
        raise CLEVRReportError(
            f"audit directory is absent or not a directory: {resolved_dir}"
        )

    image_path = resolved_dir / "image_partitions.json"
    manifest_path = resolved_dir / "selected_questions.jsonl"
    report_path = resolved_dir / "audit_report.json"

    image_document, image_size, image_path = _read_json_document(image_path)
    manifest_rows, manifest_size, manifest_path = _read_json_lines(manifest_path)
    report_document, report_size, report_path = _read_json_document(report_path)

    image_partitions, image_membership, image_count = _parse_image_partitions(
        image_document, image_path
    )
    summaries = _parse_partition_summaries(report_document, report_path)
    targets, partition_counts, selected_images = _parse_manifest(
        manifest_rows, manifest_path, image_membership
    )
    _cross_check_summaries(
        summaries,
        image_partitions,
        selected_images,
        partition_counts,
        report_path,
    )

    evidence = (
        ArtifactEvidence(str(image_path), image_size, image_count),
        ArtifactEvidence(str(manifest_path), manifest_size, len(manifest_rows)),
        ArtifactEvidence(str(report_path), report_size, 1),
    )
    return targets, evidence, partition_counts


def load_prediction_run(
    source: PredictionSource,
) -> tuple[RunPredictions, PredictionEvidence]:
    """Load one explicit self-described PUB-R1.1 prediction file."""
    if not isinstance(source, PredictionSource):
        raise CLEVRReportError("source must be a PredictionSource")

    raw_rows, size_bytes, resolved_path = _read_json_lines(source.path)
    parsed_rows: list[PredictionRow] = []
    seen_question_ids: set[int] = set()
    declared_identity: tuple[str, int] | None = None

    for line_number, value in raw_rows:
        try:
            row = _require_exact_keys(
                value, _PREDICTION_KEYS, "prediction row"
            )
            _require_protocol(row["protocol_version"])
            method = row["method"]
            seed = row["seed"]
            if type(method) is not str or method not in METHODS:
                raise ValueError(
                    "method is not part of the frozen formal comparison"
                )
            if type(seed) is not int or seed not in SEEDS:
                raise ValueError(
                    "seed is not part of the frozen formal comparison"
                )
            identity = (method, seed)
            if declared_identity is None:
                declared_identity = identity
            elif identity != declared_identity:
                raise ValueError(
                    "method and seed must agree across every row in the file"
                )

            logits = row["logits"]
            if type(logits) is not list:
                raise ValueError("logits must be a JSON array")
            prediction = PredictionRow(
                question_index=row["question_index"],
                label=row["label"],
                logits=tuple(logits),
            )
            if prediction.question_index in seen_question_ids:
                raise ValueError(
                    f"duplicate prediction question ID: {prediction.question_index}"
                )
            seen_question_ids.add(prediction.question_index)
            parsed_rows.append(prediction)
        except (TypeError, ValueError) as exc:
            raise _context_error(
                resolved_path, str(exc), line=line_number
            ) from exc

    if declared_identity is None:
        raise _context_error(resolved_path, "prediction file must not be empty")
    method, seed = declared_identity
    run = RunPredictions(
        method=method,
        seed=seed,
        rows=tuple(sorted(parsed_rows, key=lambda row: row.question_index)),
    )
    evidence = PredictionEvidence(
        method=method,
        seed=seed,
        source_path=str(resolved_path),
        size_bytes=size_bytes,
        row_count=len(parsed_rows),
        canonical_path=f"predictions/{method}_seed_{seed}.jsonl",
    )
    return run, evidence


def _validated_prediction_sources(
    prediction_sources: Sequence[PredictionSource],
) -> tuple[PredictionSource, ...]:
    if (
        not isinstance(prediction_sources, Sequence)
        or isinstance(prediction_sources, (str, bytes, bytearray))
    ):
        raise CLEVRReportError(
            "prediction_sources must be a sequence of PredictionSource records"
        )
    sources = tuple(prediction_sources)
    if len(sources) != len(METHODS) * len(SEEDS):
        raise CLEVRReportError("prediction_sources must contain exactly 20 records")
    if any(not isinstance(source, PredictionSource) for source in sources):
        raise CLEVRReportError(
            "prediction_sources must contain only PredictionSource records"
        )

    resolved_paths: list[Path] = []
    for source in sources:
        try:
            resolved_paths.append(source.path.expanduser().resolve())
        except (OSError, RuntimeError) as exc:
            raise CLEVRReportError(
                f"cannot resolve prediction source {source.path}: {exc}"
            ) from exc
    if len(set(resolved_paths)) != len(resolved_paths):
        raise CLEVRReportError(
            "prediction_sources must have 20 distinct resolved paths"
        )
    return sources


def _audit_text_snapshot(audit_dir: Path) -> dict[str, str]:
    if not isinstance(audit_dir, Path):
        raise CLEVRReportError("audit_dir must be a pathlib.Path")
    try:
        return {name: (audit_dir / name).read_bytes().decode("utf-8")
                for name in AUDIT_ARTIFACT_NAMES}
    except (OSError, UnicodeError) as exc:
        raise CLEVRReportError(f"paired data audit snapshot is unavailable: {exc}") from exc


def _load_run_provenance(
    source: PredictionSource, run: RunPredictions, evidence: PredictionEvidence,
    expected_audit: dict[str, str],
) -> tuple[PredictionEvidence, str | None]:
    path = source.run_config_path
    if path is None:
        prediction = source.path
        if prediction.parent.name == "base_test" and prediction.parent.parent.name == "predictions":
            path = prediction.parent.parent.parent / "run_config.json"
        else:
            path = prediction.with_suffix(".run_config.json")
    config, _, resolved = _read_json_document(path)
    try:
        snapshot, rgb_root = run_data_contract(
            config, method=run.method, seed=run.seed, phase="formal", target_updates=8000,
        )
        if snapshot != expected_audit:
            different = [name for name in AUDIT_ARTIFACT_NAMES if snapshot[name] != expected_audit[name]]
            raise ValueError(f"paired data differs from supplied audit: {different}")
    except ValueError as exc:
        raise _context_error(resolved, str(exc)) from exc
    return replace(evidence, run_config_source_path=str(resolved),
                   run_config_json=json.dumps(config, sort_keys=True, allow_nan=False)), rgb_root


def build_statistics_bundle(
    audit_dir: Path,
    prediction_sources: Sequence[PredictionSource],
) -> StatisticsBundle:
    """Build one typed report from explicit audit and prediction sources."""
    sources = _validated_prediction_sources(prediction_sources)
    audit_snapshot = _audit_text_snapshot(audit_dir)
    targets, audit_sources, partition_counts = load_audit_targets(audit_dir)

    loaded = tuple(load_prediction_run(source) for source in sources)
    identity_counts: dict[tuple[str, int], int] = {}
    for run, _ in loaded:
        key = (run.method, run.seed)
        identity_counts[key] = identity_counts.get(key, 0) + 1
    expected_keys = tuple(
        (method, seed) for method in METHODS for seed in SEEDS
    )
    duplicates = tuple(
        key for key, count in identity_counts.items() if count > 1
    )
    missing = tuple(key for key in expected_keys if key not in identity_counts)
    if duplicates or missing or len(identity_counts) != len(expected_keys):
        raise CLEVRReportError(
            "prediction sources do not form the exact formal grid; "
            f"duplicate={duplicates}, missing={missing}"
        )

    paired = []
    rgb_root = None
    for source, (run, evidence) in zip(sources, loaded, strict=True):
        verified, root = _load_run_provenance(source, run, evidence, audit_snapshot)
        if root is not None:
            if rgb_root is not None and root != rgb_root:
                raise CLEVRReportError("paired RGB roots disagree across prediction runs")
            rgb_root = root
        paired.append((run, verified))
    if _audit_text_snapshot(audit_dir) != audit_snapshot:
        raise CLEVRReportError("paired data audit changed while report was constructed")

    by_key = {
        (run.method, run.seed): (run, evidence)
        for run, evidence in paired
    }
    runs = tuple(by_key[key][0] for key in expected_keys)
    prediction_evidence = tuple(by_key[key][1] for key in expected_keys)
    try:
        statistics = compute_clevr_statistics(targets, runs)
    except ValueError as exc:
        raise CLEVRReportError(f"statistics validation failed: {exc}") from exc

    report = StatisticsReport(
        protocol_version=PROTOCOL_VERSION,
        report_schema_version=2,
        audit_sources=audit_sources,
        targets_path="targets.jsonl",
        manifest_partition_counts=partition_counts,
        predictions=prediction_evidence,
        statistics=statistics,
        environment=EnvironmentEvidence(
            python=platform.python_version(),
            numpy=metadata.version("numpy"),
        ),
    )
    return StatisticsBundle(report=report, targets=targets, runs=runs)


def _artifact_evidence_payload(evidence: ArtifactEvidence) -> dict[str, object]:
    return {
        "path": evidence.path,
        "size_bytes": evidence.size_bytes,
        "record_count": evidence.record_count,
    }


def _partition_counts_payload(counts: PartitionCounts) -> dict[str, object]:
    return {
        "train": counts.train,
        "dev": counts.dev,
        "base_test": counts.base_test,
        "confirmation": counts.confirmation,
    }


def _prediction_evidence_payload(
    evidence: PredictionEvidence,
) -> dict[str, object]:
    return {
        "method": evidence.method,
        "seed": evidence.seed,
        "source_path": evidence.source_path,
        "size_bytes": evidence.size_bytes,
        "row_count": evidence.row_count,
        "canonical_path": evidence.canonical_path,
        "run_config_source_path": evidence.run_config_source_path,
        "canonical_run_config_path": evidence.canonical_run_config_path,
    }


def _environment_payload(environment: EnvironmentEvidence) -> dict[str, object]:
    return {"python": environment.python, "numpy": environment.numpy}


def _cell_metrics_payload(metrics: CellMetrics) -> dict[str, object]:
    accuracy: float | str = (
        "NA" if metrics.accuracy is None else metrics.accuracy
    )
    return {
        "relation": metrics.relation,
        "label": metrics.label,
        "count": metrics.count,
        "correct_count": metrics.correct_count,
        "accuracy": accuracy,
    }


def _classification_metrics_payload(
    metrics: ClassificationMetrics,
) -> dict[str, object]:
    cross_entropy: float | str = (
        "NA" if metrics.cross_entropy is None else metrics.cross_entropy
    )
    macro_accuracy: float | str = (
        "NA" if metrics.macro_accuracy is None else metrics.macro_accuracy
    )
    return {
        "question_count": metrics.question_count,
        "correct_count": metrics.correct_count,
        "accuracy": metrics.accuracy,
        "cross_entropy": cross_entropy,
        "cells": [_cell_metrics_payload(cell) for cell in metrics.cells],
        "macro_accuracy": macro_accuracy,
    }


def _run_metrics_payload(metrics: RunMetrics) -> dict[str, object]:
    return {
        "method": metrics.method,
        "seed": metrics.seed,
        "metrics": _classification_metrics_payload(metrics.metrics),
    }


def _seed_summary_payload(summary: SeedSummary) -> dict[str, object]:
    return {
        "name": summary.name,
        "seeds": list(summary.seeds),
        "values": list(summary.values),
        "mean": summary.mean,
        "sample_sd": summary.sample_sd,
        "interval": list(summary.interval),
    }


def _paired_summary_payload(summary: PairedSummary) -> dict[str, object]:
    return {
        "name": summary.name,
        "seeds": list(summary.seeds),
        "values": list(summary.values),
        "mean": summary.mean,
        "sample_sd": summary.sample_sd,
        "interval": list(summary.interval),
    }


def _cluster_bootstrap_payload(
    summary: ClusterBootstrapSummary,
) -> dict[str, object]:
    return {
        "seed": summary.seed,
        "draws": summary.draws,
        "quantile_method": summary.quantile_method,
        "replicates": list(summary.replicates),
        "interval": list(summary.interval),
    }


def _statistics_payload(statistics: CLEVRStatistics) -> dict[str, object]:
    return {
        "run_metrics": [
            _run_metrics_payload(metrics) for metrics in statistics.run_metrics
        ],
        "method_summaries": [
            _seed_summary_payload(summary)
            for summary in statistics.method_summaries
        ],
        "b_minus_g": _paired_summary_payload(statistics.b_minus_g),
        "m10_minus_g": _paired_summary_payload(statistics.m10_minus_g),
        "train_majority": _classification_metrics_payload(
            statistics.train_majority
        ),
        "bootstrap": _cluster_bootstrap_payload(statistics.bootstrap),
    }


def statistics_report_payload(report: StatisticsReport) -> dict[str, object]:
    """Convert one validated report into the fixed JSON-compatible schema."""
    if not isinstance(report, StatisticsReport):
        raise ValueError("report must be StatisticsReport")
    return {
        "protocol_version": report.protocol_version,
        "report_schema_version": report.report_schema_version,
        "environment": _environment_payload(report.environment),
        "audit_sources": [
            _artifact_evidence_payload(evidence)
            for evidence in report.audit_sources
        ],
        "targets_path": report.targets_path,
        "manifest_partition_counts": _partition_counts_payload(
            report.manifest_partition_counts
        ),
        "predictions": [
            _prediction_evidence_payload(evidence)
            for evidence in report.predictions
        ],
        "statistics": _statistics_payload(report.statistics),
    }


def _target_payload(target: TargetRow) -> dict[str, object]:
    return {
        "partition": target.partition,
        "official_split": target.official_split,
        "image_index": target.image_index,
        "image_filename": target.image_filename,
        "question_index": target.question_index,
        "relation": target.relation,
        "label": target.label,
    }


def _prediction_payload(
    run: RunPredictions,
    row: PredictionRow,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "method": run.method,
        "seed": run.seed,
        "question_index": row.question_index,
        "label": row.label,
        "logits": list(row.logits),
    }


def _write_json_lines(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            json.dump(
                row,
                output_file,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            output_file.write("\n")


def _write_json_document(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=2,
        )
        output_file.write("\n")


def _write_lock_token(lock_fd: int, token: bytes) -> None:
    remaining = memoryview(token)
    while remaining:
        written = os.write(lock_fd, remaining)
        if written <= 0:
            raise OSError("publication lock token write made no progress")
        remaining = remaining[written:]


def _unlink_owned_lock(
    lock_path: Path,
    lock_identity: tuple[int, int] | None,
    lock_token: bytes | None,
) -> None:
    if lock_identity is None:
        raise OSError(
            f"publication lock ownership is unavailable for {lock_path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    current_fd: int | None = None
    current_stat: os.stat_result | None = None
    current_token: bytes | None = None
    verification_error: BaseException | None = None
    try:
        current_fd = os.open(lock_path, flags)
        current_stat = os.fstat(current_fd)
        if lock_token is not None:
            token_buffer = bytearray()
            read_limit = len(lock_token) + 1
            while len(token_buffer) < read_limit:
                chunk = os.read(current_fd, read_limit - len(token_buffer))
                if not chunk:
                    break
                token_buffer.extend(chunk)
            current_token = bytes(token_buffer)
    except BaseException as exc:
        verification_error = exc
    finally:
        if current_fd is not None:
            try:
                os.close(current_fd)
            except BaseException as close_error:
                if verification_error is None:
                    raise
                verification_error.add_note(
                    f"publication lock verification descriptor close also failed: "
                    f"{close_error}"
                )
    if verification_error is not None:
        if isinstance(verification_error, OSError):
            raise OSError(
                "publication lock ownership cannot be verified for "
                f"{lock_path}: {verification_error}"
            ) from verification_error
        raise verification_error
    if current_stat is None:
        raise OSError(
            f"publication lock ownership is unavailable for {lock_path}"
        )

    current_identity = (current_stat.st_dev, current_stat.st_ino)
    if current_identity != lock_identity or (
        lock_token is not None and current_token != lock_token
    ):
        raise OSError(
            "publication lock ownership changed; refusing to unlink "
            f"{lock_path}"
        )
    try:
        latest_stat = lock_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OSError(
            f"publication lock ownership cannot be verified for {lock_path}: {exc}"
        ) from exc
    latest_identity = (latest_stat.st_dev, latest_stat.st_ino)
    if latest_identity != lock_identity:
        raise OSError(
            "publication lock ownership changed; refusing to unlink "
            f"{lock_path}"
        )
    lock_path.unlink()


def _release_publication_lock(
    lock_path: Path,
    lock_fd: int,
    lock_identity: tuple[int, int] | None,
    lock_token: bytes | None,
) -> None:
    cleanup_error: BaseException | None = None
    try:
        _unlink_owned_lock(lock_path, lock_identity, lock_token)
    except BaseException as exc:
        cleanup_error = exc
    try:
        os.close(lock_fd)
    except BaseException as exc:
        if cleanup_error is None:
            cleanup_error = exc
        else:
            cleanup_error.add_note(
                f"publication lock descriptor close also failed: {exc}"
            )
    if cleanup_error is not None:
        raise cleanup_error


def _rename_staging_directory(
    staging_dir: Path,
    output_dir: Path,
) -> BaseException | None:
    staging_stat = staging_dir.stat(follow_symlinks=False)
    staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
    try:
        os.rename(staging_dir, output_dir)
    except BaseException as rename_error:
        try:
            staging_dir.stat(follow_symlinks=False)
        except FileNotFoundError:
            staging_absent = True
        except OSError:
            staging_absent = False
        else:
            staging_absent = False

        try:
            output_stat = output_dir.stat(follow_symlinks=False)
        except OSError:
            output_identity = None
        else:
            output_identity = (output_stat.st_dev, output_stat.st_ino)

        if staging_absent and output_identity == staging_identity:
            return rename_error
        raise
    return None


def publish_statistics_bundle(
    output_dir: Path,
    bundle: StatisticsBundle,
) -> None:
    """Atomically publish canonical files to one new output directory."""
    if not isinstance(output_dir, Path):
        raise CLEVRReportError("output_dir must be a pathlib.Path")
    if not isinstance(bundle, StatisticsBundle):
        raise CLEVRReportError("bundle must be a StatisticsBundle")

    try:
        absolute_output = output_dir.expanduser().absolute()
        resolved_output = absolute_output.parent.resolve() / absolute_output.name
    except (OSError, RuntimeError) as exc:
        raise CLEVRReportError(
            f"cannot resolve output directory {output_dir}: {exc}"
        ) from exc

    if resolved_output.exists() or resolved_output.is_symlink():
        raise CLEVRReportError(
            f"output directory already exists or is a symlink: {resolved_output}"
        )
    try:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLEVRReportError(
            f"cannot create output parent {resolved_output.parent}: {exc}"
        ) from exc

    lock_path = resolved_output.parent / f".{resolved_output.name}.lock"
    lock_token = secrets.token_hex(32).encode("ascii")
    committed = False
    staging_dir: Path | None = None
    lock_identity: tuple[int, int] | None = None
    initialized_lock_token: bytes | None = None
    rename_error: BaseException | None = None
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except OSError as exc:
        raise CLEVRReportError(
            f"cannot acquire publication lock {lock_path}: {exc}"
        ) from exc

    try:
        lock_stat = os.fstat(lock_fd)
        lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
        _write_lock_token(lock_fd, lock_token)
        initialized_lock_token = lock_token
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{resolved_output.name}.staging-",
                dir=resolved_output.parent,
            )
        )
        prediction_dir = staging_dir / "predictions"
        prediction_dir.mkdir()
        _write_json_lines(
            staging_dir / bundle.report.targets_path,
            tuple(_target_payload(target) for target in bundle.targets),
        )
        for run, evidence in zip(
            bundle.runs,
            bundle.report.predictions,
            strict=True,
        ):
            _write_json_document(
                staging_dir / evidence.canonical_run_config_path,
                json.loads(evidence.run_config_json),
            )
            _write_json_lines(
                staging_dir / evidence.canonical_path,
                tuple(
                    _prediction_payload(run, row)
                    for row in sorted(
                        run.rows,
                        key=lambda item: item.question_index,
                    )
                ),
            )
        _write_json_document(
            staging_dir / "statistics_report.json",
            statistics_report_payload(bundle.report),
        )
        if resolved_output.exists() or resolved_output.is_symlink():
            raise CLEVRReportError(
                "output directory appeared after publication lock acquisition: "
                f"{resolved_output}"
            )
        rename_error = _rename_staging_directory(staging_dir, resolved_output)
        committed = True
        staging_dir = None
    except BaseException as primary_error:
        staging_cleanup_error: BaseException | None = None
        if staging_dir is not None:
            try:
                shutil.rmtree(staging_dir)
            except BaseException as exc:
                staging_cleanup_error = exc
        try:
            _release_publication_lock(
                lock_path,
                lock_fd,
                lock_identity,
                initialized_lock_token,
            )
        except BaseException as cleanup_error:
            cleanup_details = f"unable to remove publication lock: {cleanup_error}"
            if staging_cleanup_error is not None:
                cleanup_details += (
                    "; unable to remove publication staging directory: "
                    f"{staging_cleanup_error}"
                )
            raise ReportPublicationError(
                "statistics bundle not committed after publication failed: "
                f"{primary_error}; {cleanup_details}",
                committed=False,
            ) from cleanup_error
        if staging_cleanup_error is not None:
            raise ReportPublicationError(
                "statistics bundle not committed after publication failed: "
                f"{primary_error}; unable to remove publication staging directory: "
                f"{staging_cleanup_error}",
                committed=False,
            ) from staging_cleanup_error
        raise

    try:
        _release_publication_lock(
            lock_path,
            lock_fd,
            lock_identity,
            initialized_lock_token,
        )
    except BaseException as cleanup_error:
        if rename_error is not None:
            cleanup_error.add_note(
                "rename also raised after the committed directory was reconciled: "
                f"{rename_error}"
            )
        raise ReportPublicationError(
            "statistics bundle committed, but unable to remove publication lock: "
            f"{cleanup_error}",
            committed=committed,
        ) from cleanup_error
    if rename_error is not None:
        raise ReportPublicationError(
            "statistics bundle committed, but rename raised after the commit "
            f"was reconciled: {rename_error}",
            committed=True,
        ) from rename_error
