"""Strict file boundary for CLEVR PUB-R1.1 statistics reporting."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.clevr_partitions import ImageIdentity
from src.clevr_statistics import TargetRow


PROTOCOL_VERSION = "PUB-R1.1"

_PARTITIONS = ("train", "dev", "base_test", "confirmation")
_TARGET_PARTITIONS = ("train", "base_test")
_PARTITION_SPLITS = {
    "train": "train",
    "dev": "val",
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


class CLEVRReportError(ValueError):
    """A reporting input violates the frozen file contract."""


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
