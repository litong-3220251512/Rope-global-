"""Model-safe runtime access to audited PUB-R1.1 CLEVR questions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from src.clevr_partitions import ImageIdentity
from src.clevr_text import Vocabulary, encode_question


_PROTOCOL_VERSION = "PUB-R1.1"
_PARTITION_SPLITS = {
    "train": "train",
    "dev": "train",
    "base_test": "val",
    "confirmation": "val",
}
_RELATIONS = {"left", "right", "front", "behind"}
_FLOAT32_UNIT_INTERVAL_TOLERANCE = 4 * torch.finfo(torch.float32).eps
_ROW_FIELDS = {
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
}
_MODEL_FIELDS = {"image", "question_tokens", "question_length", "answer"}
_QUESTION_ONLY_MODEL_FIELDS = {"question_tokens", "question_length", "answer"}


class CLEVRDatasetError(ValueError):
    """Raised when audited runtime artifacts violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class CLEVRSampleIdentity:
    """Audit-only provenance; never included in model samples or batches."""

    partition: str
    official_split: str
    image_index: int
    image_filename: str
    question_index: int
    source_position: int


@dataclass(frozen=True, slots=True)
class _RuntimeRecord:
    partition: str
    official_split: str
    image_filename: str
    image_path: Path | None
    image_index: int
    question_index: int
    source_position: int
    encoded_tokens: tuple[int, ...]
    answer: int


def _read_json_mapping(path: Path, artifact: str) -> Mapping[str, object]:
    if not path.is_file():
        raise CLEVRDatasetError(f"required {artifact} artifact is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CLEVRDatasetError(f"cannot parse {artifact} artifact {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CLEVRDatasetError(f"{artifact} artifact must contain a mapping")
    return payload


def _require_protocol(payload: Mapping[str, object], artifact: str) -> None:
    if payload.get("protocol_version") != _PROTOCOL_VERSION:
        raise CLEVRDatasetError(
            f"{artifact} protocol_version must be {_PROTOCOL_VERSION!r}"
        )


def _load_vocabulary(path: Path) -> Vocabulary:
    payload = _read_json_mapping(path, "vocabulary")
    _require_protocol(payload, "vocabulary")
    try:
        id_to_token = payload["id_to_token"]
        if not isinstance(id_to_token, list):
            raise ValueError("id_to_token must be a list")
        return Vocabulary(
            token_to_id=payload["token_to_id"],
            id_to_token=tuple(id_to_token),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CLEVRDatasetError(f"invalid vocabulary artifact: {exc}") from exc


def _load_report_counts(path: Path) -> dict[str, int]:
    payload = _read_json_mapping(path, "audit report")
    _require_protocol(payload, "audit report")
    summaries = payload.get("partition_summaries")
    if not isinstance(summaries, list):
        raise CLEVRDatasetError("audit report partition_summaries must be a list")
    counts: dict[str, int] = {}
    for index, raw_summary in enumerate(summaries):
        if not isinstance(raw_summary, Mapping):
            raise CLEVRDatasetError(f"audit report summary {index} must be a mapping")
        partition = raw_summary.get("partition")
        if partition not in _PARTITION_SPLITS:
            raise CLEVRDatasetError(
                f"audit report summary {index} has an unknown partition"
            )
        if partition in counts:
            raise CLEVRDatasetError(
                f"duplicate audit report summary for partition {partition}"
            )
        count = raw_summary.get("selected_question_count")
        if type(count) is not int or count < 0:
            raise CLEVRDatasetError(
                f"audit report summary {partition} selected_question_count "
                "must be a non-negative integer"
            )
        counts[partition] = count
    missing = set(_PARTITION_SPLITS).difference(counts)
    if missing:
        raise CLEVRDatasetError(
            f"audit report is missing partition summaries: {sorted(missing)}"
        )
    return counts


def _require_string(row: Mapping[str, object], field: str, context: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        raise CLEVRDatasetError(f"{context} field {field} must be a string")
    return value


def _require_index(row: Mapping[str, object], field: str, context: str) -> int:
    value = row[field]
    if type(value) is not int or value < 0:
        raise CLEVRDatasetError(
            f"{context} field {field} must be a non-negative integer"
        )
    return value


def _resolve_image_file(path: Path, *, data_root: Path, context: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CLEVRDatasetError(
            f"{context} image_path does not identify an existing file: {path}"
        ) from exc
    if not resolved.is_relative_to(data_root):
        raise CLEVRDatasetError(
            f"{context} image_path escapes the explicit data root: {path}"
        )
    if not resolved.is_file():
        raise CLEVRDatasetError(f"{context} image_path is not a regular file: {path}")
    return resolved


def _parse_row(
    raw_row: object,
    *,
    context: str,
    data_root: Path | None,
    vocabulary: Vocabulary,
) -> tuple[str, str, int, int, _RuntimeRecord]:
    if not isinstance(raw_row, Mapping):
        raise CLEVRDatasetError(f"{context} must contain a mapping")
    if set(raw_row) != _ROW_FIELDS:
        missing = sorted(_ROW_FIELDS.difference(raw_row))
        extra = sorted(set(raw_row).difference(_ROW_FIELDS))
        raise CLEVRDatasetError(
            f"{context} fields disagree with the contract; missing={missing}, extra={extra}"
        )

    partition = _require_string(raw_row, "partition", context)
    if partition not in _PARTITION_SPLITS:
        raise CLEVRDatasetError(f"{context} has an unknown partition {partition!r}")
    official_split = _require_string(raw_row, "official_split", context)
    expected_split = _PARTITION_SPLITS[partition]
    if official_split != expected_split:
        raise CLEVRDatasetError(
            f"{context} partition {partition} requires official_split {expected_split}"
        )
    image_index = _require_index(raw_row, "image_index", context)
    question_index = _require_index(raw_row, "question_index", context)
    source_position = _require_index(raw_row, "source_position", context)
    image_filename = _require_string(raw_row, "image_filename", context)
    try:
        ImageIdentity(official_split, image_index, image_filename)
    except ValueError as exc:
        raise CLEVRDatasetError(f"{context} has an invalid image identity: {exc}") from exc
    serialized_image_path = _require_string(raw_row, "image_path", context)
    expected_image_path = f"images/{official_split}/{image_filename}"
    if serialized_image_path != expected_image_path:
        raise CLEVRDatasetError(
            f"{context} image_path must be the canonical relative POSIX path "
            f"{expected_image_path!r}"
        )
    image_path = None
    if data_root is not None:
        image_path = _resolve_image_file(
            data_root / serialized_image_path,
            data_root=data_root,
            context=context,
        )

    question = _require_string(raw_row, "question", context)
    relation = _require_string(raw_row, "relation", context)
    if relation not in _RELATIONS:
        raise CLEVRDatasetError(f"{context} has an invalid relation {relation!r}")
    answer = raw_row["answer"]
    if type(answer) is not int or answer not in {0, 1}:
        raise CLEVRDatasetError(f"{context} answer must be integer 0 or 1")

    raw_tokens = raw_row["encoded_tokens"]
    if not isinstance(raw_tokens, list) or any(
        type(token_id) is not int for token_id in raw_tokens
    ):
        raise CLEVRDatasetError(f"{context} encoded_tokens must be a list of integers")
    encoded_tokens = tuple(raw_tokens)
    if len(encoded_tokens) < 2:
        raise CLEVRDatasetError(f"{context} encoded_tokens must contain START and END")
    if encoded_tokens[0] != 1:
        raise CLEVRDatasetError(f"{context} encoded_tokens must begin with START")
    if encoded_tokens[-1] != 2:
        raise CLEVRDatasetError(f"{context} encoded_tokens must end with END")
    if 0 in encoded_tokens:
        raise CLEVRDatasetError(f"{context} encoded_tokens must not contain NULL")
    if any(token_id < 0 or token_id >= len(vocabulary.id_to_token) for token_id in encoded_tokens):
        raise CLEVRDatasetError(f"{context} encoded_tokens contain an out-of-range token")
    if encoded_tokens != encode_question(question, vocabulary):
        raise CLEVRDatasetError(
            f"{context} encoded_tokens disagree with the original question"
        )

    return (
        partition,
        official_split,
        question_index,
        source_position,
        _RuntimeRecord(
            partition=partition,
            official_split=official_split,
            image_filename=image_filename,
            image_path=image_path,
            image_index=image_index,
            question_index=question_index,
            source_position=source_position,
            encoded_tokens=encoded_tokens,
            answer=answer,
        ),
    )


def _load_records(
    path: Path,
    *,
    data_root: Path | None,
    vocabulary: Vocabulary,
    report_counts: Mapping[str, int],
    selected_partition: str,
) -> tuple[_RuntimeRecord, ...]:
    if not path.is_file():
        raise CLEVRDatasetError(f"required selected questions artifact is missing: {path}")
    observed_counts: Counter[str] = Counter()
    question_ids: set[tuple[str, int]] = set()
    source_ids: set[tuple[str, int]] = set()
    partition_by_image: dict[tuple[str, int], str] = {}
    selected: list[_RuntimeRecord] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                context = f"selected questions line {line_number}"
                if not line.strip():
                    raise CLEVRDatasetError(f"{context} is blank")
                try:
                    raw_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CLEVRDatasetError(f"{context} contains invalid JSON: {exc}") from exc
                partition, split, question_index, source_position, record = _parse_row(
                    raw_row,
                    context=context,
                    data_root=data_root,
                    vocabulary=vocabulary,
                )
                question_key = (split, question_index)
                if question_key in question_ids:
                    raise CLEVRDatasetError(
                        f"{context} has duplicate question_index for split {split}"
                    )
                source_key = (split, source_position)
                if source_key in source_ids:
                    raise CLEVRDatasetError(
                        f"{context} has duplicate source_position for split {split}"
                    )
                image_key = (split, record.image_index)
                previous_partition = partition_by_image.get(image_key)
                if (
                    previous_partition is not None
                    and previous_partition != partition
                ):
                    raise CLEVRDatasetError(
                        f"{context} assigns {split} image {record.image_index} to "
                        f"multiple partitions: {previous_partition} and {partition}"
                    )
                question_ids.add(question_key)
                source_ids.add(source_key)
                partition_by_image[image_key] = partition
                observed_counts[partition] += 1
                if partition == selected_partition:
                    selected.append(record)
    except (OSError, UnicodeError) as exc:
        raise CLEVRDatasetError(f"cannot read selected questions artifact {path}: {exc}") from exc

    for partition in _PARTITION_SPLITS:
        if observed_counts[partition] != report_counts[partition]:
            raise CLEVRDatasetError(
                f"selected question count for {partition} is {observed_counts[partition]}, "
                f"expected {report_counts[partition]}"
            )
    if not selected:
        raise CLEVRDatasetError(f"requested partition {selected_partition} is empty")
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.image_index,
                item.question_index,
                item.source_position,
            ),
        )
    )


class CLEVRQuestionDataset(Dataset[dict[str, Tensor]]):
    """Indexable RGB/question samples from an explicit accepted audit."""

    def __init__(
        self,
        *,
        data_root: Path,
        audit_dir: Path,
        partition: str,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        if not self.data_root.is_dir():
            raise CLEVRDatasetError(
                f"CLEVR data root is not a directory: {self.data_root}"
            )
        self.audit_dir = Path(audit_dir).resolve()
        if not self.audit_dir.is_dir():
            raise CLEVRDatasetError(
                f"CLEVR audit directory is not a directory: {self.audit_dir}"
            )
        if partition not in _PARTITION_SPLITS:
            raise CLEVRDatasetError(f"unknown CLEVR partition: {partition!r}")

        vocabulary = _load_vocabulary(self.audit_dir / "vocabulary.json")
        self.vocab_size = len(vocabulary.id_to_token)
        report_counts = _load_report_counts(self.audit_dir / "audit_report.json")
        self._records = _load_records(
            self.audit_dir / "selected_questions.jsonl",
            data_root=self.data_root,
            vocabulary=vocabulary,
            report_counts=report_counts,
            selected_partition=partition,
        )

    def __len__(self) -> int:
        return len(self._records)

    def sample_identity(self, index: int) -> CLEVRSampleIdentity:
        """Project validated provenance without accessing the image filesystem."""
        record = self._records[index]
        return CLEVRSampleIdentity(
            partition=record.partition,
            official_split=record.official_split,
            image_index=record.image_index,
            image_filename=record.image_filename,
            question_index=record.question_index,
            source_position=record.source_position,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self._records[index]
        if record.image_path is None:
            raise CLEVRDatasetError(
                f"selected RGB index {index} has no resolved image_path"
            )
        image_path = _resolve_image_file(
            record.image_path,
            data_root=self.data_root,
            context=f"selected index {index}",
        )
        try:
            with Image.open(image_path, formats=("PNG",)) as source:
                image = (
                    tvf.pil_to_tensor(source.convert("RGB")).to(torch.float32)
                    / 255.0
                )
            image = tvf.resize(
                image,
                [224, 224],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise CLEVRDatasetError(
                f"cannot decode selected index {index} as PNG: {image_path}"
            ) from exc
        if image.dtype != torch.float32 or tuple(image.shape) != (3, 224, 224):
            raise CLEVRDatasetError(
                f"decoded image for selected index {index} has an invalid tensor "
                f"contract: {image_path}"
            )
        if not bool(torch.isfinite(image).all()):
            raise CLEVRDatasetError(
                f"decoded image for selected index {index} is not finite: "
                f"{image_path}"
            )
        if bool((image < -_FLOAT32_UNIT_INTERVAL_TOLERANCE).any()) or bool(
            (image > 1 + _FLOAT32_UNIT_INTERVAL_TOLERANCE).any()
        ):
            raise CLEVRDatasetError(
                f"decoded image for selected index {index} is outside the "
                f"float32 tolerance of [0,1]: "
                f"{image_path}"
            )
        tokens = torch.tensor(record.encoded_tokens, dtype=torch.int64)
        return {
            "image": image,
            "question_tokens": tokens,
            "question_length": torch.tensor(tokens.numel(), dtype=torch.int64),
            "answer": torch.tensor(record.answer, dtype=torch.int64),
        }


class CLEVRQuestionOnlyDataset(Dataset[dict[str, Tensor]]):
    """Indexable question-only samples from an explicit accepted audit."""

    def __init__(self, *, audit_dir: Path, partition: str) -> None:
        self.audit_dir = Path(audit_dir).resolve()
        if not self.audit_dir.is_dir():
            raise CLEVRDatasetError(
                f"CLEVR audit directory is not a directory: {self.audit_dir}"
            )
        if partition not in _PARTITION_SPLITS:
            raise CLEVRDatasetError(f"unknown CLEVR partition: {partition!r}")

        vocabulary = _load_vocabulary(self.audit_dir / "vocabulary.json")
        self.vocab_size = len(vocabulary.id_to_token)
        report_counts = _load_report_counts(self.audit_dir / "audit_report.json")
        self._records = _load_records(
            self.audit_dir / "selected_questions.jsonl",
            data_root=None,
            vocabulary=vocabulary,
            report_counts=report_counts,
            selected_partition=partition,
        )

    def __len__(self) -> int:
        return len(self._records)

    def sample_identity(self, index: int) -> CLEVRSampleIdentity:
        """Project validated provenance without accessing the image filesystem."""
        record = self._records[index]
        return CLEVRSampleIdentity(
            partition=record.partition,
            official_split=record.official_split,
            image_index=record.image_index,
            image_filename=record.image_filename,
            question_index=record.question_index,
            source_position=record.source_position,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self._records[index]
        tokens = torch.tensor(record.encoded_tokens, dtype=torch.int64)
        return {
            "question_tokens": tokens,
            "question_length": torch.tensor(tokens.numel(), dtype=torch.int64),
            "answer": torch.tensor(record.answer, dtype=torch.int64),
        }


def collate_clevr_questions(
    samples: Sequence[Mapping[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad audited question samples without reordering or truncation."""
    if not samples:
        raise CLEVRDatasetError("cannot collate an empty CLEVR sample sequence")
    validated = [
        _validate_collate_sample(sample, index=index)
        for index, sample in enumerate(samples)
    ]
    return {
        "image": torch.stack([sample[0] for sample in validated]),
        "question_tokens": pad_sequence(
            [sample[1] for sample in validated],
            batch_first=True,
            padding_value=0,
        ),
        "question_length": torch.stack([sample[2] for sample in validated]),
        "answer": torch.stack([sample[3] for sample in validated]),
    }


def collate_clevr_question_only(
    samples: Sequence[Mapping[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad audited question-only samples without image materialization."""
    if not samples:
        raise CLEVRDatasetError(
            "cannot collate an empty CLEVR question-only sample sequence"
        )
    validated = [
        _validate_question_only_collate_sample(sample, index=index)
        for index, sample in enumerate(samples)
    ]
    return {
        "question_tokens": pad_sequence(
            [sample[0] for sample in validated],
            batch_first=True,
            padding_value=0,
        ),
        "question_length": torch.stack([sample[1] for sample in validated]),
        "answer": torch.stack([sample[2] for sample in validated]),
    }


def _validate_collate_sample(
    sample: Mapping[str, Tensor], *, index: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    context = f"CLEVR collate sample {index}"
    if not isinstance(sample, Mapping):
        raise CLEVRDatasetError(f"{context} must be a mapping")
    if set(sample) != _MODEL_FIELDS:
        missing = sorted(_MODEL_FIELDS.difference(sample))
        extra = sorted(set(sample).difference(_MODEL_FIELDS))
        raise CLEVRDatasetError(
            f"{context} keys disagree with the model contract; "
            f"missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, Tensor) for value in sample.values()):
        raise CLEVRDatasetError(f"{context} values must all be tensors")

    image = sample["image"]
    tokens = sample["question_tokens"]
    length = sample["question_length"]
    answer = sample["answer"]

    if image.dtype != torch.float32 or tuple(image.shape) != (3, 224, 224):
        raise CLEVRDatasetError(
            f"{context} image must have dtype float32 and shape [3,224,224]"
        )
    if not bool(torch.isfinite(image).all()):
        raise CLEVRDatasetError(f"{context} image must be finite")
    if bool((image < -_FLOAT32_UNIT_INTERVAL_TOLERANCE).any()) or bool(
        (image > 1 + _FLOAT32_UNIT_INTERVAL_TOLERANCE).any()
    ):
        raise CLEVRDatasetError(
            f"{context} image must be within the float32 tolerance of [0,1]"
        )

    tokens, length, answer = _validate_question_tensors(
        tokens,
        length,
        answer,
        context=context,
    )
    return image, tokens, length, answer


def _validate_question_only_collate_sample(
    sample: Mapping[str, Tensor], *, index: int
) -> tuple[Tensor, Tensor, Tensor]:
    context = f"CLEVR question-only collate sample {index}"
    if not isinstance(sample, Mapping):
        raise CLEVRDatasetError(f"{context} must be a mapping")
    if set(sample) != _QUESTION_ONLY_MODEL_FIELDS:
        missing = sorted(_QUESTION_ONLY_MODEL_FIELDS.difference(sample))
        extra = sorted(set(sample).difference(_QUESTION_ONLY_MODEL_FIELDS))
        raise CLEVRDatasetError(
            f"{context} keys disagree with the model contract; "
            f"missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, Tensor) for value in sample.values()):
        raise CLEVRDatasetError(f"{context} values must all be tensors")
    return _validate_question_tensors(
        sample["question_tokens"],
        sample["question_length"],
        sample["answer"],
        context=context,
    )


def _validate_question_tensors(
    tokens: Tensor,
    length: Tensor,
    answer: Tensor,
    *,
    context: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if tokens.dtype != torch.int64 or tokens.ndim != 1:
        raise CLEVRDatasetError(
            f"{context} question_tokens must have dtype int64 and rank 1"
        )
    if tokens.numel() < 2:
        raise CLEVRDatasetError(
            f"{context} question_tokens must contain START and END"
        )
    if tokens[0].item() != 1:
        raise CLEVRDatasetError(f"{context} question_tokens must begin with START")
    if tokens[-1].item() != 2:
        raise CLEVRDatasetError(f"{context} question_tokens must end with END")
    if bool((tokens == 0).any()):
        raise CLEVRDatasetError(
            f"{context} question_tokens must not contain NULL before padding"
        )
    if bool((tokens < 0).any()):
        raise CLEVRDatasetError(
            f"{context} question_tokens must not contain negative IDs"
        )

    if length.dtype != torch.int64 or length.ndim != 0:
        raise CLEVRDatasetError(
            f"{context} question_length must be an int64 scalar"
        )
    if length.item() != tokens.numel():
        raise CLEVRDatasetError(
            f"{context} question_length disagrees with question_tokens"
        )
    if answer.dtype != torch.int64 or answer.ndim != 0:
        raise CLEVRDatasetError(f"{context} answer must be an int64 scalar")
    if answer.item() not in {0, 1}:
        raise CLEVRDatasetError(f"{context} answer must be 0 or 1")
    return tokens, length, answer
