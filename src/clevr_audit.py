"""Read-only validation and audit records for user-provided CLEVR files."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from PIL import Image, UnidentifiedImageError

from src.clevr_partitions import (
    ImageIdentity,
    ImagePartitions,
    build_image_partitions,
)
from src.clevr_programs import (
    AnswerMismatchError,
    ProgramNode,
    ProgramSchemaError,
    SceneSchemaError,
    audit_official_answer,
    parse_program,
    select_single_relation_exist,
    validate_scene,
)
from src.clevr_text import Vocabulary, build_vocabulary, encode_question


class CLEVRDataError(ValueError):
    """Raised when local CLEVR source files violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class SourceScene:
    official_split: str
    image_index: int
    image_filename: str
    source_position: int
    image_path: Path
    scene_data: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SourceQuestion:
    official_split: str
    question_index: int
    image_index: int
    image_filename: str
    source_position: int
    question: str
    program: tuple[ProgramNode, ...]
    answer: str


@dataclass(frozen=True, slots=True)
class SourceIndex:
    data_root: Path
    scenes: tuple[SourceScene, ...]
    questions: tuple[SourceQuestion, ...]


_PARTITIONS = ("train", "dev", "base_test", "confirmation")
_RELATIONS = ("left", "right", "front", "behind")


@dataclass(frozen=True, slots=True)
class SelectedQuestion:
    """Offline-only selected-question manifest row, not a model batch item."""

    partition: str
    official_split: str
    image_index: int
    image_filename: str
    image_path: Path
    question_index: int
    source_position: int
    question: str
    encoded_tokens: tuple[int, ...]
    answer: int
    relation: str

    def __post_init__(self) -> None:
        if self.partition not in _PARTITIONS:
            raise ValueError("partition is not part of the frozen protocol")
        if self.official_split not in {"train", "val"}:
            raise ValueError("official_split must be exactly 'train' or 'val'")
        if self.answer not in {0, 1}:
            raise ValueError("answer must use no=0 / yes=1")
        if self.relation not in _RELATIONS:
            raise ValueError("relation is not part of the frozen task")
        if not isinstance(self.encoded_tokens, tuple) or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.encoded_tokens
        ):
            raise ValueError("encoded_tokens must be non-negative integer IDs")


@dataclass(frozen=True, slots=True)
class CellCount:
    partition: str
    relation: str
    answer: int
    count: int
    accuracy: float | None = None


@dataclass(frozen=True, slots=True)
class ExclusionCount:
    partition: str
    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    partition: str
    mother_image_count: int
    selected_image_count: int
    selected_question_count: int


@dataclass(frozen=True, slots=True)
class SourceSummary:
    official_split: str
    scene_count: int
    question_count: int
    image_count: int
    image_bytes: int


@dataclass(frozen=True, slots=True)
class SourceFileSummary:
    relative_path: str
    size_bytes: int
    record_count: int


@dataclass(frozen=True, slots=True)
class AuditResult:
    selected_questions: tuple[SelectedQuestion, ...]
    vocabulary: Vocabulary
    image_partitions: ImagePartitions
    cell_counts: tuple[CellCount, ...]
    exclusion_counts: tuple[ExclusionCount, ...]
    partition_summaries: tuple[PartitionSummary, ...]
    source_summaries: tuple[SourceSummary, ...]
    file_summaries: tuple[SourceFileSummary, ...]


@dataclass(frozen=True, slots=True)
class _PendingQuestion:
    partition: str
    source: SourceQuestion
    scene: SourceScene
    answer: int
    relation: str


def _read_collection(path: Path, field: str) -> list[object]:
    if not path.is_file():
        raise CLEVRDataError(f"required CLEVR file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as source_file:
            document = json.load(source_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CLEVRDataError(f"cannot parse CLEVR JSON file {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CLEVRDataError(f"CLEVR JSON file {path} must contain a mapping")
    records = document.get(field)
    if not isinstance(records, list):
        raise CLEVRDataError(f"CLEVR JSON file {path} field {field!r} must be a list")
    return records


def _require_mapping(record: object, context: str) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise CLEVRDataError(f"{context} must be a mapping")
    return record


def _require_string(record: Mapping[str, object], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise CLEVRDataError(f"{context} {field} must be a string")
    return value


def _require_index(record: Mapping[str, object], field: str, context: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise CLEVRDataError(f"{context} {field} must be a non-negative integer")
    return value


def _freeze_json(value: object, context: str) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise CLEVRDataError(f"{context} contains a non-string mapping key")
            frozen[key] = _freeze_json(nested, context)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_json(nested, context) for nested in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise CLEVRDataError(f"{context} contains a non-JSON value")


def _verify_png(path: Path) -> None:
    if not path.is_file():
        raise CLEVRDataError(f"referenced CLEVR image is missing: {path}")
    try:
        with Image.open(path, formats=("PNG",)) as image:
            image.verify()
        with Image.open(path, formats=("PNG",)) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise CLEVRDataError(f"unreadable PNG {path}: {exc}") from exc


def _load_scenes(
    root: Path, split: str, raw_scenes: list[object]
) -> tuple[tuple[SourceScene, ...], dict[int, SourceScene]]:
    scenes: list[SourceScene] = []
    by_index: dict[int, SourceScene] = {}
    filenames: set[str] = set()
    for source_position, raw_scene in enumerate(raw_scenes):
        context = f"{split} scene {source_position}"
        scene = _require_mapping(raw_scene, context)
        source_split = _require_string(scene, "split", context)
        if source_split != split:
            raise CLEVRDataError(
                f"{context} split is {source_split!r}, expected {split!r}"
            )
        image_index = _require_index(scene, "image_index", context)
        image_filename = _require_string(scene, "image_filename", context)
        if image_index in by_index:
            raise CLEVRDataError(f"duplicate {split} scene image_index: {image_index}")
        if image_filename in filenames:
            raise CLEVRDataError(
                f"duplicate {split} scene image_filename: {image_filename}"
            )
        try:
            ImageIdentity(split, image_index, image_filename)
        except ValueError as exc:
            raise CLEVRDataError(f"{context}: {exc}") from exc
        if not isinstance(scene.get("objects"), list):
            raise CLEVRDataError(f"{context} objects must be a list")
        if not isinstance(scene.get("relationships"), Mapping):
            raise CLEVRDataError(f"{context} relationships must be a mapping")
        try:
            validate_scene(scene)
        except SceneSchemaError as exc:
            raise CLEVRDataError(f"{context} has an invalid scene: {exc}") from exc

        image_path = root / "images" / split / image_filename
        _verify_png(image_path)
        frozen_scene = _freeze_json(scene, context)
        if not isinstance(frozen_scene, Mapping):
            raise AssertionError("mapping freeze must preserve the outer mapping")
        record = SourceScene(
            official_split=split,
            image_index=image_index,
            image_filename=image_filename,
            source_position=source_position,
            image_path=image_path,
            scene_data=frozen_scene,
        )
        scenes.append(record)
        by_index[image_index] = record
        filenames.add(image_filename)
    return tuple(scenes), by_index


def _load_questions(
    split: str,
    raw_questions: list[object],
    scenes_by_index: Mapping[int, SourceScene],
) -> tuple[SourceQuestion, ...]:
    questions: list[SourceQuestion] = []
    question_indices: set[int] = set()
    for source_position, raw_question in enumerate(raw_questions):
        context = f"{split} question {source_position}"
        question = _require_mapping(raw_question, context)
        source_split = _require_string(question, "split", context)
        if source_split != split:
            raise CLEVRDataError(
                f"{context} split is {source_split!r}, expected {split!r}"
            )
        question_index = _require_index(question, "question_index", context)
        if question_index in question_indices:
            raise CLEVRDataError(
                f"duplicate {split} question question_index: {question_index}"
            )
        image_index = _require_index(question, "image_index", context)
        image_filename = _require_string(question, "image_filename", context)
        source_scene = scenes_by_index.get(image_index)
        if source_scene is None:
            raise CLEVRDataError(
                f"{context} has no matching scene for image_index {image_index}"
            )
        if image_filename != source_scene.image_filename:
            raise CLEVRDataError(
                f"{context} image_index {image_index} and image_filename "
                f"{image_filename!r} disagree with the scene"
            )
        question_text = _require_string(question, "question", context)
        answer = _require_string(question, "answer", context)
        if not answer:
            raise CLEVRDataError(f"{context} answer must be a non-empty string")
        try:
            program = parse_program(question.get("program"))
        except ProgramSchemaError as exc:
            raise CLEVRDataError(f"{context} has an invalid program: {exc}") from exc

        record = SourceQuestion(
            official_split=split,
            question_index=question_index,
            image_index=image_index,
            image_filename=image_filename,
            source_position=source_position,
            question=question_text,
            program=program,
            answer=answer,
        )
        questions.append(record)
        question_indices.add(question_index)
    return tuple(questions)


def load_source_index(data_root: Path) -> SourceIndex:
    """Validate and join the explicit local CLEVR train/val source root."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise CLEVRDataError(f"CLEVR data root is not a directory: {root}")

    all_scenes: list[SourceScene] = []
    all_questions: list[SourceQuestion] = []
    for split in ("train", "val"):
        scene_path = root / "scenes" / f"CLEVR_{split}_scenes.json"
        question_path = root / "questions" / f"CLEVR_{split}_questions.json"
        raw_scenes = _read_collection(scene_path, "scenes")
        raw_questions = _read_collection(question_path, "questions")
        scenes, scenes_by_index = _load_scenes(root, split, raw_scenes)
        questions = _load_questions(split, raw_questions, scenes_by_index)
        all_scenes.extend(scenes)
        all_questions.extend(questions)

    return SourceIndex(
        data_root=root,
        scenes=tuple(all_scenes),
        questions=tuple(all_questions),
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


def _partition_lookup(
    partitions: ImagePartitions,
) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for partition in _PARTITIONS:
        for image in getattr(partitions, partition):
            key = (image.official_split, image.image_index)
            if key in lookup:
                raise CLEVRDataError(f"image {key} appears in multiple partitions")
            lookup[key] = partition
    return lookup


def _source_file_summaries(source: SourceIndex) -> tuple[SourceFileSummary, ...]:
    summaries: list[SourceFileSummary] = []
    for split in ("train", "val"):
        entries = (
            (
                f"scenes/CLEVR_{split}_scenes.json",
                sum(scene.official_split == split for scene in source.scenes),
            ),
            (
                f"questions/CLEVR_{split}_questions.json",
                sum(question.official_split == split for question in source.questions),
            ),
        )
        for relative_path, record_count in entries:
            path = source.data_root / relative_path
            try:
                size_bytes = path.stat().st_size
            except OSError as exc:
                raise CLEVRDataError(f"cannot stat CLEVR source file {path}: {exc}") from exc
            summaries.append(
                SourceFileSummary(
                    relative_path=relative_path,
                    size_bytes=size_bytes,
                    record_count=record_count,
                )
            )
    return tuple(summaries)


def _source_summaries(source: SourceIndex) -> tuple[SourceSummary, ...]:
    summaries: list[SourceSummary] = []
    for split in ("train", "val"):
        scenes = tuple(
            scene for scene in source.scenes if scene.official_split == split
        )
        try:
            image_bytes = sum(scene.image_path.stat().st_size for scene in scenes)
        except OSError as exc:
            raise CLEVRDataError(f"cannot stat a {split} CLEVR image: {exc}") from exc
        summaries.append(
            SourceSummary(
                official_split=split,
                scene_count=len(scenes),
                question_count=sum(
                    question.official_split == split for question in source.questions
                ),
                image_count=len(scenes),
                image_bytes=image_bytes,
            )
        )
    return tuple(summaries)


def _enforce_coverage(selected_questions: tuple[SelectedQuestion, ...]) -> None:
    partition_counts = Counter(item.partition for item in selected_questions)
    for partition in _PARTITIONS:
        if partition_counts[partition] == 0:
            raise CLEVRDataError(f"required partition {partition} is empty")

    train_answers = {
        item.answer for item in selected_questions if item.partition == "train"
    }
    for answer, name in ((0, "no"), (1, "yes")):
        if answer not in train_answers:
            raise CLEVRDataError(f"train is missing answer class {name}")

    train_relations = {
        item.relation for item in selected_questions if item.partition == "train"
    }
    for relation in _RELATIONS:
        if relation not in train_relations:
            raise CLEVRDataError(f"train is missing relation {relation}")


def build_audit(source: SourceIndex) -> AuditResult:
    """Select, answer-audit, partition, and encode the frozen CLEVR task."""
    if not isinstance(source, SourceIndex):
        raise TypeError("source must be a SourceIndex")

    train_images = tuple(
        ImageIdentity(scene.official_split, scene.image_index, scene.image_filename)
        for scene in source.scenes
        if scene.official_split == "train"
    )
    val_images = tuple(
        ImageIdentity(scene.official_split, scene.image_index, scene.image_filename)
        for scene in source.scenes
        if scene.official_split == "val"
    )
    image_partitions = build_image_partitions(train_images, val_images)
    partition_by_image = _partition_lookup(image_partitions)
    scene_by_image = {
        (scene.official_split, scene.image_index): scene for scene in source.scenes
    }

    pending: list[_PendingQuestion] = []
    exclusions: Counter[tuple[str, str]] = Counter()
    for question in source.questions:
        image_key = (question.official_split, question.image_index)
        partition = partition_by_image[image_key]
        try:
            decision = select_single_relation_exist(question.program, question.answer)
        except ProgramSchemaError as exc:
            raise CLEVRDataError(
                f"{question.official_split} question {question.question_index} "
                f"violates the program contract: {exc}"
            ) from exc
        if decision.exclusion_reason is not None:
            exclusions[(partition, decision.exclusion_reason)] += 1
            continue

        scene = scene_by_image[image_key]
        try:
            audit_official_answer(
                question.program, _thaw_json(scene.scene_data), question.answer
            )
        except (ProgramSchemaError, SceneSchemaError, AnswerMismatchError) as exc:
            raise CLEVRDataError(
                f"{question.official_split} question {question.question_index} "
                f"failed official answer audit: {exc}"
            ) from exc
        if decision.relation is None or question.answer not in {"no", "yes"}:
            raise AssertionError("selected questions must have a relation and binary answer")
        pending.append(
            _PendingQuestion(
                partition=partition,
                source=question,
                scene=scene,
                answer=1 if question.answer == "yes" else 0,
                relation=decision.relation,
            )
        )

    vocabulary = build_vocabulary(
        tuple(
            item.source.question for item in pending if item.partition == "train"
        )
    )
    selected_questions = tuple(
        SelectedQuestion(
            partition=item.partition,
            official_split=item.source.official_split,
            image_index=item.source.image_index,
            image_filename=item.source.image_filename,
            image_path=item.scene.image_path,
            question_index=item.source.question_index,
            source_position=item.source.source_position,
            question=item.source.question,
            encoded_tokens=encode_question(item.source.question, vocabulary),
            answer=item.answer,
            relation=item.relation,
        )
        for item in pending
    )
    _enforce_coverage(selected_questions)

    observed_cells = Counter(
        (item.partition, item.relation, item.answer) for item in selected_questions
    )
    cell_counts = tuple(
        CellCount(
            partition=partition,
            relation=relation,
            answer=answer,
            count=observed_cells[(partition, relation, answer)],
        )
        for partition in _PARTITIONS
        for relation in _RELATIONS
        for answer in (0, 1)
    )
    exclusion_counts = tuple(
        ExclusionCount(partition=partition, reason=reason, count=count)
        for partition in _PARTITIONS
        for reason, count in sorted(
            (
                (reason, count)
                for (observed_partition, reason), count in exclusions.items()
                if observed_partition == partition
            )
        )
    )
    partition_summaries = tuple(
        PartitionSummary(
            partition=partition,
            mother_image_count=len(getattr(image_partitions, partition)),
            selected_image_count=len(
                {
                    (item.official_split, item.image_index)
                    for item in selected_questions
                    if item.partition == partition
                }
            ),
            selected_question_count=sum(
                item.partition == partition for item in selected_questions
            ),
        )
        for partition in _PARTITIONS
    )
    return AuditResult(
        selected_questions=selected_questions,
        vocabulary=vocabulary,
        image_partitions=image_partitions,
        cell_counts=cell_counts,
        exclusion_counts=exclusion_counts,
        partition_summaries=partition_summaries,
        source_summaries=_source_summaries(source),
        file_summaries=_source_file_summaries(source),
    )
