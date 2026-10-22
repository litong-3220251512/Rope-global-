"""Read-only validation and audit records for user-provided CLEVR files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from PIL import Image, UnidentifiedImageError

from src.clevr_partitions import ImageIdentity
from src.clevr_programs import ProgramNode, ProgramSchemaError, parse_program


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
    answer: object


@dataclass(frozen=True, slots=True)
class SourceIndex:
    data_root: Path
    scenes: tuple[SourceScene, ...]
    questions: tuple[SourceQuestion, ...]


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
            answer=question.get("answer"),
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
