from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

from PIL import Image

from run_clevr_data_audit import main as audit_main
from src.clevr_audit import (
    CLEVRDataError,
    SelectedQuestion,
    build_audit,
    load_source_index,
)


def _program() -> list[dict[str, object]]:
    return [
        {"function": "scene", "inputs": [], "value_inputs": []},
        {"function": "unique", "inputs": [0], "value_inputs": []},
        {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
        {"function": "exist", "inputs": [2], "value_inputs": []},
    ]


def _scene(split: str, image_index: int) -> dict[str, object]:
    return {
        "split": split,
        "image_index": image_index,
        "image_filename": f"CLEVR_{split}_{image_index:06d}.png",
        "objects": [
            {
                "color": "red",
                "shape": "cube",
                "size": "small",
                "material": "rubber",
            }
        ],
        "relationships": {
            "left": [[]],
            "right": [[]],
            "front": [[]],
            "behind": [[]],
        },
    }


def _audit_scene(split: str, image_index: int) -> dict[str, object]:
    scene = _scene(split, image_index)
    scene["objects"] = [
        {
            "color": "red",
            "shape": "cube",
            "size": "small",
            "material": "rubber",
        },
        {
            "color": "blue",
            "shape": "sphere",
            "size": "large",
            "material": "metal",
        },
    ]
    scene["relationships"] = {
        relation: [[1], []]
        for relation in ("left", "right", "front", "behind")
    }
    return scene


def _selected_program(relation: str, target_shape: str) -> list[dict[str, object]]:
    return [
        {"function": "scene", "inputs": [], "value_inputs": []},
        {
            "function": "filter_color",
            "inputs": [0],
            "value_inputs": ["red"],
        },
        {"function": "unique", "inputs": [1], "value_inputs": []},
        {"function": "relate", "inputs": [2], "value_inputs": [relation]},
        {
            "function": "filter_shape",
            "inputs": [3],
            "value_inputs": [target_shape],
        },
        {"function": "exist", "inputs": [4], "value_inputs": []},
    ]


def _excluded_program(reason: str) -> list[dict[str, object]]:
    if reason == "unsupported_function":
        return [
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "count", "inputs": [0], "value_inputs": []},
        ]
    if reason == "relate_count":
        return [
            {"function": "scene", "inputs": [], "value_inputs": []},
            {
                "function": "filter_color",
                "inputs": [0],
                "value_inputs": ["red"],
            },
            {"function": "unique", "inputs": [1], "value_inputs": []},
            {"function": "exist", "inputs": [2], "value_inputs": []},
        ]
    raise ValueError(f"unknown fixture exclusion reason: {reason}")


def _question(
    split: str,
    question_index: int,
    image_index: int,
) -> dict[str, object]:
    return {
        "split": split,
        "question_index": question_index,
        "image_index": image_index,
        "image_filename": f"CLEVR_{split}_{image_index:06d}.png",
        "question": "Is anything left of the red cube?",
        "program": _program(),
        "answer": "no",
    }


class _CLEVRFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.scenes = {
            split: [_scene(split, 0)] for split in ("train", "val")
        }
        self.questions = {
            split: [_question(split, 0, 0)] for split in ("train", "val")
        }
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "questions").mkdir(parents=True, exist_ok=True)
            (root / "scenes").mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(
                root / "images" / split / f"CLEVR_{split}_000000.png"
            )
        self.write_json()

    def write_json(self) -> None:
        for split in ("train", "val"):
            (self.root / "scenes" / f"CLEVR_{split}_scenes.json").write_text(
                json.dumps({"scenes": self.scenes[split]}), encoding="utf-8"
            )
            (
                self.root / "questions" / f"CLEVR_{split}_questions.json"
            ).write_text(
                json.dumps({"questions": self.questions[split]}), encoding="utf-8"
            )


class _AuditFixture(_CLEVRFixture):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.scenes = {
            "train": [_audit_scene("train", index) for index in range(10)],
            "val": [_audit_scene("val", index) for index in range(4)],
        }
        self.questions = {"train": [], "val": []}
        for split, count in (("train", 10), ("val", 4)):
            for image_index in range(count):
                Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(
                    root
                    / "images"
                    / split
                    / f"CLEVR_{split}_{image_index:06d}.png"
                )

        for relation in ("left", "right", "front", "behind"):
            for answer, target_shape in (("no", "cube"), ("yes", "sphere")):
                self.add_selected(
                    split="train",
                    image_index=3,
                    relation=relation,
                    answer=answer,
                    target_shape=target_shape,
                    question=f"trainword {relation} {answer}?",
                )
        self.add_excluded("train", 3, "unsupported_function", "train excluded?")
        self.add_selected(
            "train", 6, "left", "yes", "sphere", "devonly left yes?"
        )
        self.add_excluded("train", 6, "relate_count", "dev excluded?")
        self.add_selected(
            "val", 1, "left", "yes", "sphere", "baseonly left yes?"
        )
        self.add_excluded("val", 1, "unsupported_function", "base excluded?")
        self.add_selected(
            "val", 3, "right", "no", "cube", "confirmonly right no?"
        )
        self.write_json()

    def add_selected(
        self,
        split: str,
        image_index: int,
        relation: str,
        answer: str,
        target_shape: str,
        question: str,
    ) -> None:
        question_index = len(self.questions[split])
        record = _question(split, question_index, image_index)
        record["question"] = question
        record["program"] = _selected_program(relation, target_shape)
        record["answer"] = answer
        self.questions[split].append(record)

    def add_excluded(
        self,
        split: str,
        image_index: int,
        reason: str,
        question: str,
    ) -> None:
        question_index = len(self.questions[split])
        record = _question(split, question_index, image_index)
        record["question"] = question
        record["program"] = _excluded_program(reason)
        record["answer"] = 2 if reason == "unsupported_function" else "yes"
        self.questions[split].append(record)


class SourceIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = _CLEVRFixture(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_files_join_into_immutable_traceable_records(self) -> None:
        source = load_source_index(self.root)

        self.assertEqual(source.data_root, self.root.resolve())
        self.assertEqual(
            [
                (
                    scene.official_split,
                    scene.image_index,
                    scene.image_filename,
                    scene.source_position,
                    scene.image_path,
                )
                for scene in source.scenes
            ],
            [
                (
                    "train",
                    0,
                    "CLEVR_train_000000.png",
                    0,
                    self.root.resolve()
                    / "images"
                    / "train"
                    / "CLEVR_train_000000.png",
                ),
                (
                    "val",
                    0,
                    "CLEVR_val_000000.png",
                    0,
                    self.root.resolve()
                    / "images"
                    / "val"
                    / "CLEVR_val_000000.png",
                ),
            ],
        )
        self.assertEqual(
            [
                (
                    question.official_split,
                    question.question_index,
                    question.image_index,
                    question.image_filename,
                    question.source_position,
                    question.answer,
                    question.program[2].value_inputs,
                )
                for question in source.questions
            ],
            [
                ("train", 0, 0, "CLEVR_train_000000.png", 0, "no", ("left",)),
                ("val", 0, 0, "CLEVR_val_000000.png", 0, "no", ("left",)),
            ],
        )
        with self.assertRaises(FrozenInstanceError):
            source.scenes[0].image_index = 7
        with self.assertRaises(TypeError):
            source.scenes[0].scene_data["objects"] = ()
        with self.assertRaises(TypeError):
            source.scenes[0].scene_data["objects"][0]["color"] = "blue"

    def test_missing_required_json_or_referenced_png_is_a_hard_failure(self) -> None:
        missing_json = self.root / "questions" / "CLEVR_val_questions.json"
        missing_json.unlink()
        with self.assertRaisesRegex(CLEVRDataError, "CLEVR_val_questions.json"):
            load_source_index(self.root)

        self.fixture.write_json()
        missing_png = self.root / "images" / "train" / "CLEVR_train_000000.png"
        missing_png.unlink()
        with self.assertRaisesRegex(CLEVRDataError, "CLEVR_train_000000.png"):
            load_source_index(self.root)

    def test_duplicate_scene_image_id_and_filename_are_rejected_separately(self) -> None:
        duplicate_id = _scene("train", 0)
        duplicate_id["image_filename"] = "CLEVR_train_000001.png"
        self.fixture.scenes["train"].append(duplicate_id)
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "duplicate.*image_index"):
            load_source_index(self.root)

        self.fixture.scenes["train"] = [_scene("train", 0), _scene("train", 1)]
        self.fixture.scenes["train"][1]["image_filename"] = (
            "CLEVR_train_000000.png"
        )
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "duplicate.*image_filename"):
            load_source_index(self.root)

    def test_duplicate_question_id_is_rejected(self) -> None:
        duplicate = _question("train", 0, 0)
        duplicate["question"] = "Conflicting source row?"
        self.fixture.questions["train"].append(duplicate)
        self.fixture.write_json()

        with self.assertRaisesRegex(CLEVRDataError, "duplicate.*question_index"):
            load_source_index(self.root)

    def test_scene_and_question_split_mismatches_are_rejected(self) -> None:
        self.fixture.scenes["train"][0]["split"] = "val"
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "scene.*split"):
            load_source_index(self.root)

        self.fixture.scenes["train"][0]["split"] = "train"
        self.fixture.questions["train"][0]["split"] = "val"
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "question.*split"):
            load_source_index(self.root)

    def test_question_requires_an_exact_scene_id_filename_join(self) -> None:
        second_scene = _scene("train", 1)
        self.fixture.scenes["train"].append(second_scene)
        Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(
            self.root / "images" / "train" / "CLEVR_train_000001.png"
        )
        self.fixture.questions["train"][0]["image_filename"] = (
            "CLEVR_train_000001.png"
        )
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "image_index.*image_filename"):
            load_source_index(self.root)

        self.fixture.questions["train"][0]["image_index"] = 9
        self.fixture.questions["train"][0]["image_filename"] = (
            "CLEVR_train_000009.png"
        )
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "no matching scene"):
            load_source_index(self.root)

    def test_unreadable_png_is_rejected(self) -> None:
        image_path = self.root / "images" / "val" / "CLEVR_val_000000.png"
        image_path.write_bytes(b"not a PNG")

        with self.assertRaisesRegex(CLEVRDataError, "unreadable PNG"):
            load_source_index(self.root)


class AuditManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = _AuditFixture(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_split_before_filter_manifest_and_train_only_vocabulary(self) -> None:
        audit = build_audit(load_source_index(self.root))

        self.assertEqual(
            [
                (
                    summary.partition,
                    summary.mother_image_count,
                    summary.selected_image_count,
                    summary.selected_question_count,
                )
                for summary in audit.partition_summaries
            ],
            [
                ("train", 9, 1, 8),
                ("dev", 1, 1, 1),
                ("base_test", 2, 1, 1),
                ("confirmation", 2, 1, 1),
            ],
        )
        image_three_partitions = {
            record.partition
            for record in audit.selected_questions
            if record.official_split == "train" and record.image_index == 3
        }
        self.assertEqual(image_three_partitions, {"train"})

        self.assertIn("trainword", audit.vocabulary.token_to_id)
        self.assertNotIn("devonly", audit.vocabulary.token_to_id)
        self.assertNotIn("baseonly", audit.vocabulary.token_to_id)
        dev_record = next(
            record for record in audit.selected_questions if record.partition == "dev"
        )
        self.assertEqual(dev_record.encoded_tokens[1], 3)

    def test_audits_answers_and_keeps_only_offline_manifest_fields(self) -> None:
        audit = build_audit(load_source_index(self.root))

        self.assertEqual(
            tuple(field.name for field in fields(SelectedQuestion)),
            (
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
            ),
        )
        first = audit.selected_questions[0]
        self.assertEqual(
            (
                first.partition,
                first.official_split,
                first.image_index,
                first.question_index,
                first.source_position,
                first.answer,
                first.relation,
            ),
            ("train", "train", 3, 0, 0, 0, "left"),
        )
        self.fixture.questions["train"][0]["answer"] = "yes"
        self.fixture.write_json()
        with self.assertRaisesRegex(CLEVRDataError, "official answer"):
            build_audit(load_source_index(self.root))

    def test_reports_exact_exclusions_all_cells_and_source_summaries(self) -> None:
        audit = build_audit(load_source_index(self.root))

        self.assertEqual(
            [
                (item.partition, item.reason, item.count)
                for item in audit.exclusion_counts
            ],
            [
                ("train", "unsupported_function", 1),
                ("dev", "relate_count", 1),
                ("base_test", "unsupported_function", 1),
            ],
        )
        self.assertEqual(len(audit.cell_counts), 32)
        train_cells = {
            (item.relation, item.answer): item.count
            for item in audit.cell_counts
            if item.partition == "train"
        }
        self.assertEqual(
            train_cells,
            {
                ("left", 0): 1,
                ("left", 1): 1,
                ("right", 0): 1,
                ("right", 1): 1,
                ("front", 0): 1,
                ("front", 1): 1,
                ("behind", 0): 1,
                ("behind", 1): 1,
            },
        )
        self.assertTrue(all(item.accuracy is None for item in audit.cell_counts))
        self.assertEqual(
            {
                (summary.official_split, summary.scene_count, summary.question_count)
                for summary in audit.source_summaries
            },
            {("train", 10, 11), ("val", 4, 3)},
        )
        self.assertEqual(
            {
                (summary.relative_path, summary.record_count)
                for summary in audit.file_summaries
            },
            {
                ("scenes/CLEVR_train_scenes.json", 10),
                ("questions/CLEVR_train_questions.json", 11),
                ("scenes/CLEVR_val_scenes.json", 4),
                ("questions/CLEVR_val_questions.json", 3),
            },
        )
        self.assertTrue(all(summary.size_bytes > 0 for summary in audit.file_summaries))


class AuditCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = _AuditFixture(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_empty_required_partition_is_a_hard_failure(self) -> None:
        self.fixture.questions["train"] = [
            question
            for question in self.fixture.questions["train"]
            if question["image_index"] != 6
        ]
        self.fixture.write_json()

        with self.assertRaisesRegex(CLEVRDataError, "required partition dev is empty"):
            build_audit(load_source_index(self.root))

    def test_missing_training_answer_class_is_a_hard_failure(self) -> None:
        self.fixture.questions["train"] = [
            question
            for question in self.fixture.questions["train"]
            if question["image_index"] != 3 or question["answer"] != "yes"
        ]
        self.fixture.write_json()

        with self.assertRaisesRegex(CLEVRDataError, "train.*answer class.*yes"):
            build_audit(load_source_index(self.root))

    def test_missing_training_relation_is_a_hard_failure(self) -> None:
        retained: list[dict[str, object]] = []
        for question in self.fixture.questions["train"]:
            if question["image_index"] != 3:
                retained.append(question)
                continue
            program = question["program"]
            relation_nodes = [
                node for node in program if node["function"] == "relate"
            ]
            if not relation_nodes or relation_nodes[0]["value_inputs"] != ["behind"]:
                retained.append(question)
        self.fixture.questions["train"] = retained
        self.fixture.write_json()

        with self.assertRaisesRegex(CLEVRDataError, "train.*relation.*behind"):
            build_audit(load_source_index(self.root))

    def test_empty_evaluation_cell_is_retained_as_zero_with_na_accuracy(self) -> None:
        audit = build_audit(load_source_index(self.root))

        empty_cell = next(
            item
            for item in audit.cell_counts
            if (item.partition, item.relation, item.answer)
            == ("base_test", "left", 0)
        )
        self.assertEqual(empty_cell.count, 0)
        self.assertIsNone(empty_cell.accuracy)


class AuditCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self.temporary_directory.name)
        self.data_root = self.temporary_root / "CLEVR_v1.0"
        self.fixture = _AuditFixture(self.data_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_requires_explicit_data_root_and_rejects_download_option(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as missing:
            audit_main(["--output-dir", str(self.temporary_root / "output")])
        self.assertEqual(missing.exception.code, 2)
        self.assertIn("--data-root", stderr.getvalue())

        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as download:
            audit_main(
                [
                    "--data-root",
                    str(self.data_root),
                    "--output-dir",
                    str(self.temporary_root / "output"),
                    "--download",
                ]
            )
        self.assertEqual(download.exception.code, 2)
        self.assertIn("unrecognized arguments: --download", stderr.getvalue())

    def test_refuses_to_overwrite_nonempty_output_directory(self) -> None:
        output_dir = self.temporary_root / "output"
        output_dir.mkdir()
        sentinel = output_dir / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(CLEVRDataError, "nonempty output directory"):
            audit_main(
                ["--data-root", str(self.data_root), "--output-dir", str(output_dir)]
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_refuses_output_directory_symlink_without_touching_target(self) -> None:
        target = self.temporary_root / "target"
        target.mkdir()
        output_link = self.temporary_root / "output-link"
        output_link.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(CLEVRDataError, "must not be a symlink"):
            audit_main(
                [
                    "--data-root",
                    str(self.data_root),
                    "--output-dir",
                    str(output_link),
                ]
            )

        self.assertTrue(output_link.is_symlink())
        self.assertEqual(list(target.iterdir()), [])

    def test_success_atomically_writes_the_four_complete_artifacts(self) -> None:
        output_dir = self.temporary_root / "output"
        output_dir.mkdir()

        exit_code = audit_main(
            ["--data-root", str(self.data_root), "--output-dir", str(output_dir)]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {
                "image_partitions.json",
                "vocabulary.json",
                "selected_questions.jsonl",
                "audit_report.json",
            },
        )
        partitions = json.loads(
            (output_dir / "image_partitions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {name: len(rows) for name, rows in partitions["partitions"].items()},
            {"train": 9, "dev": 1, "base_test": 2, "confirmation": 2},
        )
        vocabulary = json.loads(
            (output_dir / "vocabulary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(vocabulary["token_to_id"]["<NULL>"], 0)
        self.assertEqual(vocabulary["id_to_token"][:4], [
            "<NULL>",
            "<START>",
            "<END>",
            "<UNK>",
        ])
        selected_rows = [
            json.loads(line)
            for line in (output_dir / "selected_questions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(selected_rows), 11)
        self.assertEqual(
            set(selected_rows[0]),
            {
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
            },
        )
        self.assertEqual(
            selected_rows[0]["image_path"], "images/train/CLEVR_train_000003.png"
        )
        report = json.loads(
            (output_dir / "audit_report.json").read_text(encoding="utf-8")
        )
        empty_cell = next(
            item
            for item in report["cell_counts"]
            if (item["partition"], item["relation"], item["answer"])
            == ("base_test", "left", 0)
        )
        self.assertEqual(empty_cell["count"], 0)
        self.assertEqual(empty_cell["accuracy"], "NA")
        serialized_keys = {
            key.lower()
            for artifact in output_dir.iterdir()
            for key in _json_keys(artifact)
        }
        self.assertTrue(serialized_keys.isdisjoint({"sha", "sha256", "md5", "digest"}))

    def test_failed_audit_leaves_no_final_or_temporary_artifacts(self) -> None:
        self.fixture.questions["train"][0]["answer"] = "yes"
        self.fixture.write_json()
        output_dir = self.temporary_root / "failed-output"

        with self.assertRaisesRegex(CLEVRDataError, "official answer"):
            audit_main(
                ["--data-root", str(self.data_root), "--output-dir", str(output_dir)]
            )

        self.assertFalse(output_dir.exists())
        self.assertEqual(
            list(self.temporary_root.glob(f".{output_dir.name}.tmp-*")), []
        )


def _json_keys(path: Path) -> set[str]:
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        values = [json.loads(path.read_text())]
    keys: set[str] = set()
    stack = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            keys.update(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return keys


if __name__ == "__main__":
    unittest.main()
