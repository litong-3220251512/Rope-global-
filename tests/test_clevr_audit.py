from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from PIL import Image

from src.clevr_audit import CLEVRDataError, load_source_index


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


if __name__ == "__main__":
    unittest.main()
