"""Runtime CLEVR dataset contract tests using temporary local artifacts."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from src.clevr_dataset import (
    CLEVRDatasetError,
    CLEVRQuestionDataset,
    CLEVRQuestionOnlyDataset,
    collate_clevr_questions,
)


PARTITIONS = ("train", "dev", "base_test", "confirmation")


class RuntimeDatasetFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "CLEVR_v1.0"
        self.audit_dir = self.root / "audit"
        self.audit_dir.mkdir(parents=True)
        self.rows = [
            self._row("train", "train", 2, 20, 20, "There?", [1, 5, 2], 1, "left"),
            self._row("confirmation", "val", 1, 1, 1, "There?", [1, 5, 2], 1, "behind"),
            self._row("train", "train", 1, 10, 10, "Are?", [1, 4, 2], 0, "right"),
            self._row("base_test", "val", 0, 0, 0, "Are?", [1, 4, 2], 0, "front"),
            self._row("dev", "train", 3, 30, 30, "There?", [1, 5, 2], 1, "left"),
        ]
        self.colors = {
            ("train", 1): (255, 128, 0, 64),
            ("train", 2): (0, 255, 0, 127),
            ("train", 3): (0, 0, 255, 255),
            ("val", 0): (255, 0, 0, 255),
            ("val", 1): (0, 255, 255, 255),
        }
        self._write_images()
        self.write_artifacts()

    @staticmethod
    def _row(
        partition: str,
        official_split: str,
        image_index: int,
        question_index: int,
        source_position: int,
        question: str,
        encoded_tokens: list[int],
        answer: int,
        relation: str,
    ) -> dict[str, object]:
        image_filename = f"CLEVR_{official_split}_{image_index:06d}.png"
        return {
            "partition": partition,
            "official_split": official_split,
            "image_index": image_index,
            "image_filename": image_filename,
            "image_path": f"images/{official_split}/{image_filename}",
            "question_index": question_index,
            "source_position": source_position,
            "question": question,
            "encoded_tokens": encoded_tokens,
            "answer": answer,
            "relation": relation,
        }

    def _write_images(self) -> None:
        for (split, image_index), color in self.colors.items():
            directory = self.data_root / "images" / split
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"CLEVR_{split}_{image_index:06d}.png"
            Image.new("RGBA", (3, 2), color).save(path, format="PNG")

    def write_artifacts(self) -> None:
        vocabulary = {
            "protocol_version": "PUB-R1.1",
            "token_to_id": {
                "<NULL>": 0,
                "<START>": 1,
                "<END>": 2,
                "<UNK>": 3,
                "Are": 4,
                "There": 5,
            },
            "id_to_token": ["<NULL>", "<START>", "<END>", "<UNK>", "Are", "There"],
        }
        counts = {partition: 0 for partition in PARTITIONS}
        for row in self.rows:
            counts[str(row["partition"])] += 1
        report = {
            "protocol_version": "PUB-R1.1",
            "partition_summaries": [
                {
                    "partition": partition,
                    "mother_image_count": counts[partition],
                    "selected_image_count": counts[partition],
                    "selected_question_count": counts[partition],
                }
                for partition in PARTITIONS
            ],
        }
        (self.audit_dir / "vocabulary.json").write_text(
            json.dumps(vocabulary), encoding="utf-8"
        )
        (self.audit_dir / "audit_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        with (self.audit_dir / "selected_questions.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for row in self.rows:
                output.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self.temporary.cleanup()


class RuntimeDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RuntimeDatasetFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def assert_rejected(
        self,
        fixture: RuntimeDatasetFixture,
        pattern: str,
        *,
        partition: str = "train",
        data_root: Path | None = None,
        audit_dir: Path | None = None,
    ) -> None:
        try:
            CLEVRQuestionDataset(
                data_root=fixture.data_root if data_root is None else data_root,
                audit_dir=fixture.audit_dir if audit_dir is None else audit_dir,
                partition=partition,
            )
        except Exception as error:
            self.assertIsInstance(error, CLEVRDatasetError)
            self.assertRegex(str(error), pattern)
        else:
            self.fail("invalid runtime artifacts were accepted")

    def assert_access_rejected(
        self, dataset: CLEVRQuestionDataset, index: int, pattern: str
    ) -> None:
        try:
            dataset[index]
        except Exception as error:
            self.assertIsInstance(error, CLEVRDatasetError)
            self.assertRegex(str(error), pattern)
        else:
            self.fail("invalid image access was accepted")

    @staticmethod
    def rewrite_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def read_json(path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_sorts_selected_partition_and_returns_only_model_tensors(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )

        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(
            set(sample), {"image", "question_tokens", "question_length", "answer"}
        )
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(sample["image"]).all())
        self.assertGreaterEqual(sample["image"].min().item(), 0.0)
        self.assertLessEqual(sample["image"].max().item(), 1.0)
        torch.testing.assert_close(
            sample["image"][:, 0, 0],
            torch.tensor([1.0, 128.0 / 255.0, 0.0]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sample["question_tokens"], torch.tensor([1, 4, 2]), rtol=0, atol=0
        )
        self.assertEqual(sample["question_tokens"].dtype, torch.int64)
        self.assertEqual(sample["question_length"].dtype, torch.int64)
        self.assertEqual(tuple(sample["question_length"].shape), ())
        self.assertEqual(sample["question_length"].item(), 3)
        self.assertEqual(sample["answer"].dtype, torch.int64)
        self.assertEqual(tuple(sample["answer"].shape), ())
        self.assertEqual(sample["answer"].item(), 0)

    def test_rgb_transform_matches_the_frozen_torchvision_reference(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )
        image_path = (
            self.fixture.data_root / "images" / "train" / "CLEVR_train_000001.png"
        )
        with Image.open(image_path, formats=("PNG",)) as source:
            expected = tvf.pil_to_tensor(source.convert("RGB")).to(torch.float32) / 255.0
        expected = tvf.resize(
            expected,
            [224, 224],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        torch.testing.assert_close(dataset[0]["image"], expected, rtol=0, atol=0)

    def test_native_size_resize_accepts_roundoff_without_clamping(self) -> None:
        first_path = (
            self.fixture.data_root
            / "images"
            / "train"
            / "CLEVR_train_000001.png"
        )
        second_path = (
            self.fixture.data_root
            / "images"
            / "train"
            / "CLEVR_train_000002.png"
        )
        Image.new("RGB", (480, 320), (255, 255, 255)).save(
            first_path, format="PNG"
        )
        x = torch.arange(480, dtype=torch.int64).remainder(256).to(torch.uint8)
        y = torch.arange(320, dtype=torch.int64).remainder(256).to(torch.uint8)
        nonconstant = torch.stack(
            (
                x.unsqueeze(0).expand(320, -1),
                y.unsqueeze(1).expand(-1, 480),
                (
                    x.to(torch.int64).unsqueeze(0)
                    + y.to(torch.int64).unsqueeze(1)
                )
                .remainder(256)
                .to(torch.uint8),
            )
        )
        tvf.to_pil_image(nonconstant).save(second_path, format="PNG")
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )

        for index, path in enumerate((first_path, second_path)):
            with self.subTest(path=path.name):
                with Image.open(path, formats=("PNG",)) as source:
                    expected = tvf.pil_to_tensor(source.convert("RGB")).to(
                        torch.float32
                    ) / 255.0
                expected = tvf.resize(
                    expected,
                    [224, 224],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )

                torch.testing.assert_close(
                    dataset[index]["image"], expected, rtol=0, atol=0
                )

    def test_requires_explicit_roots_partition_and_all_three_artifacts(self) -> None:
        cases = ("vocabulary.json", "audit_report.json", "selected_questions.jsonl")
        for filename in cases:
            with self.subTest(missing=filename):
                fixture = RuntimeDatasetFixture()
                try:
                    (fixture.audit_dir / filename).unlink()
                    self.assert_rejected(fixture, r"missing|required|artifact")
                finally:
                    fixture.close()

        fixture = RuntimeDatasetFixture()
        try:
            self.assert_rejected(
                fixture, r"data root|directory", data_root=fixture.root / "missing-data"
            )
            self.assert_rejected(
                fixture, r"audit.*directory", audit_dir=fixture.root / "missing-audit"
            )
            self.assert_rejected(fixture, r"partition", partition="unknown")
            fixture.rows = [row for row in fixture.rows if row["partition"] != "dev"]
            fixture.write_artifacts()
            self.assert_rejected(fixture, r"dev.*empty|empty.*dev", partition="dev")
        finally:
            fixture.close()

    def test_rejects_protocol_vocabulary_and_report_count_disagreement(self) -> None:
        def run_case(name: str, mutate, pattern: str) -> None:
            with self.subTest(case=name):
                fixture = RuntimeDatasetFixture()
                try:
                    mutate(fixture)
                    self.assert_rejected(fixture, pattern)
                finally:
                    fixture.close()

        def wrong_vocabulary_protocol(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "vocabulary.json"
            payload = self.read_json(path)
            payload["protocol_version"] = "PUB-R0"
            self.rewrite_json(path, payload)

        def wrong_report_protocol(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "audit_report.json"
            payload = self.read_json(path)
            payload["protocol_version"] = "PUB-R0"
            self.rewrite_json(path, payload)

        def invalid_vocabulary_inverse(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "vocabulary.json"
            payload = self.read_json(path)
            payload["token_to_id"]["Are"] = 5
            self.rewrite_json(path, payload)

        def missing_summary(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "audit_report.json"
            payload = self.read_json(path)
            payload["partition_summaries"] = payload["partition_summaries"][:-1]
            self.rewrite_json(path, payload)

        def duplicate_summary(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "audit_report.json"
            payload = self.read_json(path)
            payload["partition_summaries"].append(payload["partition_summaries"][0])
            self.rewrite_json(path, payload)

        def unknown_summary(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "audit_report.json"
            payload = self.read_json(path)
            payload["partition_summaries"][0]["partition"] = "other"
            self.rewrite_json(path, payload)

        def count_mismatch(fixture: RuntimeDatasetFixture) -> None:
            path = fixture.audit_dir / "audit_report.json"
            payload = self.read_json(path)
            payload["partition_summaries"][0]["selected_question_count"] = 99
            self.rewrite_json(path, payload)

        run_case("vocabulary protocol", wrong_vocabulary_protocol, r"protocol")
        run_case("report protocol", wrong_report_protocol, r"protocol")
        run_case("vocabulary inverse", invalid_vocabulary_inverse, r"vocabulary|token")
        run_case("missing summary", missing_summary, r"summary|partition")
        run_case("duplicate summary", duplicate_summary, r"duplicate.*summary|summary.*duplicate")
        run_case("unknown summary", unknown_summary, r"summary|partition")
        run_case("count mismatch", count_mismatch, r"count|train")

    def test_rejects_missing_extra_or_wrong_typed_manifest_fields(self) -> None:
        cases = (
            ("missing", lambda row: row.pop("relation")),
            ("extra", lambda row: row.update({"unexpected": 1})),
            ("boolean image", lambda row: row.update({"image_index": True})),
            ("boolean question", lambda row: row.update({"question_index": False})),
            ("boolean source", lambda row: row.update({"source_position": True})),
            ("boolean answer", lambda row: row.update({"answer": False})),
        )
        for name, mutate in cases:
            with self.subTest(case=name):
                fixture = RuntimeDatasetFixture()
                try:
                    mutate(fixture.rows[0])
                    fixture.write_artifacts()
                    self.assert_rejected(fixture, r"field|integer|answer|keys|line")
                finally:
                    fixture.close()

    def test_rejects_invalid_split_identity_tokens_answer_and_relation(self) -> None:
        cases = (
            ("split", lambda row: row.update({"official_split": "val"}), r"split"),
            ("filename", lambda row: row.update({"image_filename": "bad.png"}), r"filename|identity"),
            ("negative", lambda row: row.update({"question_index": -1}), r"question_index|non-negative"),
            ("question", lambda row: row.update({"question": 3}), r"question"),
            ("relation", lambda row: row.update({"relation": "over"}), r"relation"),
            ("answer", lambda row: row.update({"answer": 2}), r"answer"),
            ("tokens type", lambda row: row.update({"encoded_tokens": "1,5,2"}), r"token"),
            ("token bool", lambda row: row.update({"encoded_tokens": [1, True, 2]}), r"token"),
            ("token range", lambda row: row.update({"encoded_tokens": [1, 99, 2]}), r"token"),
            ("token null", lambda row: row.update({"encoded_tokens": [1, 0, 2]}), r"NULL|token"),
            ("start", lambda row: row.update({"encoded_tokens": [3, 5, 2]}), r"START|token"),
            ("end", lambda row: row.update({"encoded_tokens": [1, 5, 3]}), r"END|token"),
            ("recompute", lambda row: row.update({"encoded_tokens": [1, 4, 2]}), r"encoded|question|token"),
        )
        for name, mutate, pattern in cases:
            with self.subTest(case=name):
                fixture = RuntimeDatasetFixture()
                try:
                    mutate(fixture.rows[0])
                    fixture.write_artifacts()
                    self.assert_rejected(fixture, pattern)
                finally:
                    fixture.close()

    def test_rejects_duplicate_question_or_source_identity(self) -> None:
        for field in ("question_index", "source_position"):
            with self.subTest(field=field):
                fixture = RuntimeDatasetFixture()
                try:
                    duplicate = dict(fixture.rows[2])
                    duplicate["question_index"] = 999
                    duplicate["source_position"] = 999
                    duplicate[field] = fixture.rows[0][field]
                    fixture.rows.append(duplicate)
                    fixture.write_artifacts()
                    self.assert_rejected(fixture, rf"duplicate.*{field}|{field}.*duplicate")
                finally:
                    fixture.close()

    def test_rejects_one_image_assigned_to_multiple_partitions(self) -> None:
        dev_row = next(
            row for row in self.fixture.rows if row["partition"] == "dev"
        )
        dev_row.update(
            {
                "image_index": 1,
                "image_filename": "CLEVR_train_000001.png",
                "image_path": "images/train/CLEVR_train_000001.png",
            }
        )
        self.fixture.write_artifacts()

        self.assert_rejected(self.fixture, r"image|partition|multiple|train|dev")

    def test_validates_malformed_rows_even_outside_the_requested_partition(self) -> None:
        fixture = RuntimeDatasetFixture()
        try:
            dev_row = next(row for row in fixture.rows if row["partition"] == "dev")
            dev_row["relation"] = "over"
            fixture.write_artifacts()
            self.assert_rejected(fixture, r"relation|line", partition="train")
        finally:
            fixture.close()

        for payload, pattern in (("\n", r"blank|line"), ("{bad json}\n", r"JSON|line")):
            with self.subTest(payload=payload):
                fixture = RuntimeDatasetFixture()
                try:
                    with (fixture.audit_dir / "selected_questions.jsonl").open(
                        "a", encoding="utf-8"
                    ) as output:
                        output.write(payload)
                    self.assert_rejected(fixture, pattern, partition="train")
                finally:
                    fixture.close()

    def test_rejects_absolute_traversing_noncanonical_missing_and_symlink_escape_paths(
        self,
    ) -> None:
        path_cases = (
            ("absolute", "/tmp/outside.png", r"image_path|relative|canonical"),
            (
                "traversal",
                "images/train/../train/CLEVR_train_000002.png",
                r"image_path|canonical",
            ),
            (
                "noncanonical",
                "images/train/sub/../CLEVR_train_000002.png",
                r"image_path|canonical",
            ),
        )
        for name, image_path, pattern in path_cases:
            with self.subTest(case=name):
                fixture = RuntimeDatasetFixture()
                try:
                    fixture.rows[0]["image_path"] = image_path
                    fixture.write_artifacts()
                    self.assert_rejected(fixture, pattern)
                finally:
                    fixture.close()

        fixture = RuntimeDatasetFixture()
        try:
            missing = (
                fixture.data_root
                / "images"
                / "train"
                / "CLEVR_train_000002.png"
            )
            missing.unlink()
            self.assert_rejected(fixture, r"image|missing|file")
        finally:
            fixture.close()

        fixture = RuntimeDatasetFixture()
        try:
            linked = (
                fixture.data_root
                / "images"
                / "train"
                / "CLEVR_train_000002.png"
            )
            outside = fixture.root / "outside.png"
            Image.new("RGBA", (3, 2), (1, 2, 3, 255)).save(outside, format="PNG")
            linked.unlink()
            linked.symlink_to(outside)
            self.assert_rejected(fixture, r"outside|root|escape")
        finally:
            fixture.close()

    def test_decode_failure_after_construction_is_contextual_and_not_skipped(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )
        first_path = (
            self.fixture.data_root
            / "images"
            / "train"
            / "CLEVR_train_000001.png"
        )
        first_path.write_bytes(b"not a PNG")

        self.assert_access_rejected(
            dataset, 0, r"index 0|CLEVR_train_000001.png|decode"
        )

    def test_rejects_image_symlink_escape_introduced_after_construction(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )
        image_path = (
            self.fixture.data_root
            / "images"
            / "train"
            / "CLEVR_train_000001.png"
        )
        outside = self.fixture.root / "outside.png"
        Image.new("RGB", (3, 2), (9, 8, 7)).save(outside, format="PNG")
        image_path.unlink()
        image_path.symlink_to(outside)

        self.assert_access_rejected(dataset, 0, r"index 0|root|escape")

    def test_rejects_parent_symlink_escape_introduced_after_construction(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )
        train_directory = self.fixture.data_root / "images" / "train"
        retained_directory = self.fixture.root / "retained-train"
        outside_directory = self.fixture.root / "outside-train"
        train_directory.rename(retained_directory)
        outside_directory.mkdir()
        Image.new("RGB", (3, 2), (9, 8, 7)).save(
            outside_directory / "CLEVR_train_000001.png", format="PNG"
        )
        train_directory.symlink_to(outside_directory, target_is_directory=True)

        self.assert_access_rejected(dataset, 0, r"index 0|root|escape")

    def test_multiple_questions_may_share_one_image_without_mutable_sample_storage(
        self,
    ) -> None:
        duplicate = dict(self.fixture.rows[2])
        duplicate["question_index"] = 11
        duplicate["source_position"] = 11
        self.fixture.rows.append(duplicate)
        self.fixture.write_artifacts()
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )

        first = dataset[0]["image"]
        first.zero_()
        second = dataset[1]["image"]

        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(
            second[:, 0, 0],
            torch.tensor([1.0, 128.0 / 255.0, 0.0]),
            rtol=0,
            atol=0,
        )

    def test_dataset_survives_pickle_round_trip(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )

        restored = pickle.loads(pickle.dumps(dataset))

        self.assertEqual(len(restored), 2)
        torch.testing.assert_close(
            restored[0]["question_tokens"],
            torch.tensor([1, 4, 2], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_spawn_dataloader_reads_dataset_in_worker_process(self) -> None:
        dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=1,
            multiprocessing_context="spawn",
            collate_fn=collate_clevr_questions,
        )

        batch = next(iter(loader))

        torch.testing.assert_close(
            batch["answer"],
            torch.tensor([0, 1], dtype=torch.int64),
            rtol=0,
            atol=0,
        )


class CLEVRQuestionOnlyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RuntimeDatasetFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def assert_rejected(
        self,
        pattern: str,
        *,
        audit_dir: Path | None = None,
        partition: str = "train",
    ) -> None:
        try:
            CLEVRQuestionOnlyDataset(
                audit_dir=(
                    self.fixture.audit_dir if audit_dir is None else audit_dir
                ),
                partition=partition,
            )
        except Exception as error:
            self.assertIsInstance(error, CLEVRDatasetError)
            self.assertRegex(str(error), pattern)
        else:
            self.fail("invalid question-only runtime artifacts were accepted")

    def test_needs_no_data_root_or_images_and_never_opens_png(self) -> None:
        shutil.rmtree(self.fixture.data_root)

        with mock.patch(
            "src.clevr_dataset.Image.open",
            side_effect=AssertionError("Q-only must not open a PNG"),
        ) as image_open:
            dataset = CLEVRQuestionOnlyDataset(
                audit_dir=self.fixture.audit_dir,
                partition="train",
            )
            sample = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            set(sample), {"question_tokens", "question_length", "answer"}
        )
        torch.testing.assert_close(
            sample["question_tokens"],
            torch.tensor([1, 4, 2], dtype=torch.int64),
            rtol=0,
            atol=0,
        )
        self.assertEqual(sample["question_tokens"].dtype, torch.int64)
        self.assertEqual(sample["question_length"].dtype, torch.int64)
        self.assertEqual(tuple(sample["question_length"].shape), ())
        self.assertEqual(sample["question_length"].item(), 3)
        self.assertEqual(sample["answer"].dtype, torch.int64)
        self.assertEqual(tuple(sample["answer"].shape), ())
        self.assertEqual(sample["answer"].item(), 0)
        image_open.assert_not_called()

    def test_matches_rgb_question_projection_and_order(self) -> None:
        rgb_dataset = CLEVRQuestionDataset(
            data_root=self.fixture.data_root,
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )
        question_dataset = CLEVRQuestionOnlyDataset(
            audit_dir=self.fixture.audit_dir,
            partition="train",
        )

        self.assertEqual(len(question_dataset), len(rgb_dataset))
        for index in range(len(rgb_dataset)):
            rgb_sample = rgb_dataset[index]
            question_sample = question_dataset[index]
            for field in ("question_tokens", "question_length", "answer"):
                with self.subTest(index=index, field=field):
                    self.assertTrue(
                        torch.equal(question_sample[field], rgb_sample[field])
                    )

    def test_rejects_noncanonical_image_metadata_without_resolving_files(
        self,
    ) -> None:
        self.fixture.rows[0]["image_path"] = (
            "images/train/../train/CLEVR_train_000002.png"
        )
        self.fixture.write_artifacts()
        shutil.rmtree(self.fixture.data_root)

        self.assert_rejected(r"image_path|canonical")

    def test_rejects_invalid_constructor_inputs_and_unrequested_bad_rows(
        self,
    ) -> None:
        self.assert_rejected(r"partition", partition="unknown")
        self.assert_rejected(
            r"audit.*directory",
            audit_dir=self.fixture.root / "missing-audit",
        )

        dev_row = next(
            row for row in self.fixture.rows if row["partition"] == "dev"
        )
        dev_row["image_path"] = "images/train/not-the-image-identity.png"
        self.fixture.write_artifacts()
        shutil.rmtree(self.fixture.data_root)

        self.assert_rejected(r"image_path|canonical", partition="train")


class CLEVRQuestionCollateTests(unittest.TestCase):
    @staticmethod
    def sample(
        tokens: list[int], *, image_value: float, answer: int
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.full((3, 224, 224), image_value, dtype=torch.float32),
            "question_tokens": torch.tensor(tokens, dtype=torch.int64),
            "question_length": torch.tensor(len(tokens), dtype=torch.int64),
            "answer": torch.tensor(answer, dtype=torch.int64),
        }

    def assert_collate_rejected(
        self, samples: object, pattern: str = r"sample|collate|batch|tensor|key"
    ) -> None:
        try:
            collate_clevr_questions(samples)  # type: ignore[arg-type]
        except Exception as error:
            self.assertIsInstance(error, CLEVRDatasetError)
            self.assertRegex(str(error), pattern)
        else:
            self.fail("invalid CLEVR samples were collated")

    def test_pads_with_null_without_truncation_and_preserves_sample_order(
        self,
    ) -> None:
        samples = [
            self.sample([1, 7, 2], image_value=0.25, answer=1),
            self.sample([1, 5, 6, 2], image_value=0.75, answer=0),
        ]

        batch = collate_clevr_questions(samples)

        self.assertEqual(
            set(batch), {"image", "question_tokens", "question_length", "answer"}
        )
        torch.testing.assert_close(
            batch["question_tokens"],
            torch.tensor([[1, 7, 2, 0], [1, 5, 6, 2]], dtype=torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batch["question_length"],
            torch.tensor([3, 4], dtype=torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batch["answer"], torch.tensor([1, 0], dtype=torch.int64), rtol=0, atol=0
        )
        torch.testing.assert_close(
            batch["image"][:, 0, 0, 0],
            torch.tensor([0.25, 0.75], dtype=torch.float32),
            rtol=0,
            atol=0,
        )

    def test_dataloader_batch_contains_exactly_the_four_model_fields(self) -> None:
        fixture = RuntimeDatasetFixture()
        try:
            dataset = CLEVRQuestionDataset(
                data_root=fixture.data_root,
                audit_dir=fixture.audit_dir,
                partition="train",
            )
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_clevr_questions,
            )

            batch = next(iter(loader))

            self.assertEqual(
                set(batch),
                {"image", "question_tokens", "question_length", "answer"},
            )
            self.assertEqual(tuple(batch["image"].shape), (2, 3, 224, 224))
            self.assertEqual(tuple(batch["question_tokens"].shape), (2, 3))
            torch.testing.assert_close(
                batch["answer"],
                torch.tensor([0, 1], dtype=torch.int64),
                rtol=0,
                atol=0,
            )
        finally:
            fixture.close()

    def test_accepts_two_ulp_resize_roundoff_at_the_unit_interval_boundary(
        self,
    ) -> None:
        one = torch.tensor(1.0, dtype=torch.float32)
        positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
        first_successor = torch.nextafter(one, positive_infinity)
        resize_roundoff = torch.nextafter(first_successor, positive_infinity).item()
        sample = self.sample([1, 7, 2], image_value=resize_roundoff, answer=1)

        batch = collate_clevr_questions([sample])

        self.assertEqual(batch["image"][0, 0, 0, 0].item(), resize_roundoff)

    def test_rejects_empty_batches_and_missing_or_extra_keys(self) -> None:
        self.assert_collate_rejected([], r"empty|sample")

        missing = self.sample([1, 7, 2], image_value=0.25, answer=1)
        missing.pop("answer")
        self.assert_collate_rejected([missing], r"key|field|sample 0")

        extra = self.sample([1, 7, 2], image_value=0.25, answer=1)
        extra["image_path"] = torch.tensor(0)
        self.assert_collate_rejected([extra], r"key|field|sample 0")

        non_tensor = self.sample([1, 7, 2], image_value=0.25, answer=1)
        non_tensor["answer"] = 1  # type: ignore[assignment]
        self.assert_collate_rejected([non_tensor], r"answer|tensor|sample 0")

    def test_rejects_wrong_sample_dtypes_shapes_lengths_and_answers(self) -> None:
        def run_case(name: str, mutate, pattern: str) -> None:
            with self.subTest(case=name):
                sample = self.sample([1, 7, 2], image_value=0.25, answer=1)
                mutate(sample)
                self.assert_collate_rejected([sample], pattern)

        run_case(
            "image dtype",
            lambda sample: sample.update({"image": sample["image"].to(torch.float64)}),
            r"image|dtype|sample 0",
        )
        run_case(
            "image shape",
            lambda sample: sample.update({"image": sample["image"][:, :223]}),
            r"image|shape|sample 0",
        )
        run_case(
            "image nonfinite",
            lambda sample: sample["image"].view(-1).__setitem__(0, float("nan")),
            r"image|finite|sample 0",
        )
        run_case(
            "image below range",
            lambda sample: sample["image"].view(-1).__setitem__(0, -0.01),
            r"image|range|sample 0|\[0,1\]",
        )
        run_case(
            "image above range",
            lambda sample: sample["image"].view(-1).__setitem__(0, 1.01),
            r"image|range|sample 0|\[0,1\]",
        )
        run_case(
            "token dtype",
            lambda sample: sample.update(
                {"question_tokens": sample["question_tokens"].to(torch.int32)}
            ),
            r"token|dtype|sample 0",
        )
        run_case(
            "token rank",
            lambda sample: sample.update(
                {"question_tokens": sample["question_tokens"].unsqueeze(0)}
            ),
            r"token|shape|rank|sample 0",
        )
        run_case(
            "empty tokens",
            lambda sample: sample.update(
                {"question_tokens": torch.empty(0, dtype=torch.int64)}
            ),
            r"token|empty|START|END|sample 0",
        )
        run_case(
            "length dtype",
            lambda sample: sample.update(
                {"question_length": sample["question_length"].to(torch.int32)}
            ),
            r"length|dtype|sample 0",
        )
        run_case(
            "length shape",
            lambda sample: sample.update(
                {"question_length": sample["question_length"].unsqueeze(0)}
            ),
            r"length|shape|scalar|sample 0",
        )
        run_case(
            "length mismatch",
            lambda sample: sample.update(
                {"question_length": torch.tensor(2, dtype=torch.int64)}
            ),
            r"length|token|sample 0",
        )
        run_case(
            "answer dtype",
            lambda sample: sample.update(
                {"answer": sample["answer"].to(torch.int32)}
            ),
            r"answer|dtype|sample 0",
        )
        run_case(
            "answer shape",
            lambda sample: sample.update({"answer": sample["answer"].unsqueeze(0)}),
            r"answer|shape|scalar|sample 0",
        )
        run_case(
            "answer value",
            lambda sample: sample.update(
                {"answer": torch.tensor(2, dtype=torch.int64)}
            ),
            r"answer|0 or 1|sample 0",
        )

    def test_rejects_tokens_without_exact_start_end_or_with_null_before_padding(
        self,
    ) -> None:
        for name, tokens, pattern in (
            ("wrong start", [3, 7, 2], r"START|token|sample 0"),
            ("wrong end", [1, 7, 3], r"END|token|sample 0"),
            ("embedded null", [1, 0, 2], r"NULL|token|sample 0"),
            ("negative token", [1, -1, 2], r"negative|token|sample 0"),
        ):
            with self.subTest(case=name):
                self.assert_collate_rejected(
                    [self.sample(tokens, image_value=0.25, answer=1)], pattern
                )


if __name__ == "__main__":
    unittest.main()
