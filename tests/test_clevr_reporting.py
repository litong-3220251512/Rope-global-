from __future__ import annotations

import copy
import errno
import importlib.metadata
import json
import math
import platform
import stat
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

import src.clevr_reporting as clevr_reporting
import run_clevr_data_audit as audit_producer
from run_clevr_statistics import main as statistics_cli_main
from src.clevr_audit import AuditResult, PartitionSummary, SelectedQuestion
from src.clevr_partitions import ImageIdentity, build_image_partitions
from src.clevr_reporting import (
    ArtifactEvidence,
    CLEVRReportError,
    EnvironmentEvidence,
    PartitionCounts,
    PredictionEvidence,
    PredictionSource,
    StatisticsBundle,
    StatisticsReport,
    build_statistics_bundle,
    load_audit_targets,
    load_prediction_run,
    publish_statistics_bundle,
    statistics_report_payload,
)
from src.clevr_statistics import (
    PredictionRow,
    RunPredictions,
    TargetRow,
    compute_clevr_statistics,
)
from src.clevr_text import build_vocabulary


_IMAGE_PARTITIONS = {
    "protocol_version": "PUB-R1.1",
    "partitions": {
        "train": [
            {
                "official_split": "train",
                "image_index": 2,
                "image_filename": "CLEVR_train_000002.png",
            },
            {
                "official_split": "train",
                "image_index": 3,
                "image_filename": "CLEVR_train_000003.png",
            },
        ],
        "dev": [
            {
                "official_split": "train",
                "image_index": 12,
                "image_filename": "CLEVR_train_000012.png",
            }
        ],
        "base_test": [
            {
                "official_split": "val",
                "image_index": 11,
                "image_filename": "CLEVR_val_000011.png",
            }
        ],
        "confirmation": [
            {
                "official_split": "val",
                "image_index": 13,
                "image_filename": "CLEVR_val_000013.png",
            }
        ],
    },
}

_MANIFEST_ROWS = (
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 50,
        "source_position": 2,
        "question": "Is anything behind the object?",
        "encoded_tokens": [1, 9, 2],
        "answer": 1,
        "relation": "behind",
    },
    {
        "partition": "confirmation",
        "official_split": "val",
        "image_index": 13,
        "image_filename": "CLEVR_val_000013.png",
        "image_path": "images/val/CLEVR_val_000013.png",
        "question_index": 60,
        "source_position": 3,
        "question": "Is anything right of the object?",
        "encoded_tokens": [1, 8, 2],
        "answer": 0,
        "relation": "right",
    },
    {
        "partition": "train",
        "official_split": "train",
        "image_index": 2,
        "image_filename": "CLEVR_train_000002.png",
        "image_path": "images/train/CLEVR_train_000002.png",
        "question_index": 20,
        "source_position": 1,
        "question": "Is anything right of the object?",
        "encoded_tokens": [1, 6, 2],
        "answer": 1,
        "relation": "right",
    },
    {
        "partition": "dev",
        "official_split": "train",
        "image_index": 12,
        "image_filename": "CLEVR_train_000012.png",
        "image_path": "images/train/CLEVR_train_000012.png",
        "question_index": 30,
        "source_position": 2,
        "question": "Is anything left of the object?",
        "encoded_tokens": [1, 5, 2],
        "answer": 1,
        "relation": "left",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 40,
        "source_position": 1,
        "question": "Is anything in front of the object?",
        "encoded_tokens": [1, 7, 2],
        "answer": 0,
        "relation": "front",
    },
    {
        "partition": "train",
        "official_split": "train",
        "image_index": 2,
        "image_filename": "CLEVR_train_000002.png",
        "image_path": "images/train/CLEVR_train_000002.png",
        "question_index": 10,
        "source_position": 0,
        "question": "Is anything left of the object?",
        "encoded_tokens": [1, 5, 2],
        "answer": 0,
        "relation": "left",
    },
)

_AUDIT_REPORT = {
    "protocol_version": "PUB-R1.1",
    "environment": {"python": "3.13"},
    "partition_summaries": [
        {
            "partition": "train",
            "mother_image_count": 2,
            "selected_image_count": 1,
            "selected_question_count": 2,
        },
        {
            "partition": "dev",
            "mother_image_count": 1,
            "selected_image_count": 1,
            "selected_question_count": 1,
        },
        {
            "partition": "base_test",
            "mother_image_count": 1,
            "selected_image_count": 1,
            "selected_question_count": 2,
        },
        {
            "partition": "confirmation",
            "mother_image_count": 1,
            "selected_image_count": 1,
            "selected_question_count": 1,
        },
    ],
    "cell_counts": [],
    "exclusion_counts": [],
    "source_summaries": [],
    "file_summaries": [],
}

_REPORTING_MANIFEST_ROWS = (
    {
        "partition": "train",
        "official_split": "train",
        "image_index": 2,
        "image_filename": "CLEVR_train_000002.png",
        "image_path": "images/train/CLEVR_train_000002.png",
        "question_index": 10,
        "source_position": 0,
        "question": "Is anything left of the object?",
        "encoded_tokens": [1, 5, 2],
        "answer": 0,
        "relation": "left",
    },
    {
        "partition": "train",
        "official_split": "train",
        "image_index": 2,
        "image_filename": "CLEVR_train_000002.png",
        "image_path": "images/train/CLEVR_train_000002.png",
        "question_index": 20,
        "source_position": 1,
        "question": "Is anything right of the object?",
        "encoded_tokens": [1, 6, 2],
        "answer": 1,
        "relation": "right",
    },
    {
        "partition": "dev",
        "official_split": "train",
        "image_index": 12,
        "image_filename": "CLEVR_train_000012.png",
        "image_path": "images/train/CLEVR_train_000012.png",
        "question_index": 30,
        "source_position": 2,
        "question": "Is anything left of the object?",
        "encoded_tokens": [1, 5, 2],
        "answer": 1,
        "relation": "left",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 100,
        "source_position": 10,
        "question": "Left no?",
        "encoded_tokens": [1, 10, 2],
        "answer": 0,
        "relation": "left",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 101,
        "source_position": 11,
        "question": "Left yes?",
        "encoded_tokens": [1, 11, 2],
        "answer": 1,
        "relation": "left",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 102,
        "source_position": 12,
        "question": "Right no?",
        "encoded_tokens": [1, 12, 2],
        "answer": 0,
        "relation": "right",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 103,
        "source_position": 13,
        "question": "Right yes?",
        "encoded_tokens": [1, 13, 2],
        "answer": 1,
        "relation": "right",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 104,
        "source_position": 14,
        "question": "Front no?",
        "encoded_tokens": [1, 14, 2],
        "answer": 0,
        "relation": "front",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 105,
        "source_position": 15,
        "question": "Front yes?",
        "encoded_tokens": [1, 15, 2],
        "answer": 1,
        "relation": "front",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 106,
        "source_position": 16,
        "question": "Behind no?",
        "encoded_tokens": [1, 16, 2],
        "answer": 0,
        "relation": "behind",
    },
    {
        "partition": "base_test",
        "official_split": "val",
        "image_index": 11,
        "image_filename": "CLEVR_val_000011.png",
        "image_path": "images/val/CLEVR_val_000011.png",
        "question_index": 107,
        "source_position": 17,
        "question": "Behind yes?",
        "encoded_tokens": [1, 17, 2],
        "answer": 1,
        "relation": "behind",
    },
    {
        "partition": "confirmation",
        "official_split": "val",
        "image_index": 13,
        "image_filename": "CLEVR_val_000013.png",
        "image_path": "images/val/CLEVR_val_000013.png",
        "question_index": 60,
        "source_position": 3,
        "question": "Is anything right of the object?",
        "encoded_tokens": [1, 8, 2],
        "answer": 0,
        "relation": "right",
    },
)

_REPORTING_AUDIT_REPORT = {
    "protocol_version": "PUB-R1.1",
    "environment": {"python": "3.13"},
    "partition_summaries": [
        {
            "partition": "train",
            "mother_image_count": 2,
            "selected_image_count": 1,
            "selected_question_count": 2,
        },
        {
            "partition": "dev",
            "mother_image_count": 1,
            "selected_image_count": 1,
            "selected_question_count": 1,
        },
        {
            "partition": "base_test",
            "mother_image_count": 1,
            "selected_image_count": 1,
            "selected_question_count": 8,
        },
        {
            "partition": "confirmation",
            "mother_image_count": 1,
            "selected_image_count": 1,
            "selected_question_count": 1,
        },
    ],
    "cell_counts": [],
    "exclusion_counts": [],
    "source_summaries": [],
    "file_summaries": [],
}

_BALANCED_PREDICTION_ROWS = (
    {"question_index": 100, "label": 0, "logits": [0.0, 0.0]},
    {"question_index": 101, "label": 1, "logits": [0.0, 0.0]},
    {"question_index": 102, "label": 0, "logits": [0.0, 0.0]},
    {"question_index": 103, "label": 1, "logits": [0.0, 0.0]},
    {"question_index": 104, "label": 0, "logits": [0.0, 0.0]},
    {"question_index": 105, "label": 1, "logits": [0.0, 0.0]},
    {"question_index": 106, "label": 0, "logits": [0.0, 0.0]},
    {"question_index": 107, "label": 1, "logits": [0.0, 0.0]},
)

_FORMAL_RUN_KEYS = (
    ("G", 101), ("G", 102), ("G", 103), ("G", 104), ("G", 105),
    ("B", 101), ("B", 102), ("B", 103), ("B", 104), ("B", 105),
    ("M10", 101), ("M10", 102), ("M10", 103), ("M10", 104), ("M10", 105),
    ("Q-only", 101), ("Q-only", 102), ("Q-only", 103),
    ("Q-only", 104), ("Q-only", 105),
)


def _write_run_provenance(audit_dir, prediction_path, method, seed):
    names = ("selected_questions.jsonl", "vocabulary.json", "audit_report.json", "image_partitions.json")
    config = {
        "protocol_version": "PUB-R1.1", "phase": "formal", "method": method,
        "seed": seed, "target_updates": 8000,
        "data": {"data_root": None if method == "Q-only" else "/explicit/rgb",
                 "audit_artifacts": {name: (audit_dir / name).read_bytes().decode() for name in names}},
    }
    prediction_path.with_suffix(".run_config.json").write_text(json.dumps(config))


class AuditBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.audit_dir = Path(self.temporary_directory.name) / "audit"
        self.audit_dir.mkdir()
        self._reset_bundle()

    def _reset_bundle(self) -> None:
        self.image_partitions = copy.deepcopy(_IMAGE_PARTITIONS)
        self.manifest_rows = copy.deepcopy(list(_MANIFEST_ROWS))
        self.audit_report = copy.deepcopy(_AUDIT_REPORT)
        self._write_bundle()

    def _write_bundle(self) -> None:
        (self.audit_dir / "image_partitions.json").write_text(
            json.dumps(self.image_partitions, indent=2) + "\n", encoding="utf-8"
        )
        self._write_manifest()
        (self.audit_dir / "audit_report.json").write_text(
            json.dumps(self.audit_report, indent=2) + "\n", encoding="utf-8"
        )

    def _write_manifest(self) -> None:
        (self.audit_dir / "selected_questions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.manifest_rows),
            encoding="utf-8",
        )

    def _assert_bundle_error(
        self, artifact_name: str | None = None, line: int | None = None
    ) -> str:
        with self.assertRaises(CLEVRReportError) as caught:
            load_audit_targets(self.audit_dir)
        message = str(caught.exception)
        if artifact_name is not None:
            expected_path = str((self.audit_dir / artifact_name).resolve())
            self.assertIn(expected_path, message)
        if line is not None:
            self.assertIn(f"line {line}", message)
        return message

    def test_valid_bundle_returns_canonical_targets_evidence_and_counts(self) -> None:
        # Mutation caught: trusting manifest order or deriving evidence/counts indirectly.
        targets, evidence, counts = load_audit_targets(self.audit_dir)

        self.assertEqual(
            targets,
            (
                TargetRow(
                    partition="train",
                    official_split="train",
                    image_index=2,
                    image_filename="CLEVR_train_000002.png",
                    question_index=10,
                    relation="left",
                    label=0,
                ),
                TargetRow(
                    partition="train",
                    official_split="train",
                    image_index=2,
                    image_filename="CLEVR_train_000002.png",
                    question_index=20,
                    relation="right",
                    label=1,
                ),
                TargetRow(
                    partition="base_test",
                    official_split="val",
                    image_index=11,
                    image_filename="CLEVR_val_000011.png",
                    question_index=40,
                    relation="front",
                    label=0,
                ),
                TargetRow(
                    partition="base_test",
                    official_split="val",
                    image_index=11,
                    image_filename="CLEVR_val_000011.png",
                    question_index=50,
                    relation="behind",
                    label=1,
                ),
            ),
        )
        self.assertEqual(
            evidence,
            (
                ArtifactEvidence(
                    path=str((self.audit_dir / "image_partitions.json").resolve()),
                    size_bytes=(self.audit_dir / "image_partitions.json").stat().st_size,
                    record_count=5,
                ),
                ArtifactEvidence(
                    path=str((self.audit_dir / "selected_questions.jsonl").resolve()),
                    size_bytes=(self.audit_dir / "selected_questions.jsonl").stat().st_size,
                    record_count=6,
                ),
                ArtifactEvidence(
                    path=str((self.audit_dir / "audit_report.json").resolve()),
                    size_bytes=(self.audit_dir / "audit_report.json").stat().st_size,
                    record_count=1,
                ),
            ),
        )
        self.assertEqual(
            counts, PartitionCounts(train=2, dev=1, base_test=2, confirmation=1)
        )

    def test_loads_artifacts_serialized_from_existing_audit_producer(self) -> None:
        # Mutation caught: mapping producer-created dev images to official val.
        producer_dir = self.audit_dir.parent / "producer-audit"
        producer_dir.mkdir()
        data_root = self.audit_dir.parent / "CLEVR_v1.0"
        partitions = build_image_partitions(
            tuple(
                ImageIdentity("train", index, f"CLEVR_train_{index:06d}.png")
                for index in range(10)
            ),
            tuple(
                ImageIdentity("val", index, f"CLEVR_val_{index:06d}.png")
                for index in range(2)
            ),
        )
        selected_questions = tuple(
            SelectedQuestion(
                partition=partition,
                official_split=image.official_split,
                image_index=image.image_index,
                image_filename=image.image_filename,
                image_path=data_root
                / "images"
                / image.official_split
                / image.image_filename,
                question_index=question_index,
                source_position=source_position,
                question=f"{partition} question?",
                encoded_tokens=(1, 3, 2),
                answer=answer,
                relation=relation,
            )
            for partition, image, question_index, source_position, answer, relation in (
                ("train", partitions.train[0], 10, 100, 0, "left"),
                ("dev", partitions.dev[0], 11, 101, 1, "right"),
                ("base_test", partitions.base_test[0], 20, 200, 0, "front"),
                (
                    "confirmation",
                    partitions.confirmation[0],
                    21,
                    201,
                    1,
                    "behind",
                ),
            )
        )
        audit = AuditResult(
            selected_questions=selected_questions,
            vocabulary=build_vocabulary(("train question?",)),
            image_partitions=partitions,
            cell_counts=(),
            exclusion_counts=(),
            partition_summaries=tuple(
                PartitionSummary(
                    partition=partition,
                    mother_image_count=len(getattr(partitions, partition)),
                    selected_image_count=1,
                    selected_question_count=1,
                )
                for partition in ("train", "dev", "base_test", "confirmation")
            ),
            source_summaries=(),
            file_summaries=(),
        )
        audit_producer._write_artifacts(producer_dir, audit, data_root)

        targets, _, counts = load_audit_targets(producer_dir)

        self.assertEqual(
            tuple((target.partition, target.official_split) for target in targets),
            (("train", "train"), ("base_test", "val")),
        )
        self.assertEqual(
            counts,
            PartitionCounts(train=1, dev=1, base_test=1, confirmation=1),
        )

    def test_symlinked_audit_inputs_record_resolved_target_paths(self) -> None:
        # Mutation caught: discarding each reader's resolved artifact identity.
        target_dir = self.audit_dir.parent / "resolved-artifacts"
        target_dir.mkdir()
        target_paths = []
        for name in (
            "image_partitions.json",
            "selected_questions.jsonl",
            "audit_report.json",
        ):
            fixed_path = self.audit_dir / name
            target_path = target_dir / name
            fixed_path.rename(target_path)
            fixed_path.symlink_to(target_path)
            target_paths.append(target_path)

        _, evidence, _ = load_audit_targets(self.audit_dir)

        self.assertEqual(
            tuple(item.path for item in evidence),
            (
                str(target_paths[0]),
                str(target_paths[1]),
                str(target_paths[2]),
            ),
        )

    def test_output_records_are_frozen_and_strictly_typed(self) -> None:
        # Mutation caught: mutable records or bool/coercible values crossing the boundary.
        evidence = ArtifactEvidence("/tmp/evidence.json", 0, 0)
        counts = PartitionCounts(0, 1, 2, 3)
        with self.assertRaises(FrozenInstanceError):
            evidence.record_count = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            counts.train = 1  # type: ignore[misc]

        invalid_evidence = (
            ("", 0, 0),
            ("   ", 0, 0),
            ("relative/file.json", 0, 0),
            (Path("/tmp/file.json"), 0, 0),
            ("/tmp/file.json", True, 0),
            ("/tmp/file.json", -1, 0),
            ("/tmp/file.json", 0, False),
            ("/tmp/file.json", 0, -1),
        )
        for arguments in invalid_evidence:
            with self.subTest(evidence=arguments):
                with self.assertRaises(ValueError):
                    ArtifactEvidence(*arguments)  # type: ignore[arg-type]

        invalid_counts = (
            (True, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, "1", 0, 0),
            (0, 0, 1.0, 0),
        )
        for arguments in invalid_counts:
            with self.subTest(counts=arguments):
                with self.assertRaises(ValueError):
                    PartitionCounts(*arguments)  # type: ignore[arg-type]

    def test_requires_path_directory_and_three_regular_inputs(self) -> None:
        # Mutation caught: path coercion, discovery, or silently skipping missing inputs.
        with self.assertRaises(CLEVRReportError):
            load_audit_targets(str(self.audit_dir))  # type: ignore[arg-type]
        with self.assertRaises(CLEVRReportError) as absent_directory:
            load_audit_targets(self.audit_dir / "absent")
        self.assertIn(str((self.audit_dir / "absent").resolve()), str(absent_directory.exception))
        with self.assertRaises(CLEVRReportError):
            load_audit_targets(self.audit_dir / "audit_report.json")

        missing = self.audit_dir / "selected_questions.jsonl"
        missing.unlink()
        self._assert_bundle_error("selected_questions.jsonl")

        self._reset_bundle()
        not_file = self.audit_dir / "audit_report.json"
        not_file.unlink()
        not_file.mkdir()
        self._assert_bundle_error("audit_report.json")

    def test_rejects_wrong_and_boolean_protocol_fields(self) -> None:
        # Mutation caught: accepting another protocol or Python bool as protocol metadata.
        for artifact_name, value in (
            ("image_partitions.json", "PUB-R1.0"),
            ("image_partitions.json", True),
            ("audit_report.json", "PUB-R1.0"),
            ("audit_report.json", False),
        ):
            with self.subTest(artifact=artifact_name, value=value):
                self._reset_bundle()
                document = (
                    self.image_partitions
                    if artifact_name == "image_partitions.json"
                    else self.audit_report
                )
                document["protocol_version"] = value
                self._write_bundle()
                self._assert_bundle_error(artifact_name)

    def test_rejects_duplicate_keys_at_top_level_nested_and_jsonl(self) -> None:
        # Mutation caught: json's default last-duplicate-key-wins behavior.
        partition_path = self.audit_dir / "image_partitions.json"
        partition_path.write_text(
            '{"protocol_version":"PUB-R1.1","protocol_version":"PUB-R1.1",'
            '"partitions":{}}',
            encoding="utf-8",
        )
        self._assert_bundle_error("image_partitions.json")

        self._reset_bundle()
        report_path = self.audit_dir / "audit_report.json"
        report_text = report_path.read_text(encoding="utf-8")
        report_path.write_text(
            report_text.replace(
                '"python": "3.13"', '"python": "3.13", "python": "3.13"'
            ),
            encoding="utf-8",
        )
        self._assert_bundle_error("audit_report.json")

        self._reset_bundle()
        manifest_path = self.audit_dir / "selected_questions.jsonl"
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace('"answer": 0', '"answer": 0, "answer": 1')
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._assert_bundle_error("selected_questions.jsonl", line=2)

    def test_rejects_nonstandard_nan_and_infinity_constants(self) -> None:
        # Mutation caught: Python JSON extensions admitting non-finite constants.
        report_path = self.audit_dir / "audit_report.json"
        report_path.write_text(
            report_path.read_text(encoding="utf-8").replace(
                '"python": "3.13"', '"python": NaN'
            ),
            encoding="utf-8",
        )
        self._assert_bundle_error("audit_report.json")

        self._reset_bundle()
        manifest_path = self.audit_dir / "selected_questions.jsonl"
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        lines[3] = lines[3].replace('"encoded_tokens": [1, 5, 2]', '"encoded_tokens": [1, Infinity, 2]')
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._assert_bundle_error("selected_questions.jsonl", line=4)

    def test_rejects_blank_empty_and_nonobject_manifest_records(self) -> None:
        # Mutation caught: dropping blank rows or treating arbitrary JSON values as records.
        manifest_path = self.audit_dir / "selected_questions.jsonl"
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        manifest_path.write_text("\n".join(lines[:2] + ["   "] + lines[2:]) + "\n", encoding="utf-8")
        self._assert_bundle_error("selected_questions.jsonl", line=3)

        self._reset_bundle()
        manifest_path.write_text("[]\n", encoding="utf-8")
        self._assert_bundle_error("selected_questions.jsonl", line=1)

        self._reset_bundle()
        manifest_path.write_text("", encoding="utf-8")
        self._assert_bundle_error("selected_questions.jsonl")

    def test_rejects_exact_key_mismatches_in_every_audit_schema(self) -> None:
        # Mutation caught: accepting missing or extra fields from adjacent/runner schemas.
        cases = (
            ("image_root_extra", "image_partitions.json"),
            ("image_record_missing", "image_partitions.json"),
            ("manifest_extra", "selected_questions.jsonl"),
            ("report_root_extra", "audit_report.json"),
        )
        for case, artifact_name in cases:
            with self.subTest(case=case):
                self._reset_bundle()
                if case == "image_root_extra":
                    self.image_partitions["unexpected"] = []
                elif case == "image_record_missing":
                    del self.image_partitions["partitions"]["train"][0]["image_filename"]
                elif case == "manifest_extra":
                    self.manifest_rows[2]["runner_accuracy"] = 1.0
                else:
                    self.audit_report["unexpected"] = []
                self._write_bundle()
                self._assert_bundle_error(
                    artifact_name, line=3 if case == "manifest_extra" else None
                )

    def test_rejects_boolean_and_negative_identity_fields(self) -> None:
        # Mutation caught: bool-as-int coercion or accepting negative source identities.
        cases = (
            ("image_index", True, "image_partitions.json", None),
            ("image_index", -1, "image_partitions.json", None),
            ("question_index", True, "selected_questions.jsonl", 1),
            ("question_index", -1, "selected_questions.jsonl", 1),
            ("source_position", False, "selected_questions.jsonl", 1),
            ("source_position", -1, "selected_questions.jsonl", 1),
        )
        for field, value, artifact_name, line in cases:
            with self.subTest(field=field, value=value):
                self._reset_bundle()
                if artifact_name == "image_partitions.json":
                    self.image_partitions["partitions"]["train"][0][field] = value
                else:
                    self.manifest_rows[0][field] = value
                self._write_bundle()
                self._assert_bundle_error(artifact_name, line=line)

    def test_rejects_split_filename_and_image_path_mismatches(self) -> None:
        # Mutation caught: independently trusting redundant image identity fields.
        cases = (
            ("official_split", "train"),
            ("image_filename", "CLEVR_val_999999.png"),
            ("image_path", "images/val/CLEVR_val_999999.png"),
        )
        for field, value in cases:
            with self.subTest(field=field):
                self._reset_bundle()
                self.manifest_rows[0][field] = value
                self._write_bundle()
                self._assert_bundle_error("selected_questions.jsonl", line=1)

    def test_rejects_question_token_answer_relation_and_partition_types(self) -> None:
        # Mutation caught: weak manifest field checks or bool labels/tokens.
        cases = (
            ("question", 123),
            ("encoded_tokens", "1 2"),
            ("encoded_tokens", [1, True, 2]),
            ("encoded_tokens", [1, -1, 2]),
            ("answer", True),
            ("answer", 2),
            ("relation", "above"),
            ("partition", "holdout"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self._reset_bundle()
                self.manifest_rows[4][field] = value
                self._write_bundle()
                self._assert_bundle_error("selected_questions.jsonl", line=5)

    def test_rejects_duplicate_question_and_source_positions(self) -> None:
        # Mutation caught: allowing two rows to claim one official source identity.
        for field in ("question_index", "source_position"):
            with self.subTest(field=field):
                self._reset_bundle()
                self.manifest_rows[5][field] = self.manifest_rows[2][field]
                self._write_bundle()
                self._assert_bundle_error("selected_questions.jsonl", line=6)

    def test_rejects_duplicate_images_and_wrong_partition_official_split(self) -> None:
        # Mutation caught: one image inhabiting multiple partitions or wrong mother split.
        self.image_partitions["partitions"]["dev"][0] = copy.deepcopy(
            self.image_partitions["partitions"]["base_test"][0]
        )
        self._write_bundle()
        self._assert_bundle_error("image_partitions.json")

        self._reset_bundle()
        self.image_partitions["partitions"]["dev"][0] = {
            "official_split": "val",
            "image_index": 12,
            "image_filename": "CLEVR_val_000012.png",
        }
        self._write_bundle()
        self._assert_bundle_error("image_partitions.json")

    def test_rejects_selected_images_absent_or_assigned_to_another_partition(self) -> None:
        # Mutation caught: accepting a schema-valid manifest from another audit cohort.
        self.manifest_rows[2].update(
            {
                "image_index": 999,
                "image_filename": "CLEVR_train_000999.png",
                "image_path": "images/train/CLEVR_train_000999.png",
            }
        )
        self._write_bundle()
        self._assert_bundle_error("selected_questions.jsonl", line=3)

        self._reset_bundle()
        self.manifest_rows[3].update(
            {
                "image_index": 2,
                "image_filename": "CLEVR_train_000002.png",
                "image_path": "images/train/CLEVR_train_000002.png",
            }
        )
        self._write_bundle()
        self._assert_bundle_error("selected_questions.jsonl", line=4)

    def test_rejects_wrong_summary_counts_and_noninteger_counts(self) -> None:
        # Mutation caught: trusting report totals without recomputing bundle membership.
        for field in (
            "mother_image_count",
            "selected_image_count",
            "selected_question_count",
        ):
            with self.subTest(field=field):
                self._reset_bundle()
                self.audit_report["partition_summaries"][0][field] += 1
                self._write_bundle()
                self._assert_bundle_error("audit_report.json")

        for value in (True, -1, "1"):
            with self.subTest(summary_count=value):
                self._reset_bundle()
                self.audit_report["partition_summaries"][1][
                    "selected_question_count"
                ] = value
                self._write_bundle()
                self._assert_bundle_error("audit_report.json")

    def test_rejects_wrong_summary_order_schema_and_report_container_types(self) -> None:
        # Mutation caught: accepting reordered/incomplete summaries or unchecked containers.
        self.audit_report["partition_summaries"][0:2] = reversed(
            self.audit_report["partition_summaries"][0:2]
        )
        self._write_bundle()
        self._assert_bundle_error("audit_report.json")

        self._reset_bundle()
        self.audit_report["partition_summaries"][0]["unexpected"] = 0
        self._write_bundle()
        self._assert_bundle_error("audit_report.json")

        for field, wrong_value in (
            ("environment", []),
            ("cell_counts", {}),
            ("exclusion_counts", {}),
            ("source_summaries", {}),
            ("file_summaries", {}),
            ("partition_summaries", {}),
        ):
            with self.subTest(field=field):
                self._reset_bundle()
                self.audit_report[field] = wrong_value
                self._write_bundle()
                self._assert_bundle_error("audit_report.json")

    def test_requires_nonempty_train_and_base_test_targets(self) -> None:
        # Mutation caught: returning a cohort that cannot support the formal comparison.
        for partition in ("train", "base_test"):
            with self.subTest(partition=partition):
                self._reset_bundle()
                self.manifest_rows = [
                    row for row in self.manifest_rows if row["partition"] != partition
                ]
                summary = next(
                    item
                    for item in self.audit_report["partition_summaries"]
                    if item["partition"] == partition
                )
                summary["selected_image_count"] = 0
                summary["selected_question_count"] = 0
                self._write_bundle()
                self._assert_bundle_error("selected_questions.jsonl")


class PredictionFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.prediction_path = self.root / "declared-run.jsonl"
        self.rows = [
            {
                "protocol_version": "PUB-R1.1",
                "method": "B",
                "seed": 103,
                "question_index": 9,
                "label": 1,
                "logits": [-0.25, 0.75],
            },
            {
                "protocol_version": "PUB-R1.1",
                "method": "B",
                "seed": 103,
                "question_index": 3,
                "label": 0,
                "logits": [1.0, -1.0],
            },
        ]
        self._write_rows()

    def _write_rows(self) -> None:
        self.prediction_path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )

    def _assert_prediction_error(self, line: int | None = None) -> str:
        with self.assertRaises(CLEVRReportError) as caught:
            load_prediction_run(PredictionSource(self.prediction_path))
        message = str(caught.exception)
        self.assertIn(str(self.prediction_path.resolve()), message)
        if line is not None:
            self.assertIn(f"line {line}", message)
        return message

    def test_loads_literal_self_described_run_and_source_evidence(self) -> None:
        # Mutation caught: trusting filename/order or retaining mutable JSON arrays.
        run, evidence = load_prediction_run(PredictionSource(self.prediction_path))

        self.assertEqual(
            run,
            RunPredictions(
                method="B",
                seed=103,
                rows=(
                    PredictionRow(3, 0, (1.0, -1.0)),
                    PredictionRow(9, 1, (-0.25, 0.75)),
                ),
            ),
        )
        self.assertEqual(
            evidence,
            PredictionEvidence(
                method="B",
                seed=103,
                source_path=str(self.prediction_path.resolve()),
                size_bytes=self.prediction_path.stat().st_size,
                row_count=2,
                canonical_path="predictions/B_seed_103.jsonl",
            ),
        )

    def test_prediction_records_are_frozen_and_reject_inconsistent_values(self) -> None:
        # Mutation caught: mutable descriptors or evidence detached from its run identity.
        source = PredictionSource(Path("/tmp/predictions.jsonl"))
        evidence = PredictionEvidence(
            "G", 101, "/tmp/predictions.jsonl", 1, 1,
            "predictions/G_seed_101.jsonl",
        )
        with self.assertRaises(FrozenInstanceError):
            source.path = Path("/tmp/other.jsonl")  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            evidence.row_count = 2  # type: ignore[misc]

        for invalid_path in ("/tmp/predictions.jsonl", None, 1):
            with self.subTest(source_path=invalid_path):
                with self.assertRaises(ValueError):
                    PredictionSource(invalid_path)  # type: ignore[arg-type]

        invalid_evidence = (
            (True, 101, "/tmp/p.jsonl", 1, 1, "predictions/G_seed_101.jsonl"),
            ("other", 101, "/tmp/p.jsonl", 1, 1, "predictions/other_seed_101.jsonl"),
            ("G", True, "/tmp/p.jsonl", 1, 1, "predictions/G_seed_101.jsonl"),
            ("G", 999, "/tmp/p.jsonl", 1, 1, "predictions/G_seed_999.jsonl"),
            ("G", 101, "relative.jsonl", 1, 1, "predictions/G_seed_101.jsonl"),
            ("G", 101, "/tmp/p.jsonl", True, 1, "predictions/G_seed_101.jsonl"),
            ("G", 101, "/tmp/p.jsonl", -1, 1, "predictions/G_seed_101.jsonl"),
            ("G", 101, "/tmp/p.jsonl", 1, 0, "predictions/G_seed_101.jsonl"),
            ("G", 101, "/tmp/p.jsonl", 1, 1, "G_seed_101.jsonl"),
            ("G", 101, "/tmp/p.jsonl", 1, 1, "predictions/B_seed_101.jsonl"),
        )
        for arguments in invalid_evidence:
            with self.subTest(evidence=arguments):
                with self.assertRaises(ValueError):
                    PredictionEvidence(*arguments)  # type: ignore[arg-type]

    def test_requires_prediction_source_and_regular_file(self) -> None:
        # Mutation caught: coercing caller input or silently accepting absent/directories.
        with self.assertRaises(CLEVRReportError):
            load_prediction_run(self.prediction_path)  # type: ignore[arg-type]

        missing = self.root / "missing.jsonl"
        with self.assertRaises(CLEVRReportError) as absent:
            load_prediction_run(PredictionSource(missing))
        self.assertIn(str(missing.resolve()), str(absent.exception))

        directory = self.root / "directory.jsonl"
        directory.mkdir()
        with self.assertRaises(CLEVRReportError) as not_file:
            load_prediction_run(PredictionSource(directory))
        self.assertIn(str(directory.resolve()), str(not_file.exception))

    def test_rejects_empty_blank_and_nonobject_prediction_records(self) -> None:
        # Mutation caught: skipping records that do not describe one prediction row.
        self.prediction_path.write_text("", encoding="utf-8")
        self._assert_prediction_error()

        self.prediction_path.write_text(
            json.dumps(self.rows[0]) + "\n   \n" + json.dumps(self.rows[1]) + "\n",
            encoding="utf-8",
        )
        self._assert_prediction_error(line=2)

        for value in ([], "row", 3):
            with self.subTest(value=value):
                self.prediction_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                self._assert_prediction_error(line=1)

    def test_rejects_duplicate_keys_and_nonstandard_json_numbers(self) -> None:
        # Mutation caught: JSON last-key-wins or Python's NaN/Infinity extensions.
        self.prediction_path.write_text(
            '{"protocol_version":"PUB-R1.1","method":"B","seed":103,'
            '"question_index":3,"question_index":9,"label":0,"logits":[0,0]}\n',
            encoding="utf-8",
        )
        self._assert_prediction_error(line=1)

        for nonstandard in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(nonstandard=nonstandard):
                self.prediction_path.write_text(
                    '{"protocol_version":"PUB-R1.1","method":"B","seed":103,'
                    f'"question_index":3,"label":0,"logits":[{nonstandard},0]}}\n',
                    encoding="utf-8",
                )
                self._assert_prediction_error(line=1)

    def test_rejects_exact_key_mismatches_including_runner_accuracy(self) -> None:
        # Mutation caught: ingesting runner summaries or defaulting omitted fields.
        for mutation in ("missing", "extra", "runner"):
            with self.subTest(mutation=mutation):
                self.rows = copy.deepcopy(self.rows)
                if mutation == "missing":
                    del self.rows[0]["label"]
                elif mutation == "extra":
                    self.rows[0]["unexpected"] = 0
                else:
                    self.rows[0]["accuracy"] = 1.0
                self._write_rows()
                self._assert_prediction_error(line=1)
                self.rows = copy.deepcopy(PredictionFileTests._literal_rows())

    @staticmethod
    def _literal_rows() -> list[dict[str, object]]:
        return [
            {
                "protocol_version": "PUB-R1.1",
                "method": "B",
                "seed": 103,
                "question_index": 9,
                "label": 1,
                "logits": [-0.25, 0.75],
            },
            {
                "protocol_version": "PUB-R1.1",
                "method": "B",
                "seed": 103,
                "question_index": 3,
                "label": 0,
                "logits": [1.0, -1.0],
            },
        ]

    def test_rejects_wrong_protocol_and_method_seed_domains(self) -> None:
        # Mutation caught: accepting adjacent protocols, bool aliases, or informal runs.
        cases = (
            ("protocol_version", "PUB-R1.0"),
            ("protocol_version", True),
            ("method", "other"),
            ("method", True),
            ("seed", 999),
            ("seed", True),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.rows = self._literal_rows()
                self.rows[0][field] = value
                self._write_rows()
                self._assert_prediction_error(line=1)

    def test_rejects_method_seed_disagreement_between_rows(self) -> None:
        # Mutation caught: assigning file identity from only one arbitrary row.
        for field, value in (("method", "G"), ("seed", 104)):
            with self.subTest(field=field):
                self.rows = self._literal_rows()
                self.rows[1][field] = value
                self._write_rows()
                self._assert_prediction_error(line=2)

    def test_rejects_duplicate_question_ids(self) -> None:
        # Mutation caught: allowing one prediction to overwrite another by ID.
        self.rows[1]["question_index"] = self.rows[0]["question_index"]
        self._write_rows()
        self._assert_prediction_error(line=2)

    def test_rejects_every_prediction_row_boundary(self) -> None:
        # Mutation caught: bool/int coercion, invalid labels, or malformed/nonfinite logits.
        cases = (
            ("question_index", True),
            ("question_index", -1),
            ("question_index", 3.0),
            ("label", False),
            ("label", 2),
            ("label", 1.0),
            ("logits", "0 0"),
            ("logits", [0.0]),
            ("logits", [0.0, 0.0, 0.0]),
            ("logits", [True, 0.0]),
            ("logits", ["0", 0.0]),
            ("logits", [1e309, 0.0]),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.rows = self._literal_rows()
                self.rows[0][field] = value
                self._write_rows()
                self._assert_prediction_error(line=1)


class StatisticsBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.audit_dir = self.root / "audit"
        self.audit_dir.mkdir()
        self.manifest_rows = copy.deepcopy(list(_REPORTING_MANIFEST_ROWS))
        self._write_audit_bundle()

        self.prediction_dir = self.root / "inputs"
        self.prediction_dir.mkdir()
        self.sources: list[PredictionSource] = []
        self.source_by_key: dict[tuple[str, int], PredictionSource] = {}
        for method, seed in _FORMAL_RUN_KEYS:
            path = self.prediction_dir / f"input-{method}-{seed}.jsonl"
            self._write_prediction_file(path, method, seed)
            _write_run_provenance(self.audit_dir, path, method, seed)
            source = PredictionSource(path)
            self.sources.append(source)
            self.source_by_key[(method, seed)] = source

    def _write_audit_bundle(self) -> None:
        (self.audit_dir / "vocabulary.json").write_text('{"tokens":["red"]}\n')
        (self.audit_dir / "image_partitions.json").write_text(
            json.dumps(_IMAGE_PARTITIONS, indent=2) + "\n", encoding="utf-8"
        )
        (self.audit_dir / "selected_questions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.manifest_rows),
            encoding="utf-8",
        )
        (self.audit_dir / "audit_report.json").write_text(
            json.dumps(_REPORTING_AUDIT_REPORT, indent=2) + "\n",
            encoding="utf-8",
        )
        for (method, seed), source in getattr(self, "source_by_key", {}).items():
            _write_run_provenance(self.audit_dir, source.path, method, seed)

    @staticmethod
    def _write_prediction_file(path: Path, method: str, seed: int) -> None:
        rows = [
            {
                "protocol_version": "PUB-R1.1",
                "method": method,
                "seed": seed,
                "question_index": row["question_index"],
                "label": row["label"],
                "logits": row["logits"],
            }
            for row in _BALANCED_PREDICTION_ROWS
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _build(self) -> StatisticsBundle:
        return build_statistics_bundle(self.audit_dir, self.sources)

    def test_builds_complete_ordered_grid_with_hand_checked_statistics(self) -> None:
        # Mutation caught: omitting/reordering a run or trusting any input metric.
        bundle = self._build()
        report = bundle.report

        self.assertEqual(report.protocol_version, "PUB-R1.1")
        self.assertEqual(report.report_schema_version, 2)
        self.assertEqual(report.targets_path, "targets.jsonl")
        self.assertEqual(
            report.manifest_partition_counts,
            PartitionCounts(train=2, dev=1, base_test=8, confirmation=1),
        )
        self.assertEqual(
            report.environment,
            EnvironmentEvidence(
                python=platform.python_version(),
                numpy=importlib.metadata.version("numpy"),
            ),
        )
        self.assertEqual(
            tuple(Path(item.path).name for item in report.audit_sources),
            (
                "image_partitions.json",
                "selected_questions.jsonl",
                "audit_report.json",
            ),
        )
        self.assertEqual(
            tuple((run.method, run.seed) for run in bundle.runs),
            _FORMAL_RUN_KEYS,
        )
        self.assertEqual(
            tuple((item.method, item.seed) for item in report.predictions),
            _FORMAL_RUN_KEYS,
        )
        for evidence, key in zip(
            report.predictions, _FORMAL_RUN_KEYS, strict=True
        ):
            method, seed = key
            source_path = self.source_by_key[key].path.resolve()
            self.assertEqual(evidence.source_path, str(source_path))
            self.assertEqual(evidence.size_bytes, source_path.stat().st_size)
            self.assertEqual(evidence.row_count, 8)
            self.assertEqual(
                evidence.canonical_path,
                f"predictions/{method}_seed_{seed}.jsonl",
            )

        expected_cells = (
            ("left", 0, 1, 1, 1.0),
            ("left", 1, 1, 0, 0.0),
            ("right", 0, 1, 1, 1.0),
            ("right", 1, 1, 0, 0.0),
            ("front", 0, 1, 1, 1.0),
            ("front", 1, 1, 0, 0.0),
            ("behind", 0, 1, 1, 1.0),
            ("behind", 1, 1, 0, 0.0),
        )
        for run_metrics in report.statistics.run_metrics:
            metrics = run_metrics.metrics
            self.assertEqual(metrics.question_count, 8)
            self.assertEqual(metrics.correct_count, 4)
            self.assertEqual(metrics.accuracy, 0.5)
            self.assertAlmostEqual(metrics.cross_entropy, math.log(2.0))
            self.assertEqual(metrics.macro_accuracy, 0.5)
            self.assertEqual(
                tuple(
                    (
                        cell.relation,
                        cell.label,
                        cell.count,
                        cell.correct_count,
                        cell.accuracy,
                    )
                    for cell in metrics.cells
                ),
                expected_cells,
            )

        self.assertEqual(
            tuple(summary.values for summary in report.statistics.method_summaries),
            ((0.5,) * 5,) * 4,
        )
        for paired in (
            report.statistics.b_minus_g,
            report.statistics.m10_minus_g,
        ):
            self.assertEqual(paired.values, (0.0,) * 5)
            self.assertEqual(paired.mean, 0.0)
            self.assertEqual(paired.sample_sd, 0.0)
            self.assertEqual(paired.interval, (0.0, 0.0))

        majority = report.statistics.train_majority
        self.assertEqual(majority.correct_count, 4)
        self.assertEqual(majority.accuracy, 0.5)
        self.assertIsNone(majority.cross_entropy)
        self.assertEqual(majority.macro_accuracy, 0.5)
        self.assertEqual(
            tuple(cell.accuracy for cell in majority.cells),
            (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        )

        bootstrap = report.statistics.bootstrap
        self.assertEqual(bootstrap.seed, 31001)
        self.assertEqual(bootstrap.draws, 2000)
        self.assertEqual(bootstrap.quantile_method, "linear")
        self.assertEqual(len(bootstrap.replicates), 2000)
        self.assertEqual(set(bootstrap.replicates), {0.0})
        self.assertEqual(bootstrap.interval, (0.0, 0.0))

    def test_serializes_fixed_payload_and_only_explicit_na_locations(self) -> None:
        # Mutation caught: generic asdict/null replacement or dropped bootstrap draws.
        payload = statistics_report_payload(self._build().report)

        self.assertEqual(
            tuple(payload),
            (
                "protocol_version",
                "report_schema_version",
                "environment",
                "audit_sources",
                "targets_path",
                "manifest_partition_counts",
                "predictions",
                "statistics",
            ),
        )
        self.assertEqual(
            payload["environment"],
            {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
            },
        )
        statistics = payload["statistics"]
        self.assertIsInstance(statistics, dict)
        self.assertEqual(
            tuple(statistics),
            (
                "run_metrics",
                "method_summaries",
                "b_minus_g",
                "m10_minus_g",
                "train_majority",
                "bootstrap",
            ),
        )
        self.assertEqual(
            statistics["train_majority"]["cross_entropy"],  # type: ignore[index]
            "NA",
        )
        replicates = statistics["bootstrap"]["replicates"]  # type: ignore[index]
        self.assertIsInstance(replicates, list)
        self.assertEqual(len(replicates), 2000)
        self.assertEqual(set(replicates), {0.0})

        none_paths: list[tuple[object, ...]] = []
        na_paths: list[tuple[object, ...]] = []

        def visit(value: object, path: tuple[object, ...] = ()) -> None:
            if value is None:
                none_paths.append(path)
            elif value == "NA":
                na_paths.append(path)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, path + (key,))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, path + (index,))

        visit(payload)
        self.assertEqual(none_paths, [])
        self.assertEqual(
            na_paths,
            [("statistics", "train_majority", "cross_entropy")],
        )
        self.assertTrue(
            all(
                path[-1] in {"accuracy", "macro_accuracy", "cross_entropy"}
                for path in na_paths
            )
        )
        json.dumps(payload, allow_nan=False)

    def test_report_records_are_frozen_and_reject_inconsistent_construction(self) -> None:
        # Mutation caught: mutable reports or records that accept noncanonical members.
        bundle = self._build()
        report = bundle.report
        with self.assertRaises(FrozenInstanceError):
            report.report_schema_version = 2  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            bundle.runs = ()  # type: ignore[misc]

        for arguments in (("", "2.5"), ("3.13", ""), (True, "2.5")):
            with self.subTest(environment=arguments):
                with self.assertRaises(ValueError):
                    EnvironmentEvidence(*arguments)  # type: ignore[arg-type]

        invalid_reports = (
            {"protocol_version": "PUB-R1.0"},
            {"protocol_version": True},
            {"report_schema_version": 1},
            {"report_schema_version": True},
            {"audit_sources": list(report.audit_sources)},
            {"audit_sources": report.audit_sources[:2]},
            {"targets_path": "other.jsonl"},
            {"manifest_partition_counts": "counts"},
            {"predictions": tuple(reversed(report.predictions))},
            {"statistics": "statistics"},
            {"environment": "environment"},
        )
        for changes in invalid_reports:
            with self.subTest(report=changes):
                with self.assertRaises(ValueError):
                    replace(report, **changes)

        invalid_bundles = (
            {"report": "report"},
            {"targets": list(bundle.targets)},
            {"targets": tuple(reversed(bundle.targets))},
            {"runs": list(bundle.runs)},
            {"runs": bundle.runs[:-1]},
            {"runs": tuple(reversed(bundle.runs))},
        )
        for changes in invalid_bundles:
            with self.subTest(bundle=changes):
                with self.assertRaises(ValueError):
                    replace(bundle, **changes)

    def test_direct_construction_rejects_trailing_and_interleaved_wrong_types(self) -> None:
        # Mutation caught: filtering malformed members before formal-grid comparison.
        bundle = self._build()
        report = bundle.report
        malformed_prediction_tuples = (
            report.predictions + (object(),),
            report.predictions[:7] + (object(),) + report.predictions[7:],
        )
        for predictions in malformed_prediction_tuples:
            with self.subTest(record="report", position=len(predictions)):
                with self.assertRaises(ValueError):
                    replace(report, predictions=predictions)

        malformed_run_tuples = (
            bundle.runs + (object(),),
            bundle.runs[:7] + (object(),) + bundle.runs[7:],
        )
        for runs in malformed_run_tuples:
            with self.subTest(record="bundle", position=len(runs)):
                with self.assertRaises(ValueError):
                    replace(bundle, runs=runs)

    def test_report_and_bundle_reject_cross_component_count_mismatches(self) -> None:
        # Mutation caught: independently valid components carrying conflicting counts.
        bundle = self._build()
        report = bundle.report

        wrong_row_count = replace(
            report.predictions[0],
            row_count=report.predictions[0].row_count + 1,
        )
        with self.assertRaises(ValueError):
            replace(
                report,
                predictions=(wrong_row_count,) + report.predictions[1:],
            )

        with self.assertRaises(ValueError):
            replace(
                report,
                manifest_partition_counts=replace(
                    report.manifest_partition_counts,
                    base_test=report.manifest_partition_counts.base_test + 1,
                ),
            )

        longer_first_run = replace(
            bundle.runs[0],
            rows=bundle.runs[0].rows + (bundle.runs[0].rows[0],),
        )
        with self.assertRaises(ValueError):
            replace(bundle, runs=(longer_first_run,) + bundle.runs[1:])

    def test_bundle_rejects_cross_spliced_report_from_different_valid_logits(self) -> None:
        # Mutation caught: comparing only formal identities across bundle components.
        first = self._build()
        for source in self.sources:
            rows = [
                json.loads(line)
                for line in source.path.read_text(encoding="utf-8").splitlines()
            ]
            for row in rows:
                row["logits"] = [0.0, 2.0]
            source.path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        second = self._build()
        self.assertEqual(first.targets, second.targets)
        self.assertNotEqual(first.runs, second.runs)
        self.assertNotEqual(first.report.statistics, second.report.statistics)

        with self.assertRaises(ValueError):
            StatisticsBundle(
                report=first.report,
                targets=second.targets,
                runs=second.runs,
            )

    def test_reordering_inputs_preserves_metrics_but_records_new_provenance(self) -> None:
        # Mutation caught: depending on manifest, source argument, or row order.
        first = self._build()

        self.manifest_rows.reverse()
        self._write_audit_bundle()
        for source in self.sources:
            lines = source.path.read_text(encoding="utf-8").splitlines()
            source.path.write_text(
                "\n".join(reversed(lines)) + "\n", encoding="utf-8"
            )
        self.sources.reverse()

        second = self._build()
        self.assertEqual(second.targets, first.targets)
        self.assertEqual(second.runs, first.runs)
        self.assertEqual(second.report.statistics, first.report.statistics)
        self.assertNotEqual(second.report.predictions[0].run_config_json,
                            first.report.predictions[0].run_config_json)
        self.assertEqual(
            statistics_report_payload(second.report),
            statistics_report_payload(first.report),
        )

    def test_rejects_invalid_source_collections_before_prediction_io(self) -> None:
        # Mutation caught: opening arbitrary paths before validating the source set.
        nonexistent = [
            PredictionSource(self.root / f"absent-{index}.jsonl")
            for index in range(21)
        ]
        cases = (
            (nonexistent[:19], "exactly 20"),
            (nonexistent, "exactly 20"),
            (nonexistent[:10] + [self.root / "wrong"] + nonexistent[11:20],
             "PredictionSource"),
            (nonexistent[:19] + [nonexistent[0]], "distinct"),
        )
        for sources, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(CLEVRReportError) as caught:
                    build_statistics_bundle(self.audit_dir, sources)  # type: ignore[arg-type]
                self.assertIn(expected, str(caught.exception))

        with self.assertRaises(CLEVRReportError) as nonsequence:
            build_statistics_bundle(self.audit_dir, 20)  # type: ignore[arg-type]
        self.assertIn("sequence", str(nonsequence.exception))

    def test_rejects_duplicate_and_missing_formal_identities_after_read(self) -> None:
        # Mutation caught: accepting 20 files without requiring the Cartesian grid.
        source = self.source_by_key[("Q-only", 105)]
        rows = [
            json.loads(line)
            for line in source.path.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["seed"] = 104
        source.path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

        with self.assertRaises(CLEVRReportError) as caught:
            self._build()
        message = str(caught.exception)
        self.assertIn("duplicate", message)
        self.assertIn("missing", message)

    def test_distinct_copy_cannot_relabel_run_but_equal_logits_are_valid(self) -> None:
        # Mutation caught: assigning formal identity from argument position or logits.
        original = self.source_by_key[("G", 101)].path
        copied = self.source_by_key[("B", 101)].path
        copied.write_bytes(original.read_bytes())

        with self.assertRaises(CLEVRReportError):
            self._build()

        rows = [
            json.loads(line)
            for line in copied.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["method"] = "B"
            row["seed"] = 101
        copied.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

        bundle = self._build()
        self.assertEqual(
            tuple((run.method, run.seed) for run in bundle.runs),
            _FORMAL_RUN_KEYS,
        )
        self.assertEqual(
            bundle.runs[0].rows,
            bundle.runs[5].rows,
        )


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.audit_dir = self.root / "audit"
        self.prediction_dir = self.root / "inputs"
        self.audit_dir.mkdir()
        self.prediction_dir.mkdir()
        self._write_inputs(
            self.audit_dir,
            self.prediction_dir,
            reverse_rows=False,
        )

    @staticmethod
    def _write_inputs(
        audit_dir: Path,
        prediction_dir: Path,
        *,
        reverse_rows: bool,
    ) -> None:
        (audit_dir / "vocabulary.json").write_text('{"tokens":["red"]}\n')
        manifest_rows = copy.deepcopy(list(_REPORTING_MANIFEST_ROWS))
        prediction_rows = copy.deepcopy(list(_BALANCED_PREDICTION_ROWS))
        if reverse_rows:
            manifest_rows.reverse()
            prediction_rows.reverse()
        (audit_dir / "image_partitions.json").write_text(
            json.dumps(_IMAGE_PARTITIONS, indent=2) + "\n",
            encoding="utf-8",
        )
        (audit_dir / "selected_questions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        (audit_dir / "audit_report.json").write_text(
            json.dumps(_REPORTING_AUDIT_REPORT, indent=2) + "\n",
            encoding="utf-8",
        )
        for method, seed in _FORMAL_RUN_KEYS:
            rows = [
                {
                    "protocol_version": "PUB-R1.1",
                    "method": method,
                    "seed": seed,
                    "question_index": row["question_index"],
                    "label": row["label"],
                    "logits": row["logits"],
                }
                for row in prediction_rows
            ]
            (prediction_dir / f"input-{method}-{seed}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            _write_run_provenance(audit_dir, prediction_dir / f"input-{method}-{seed}.jsonl", method, seed)

    @staticmethod
    def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    @classmethod
    def _strict_json(cls, text: str) -> object:
        return json.loads(
            text,
            object_pairs_hook=cls._strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonstandard constant {value}")
            ),
        )

    @classmethod
    def _strict_jsonl(cls, path: Path) -> list[dict[str, object]]:
        raw = path.read_bytes()
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise AssertionError(f"{path} must have one terminal newline")
        rows = [cls._strict_json(line) for line in raw.decode("utf-8").splitlines()]
        if any(type(row) is not dict for row in rows):
            raise AssertionError(f"{path} must contain only JSON objects")
        return rows  # type: ignore[return-value]

    @staticmethod
    def _prediction_paths(prediction_dir: Path) -> list[Path]:
        return [
            prediction_dir / f"input-{method}-{seed}.jsonl"
            for method, seed in _FORMAL_RUN_KEYS
        ]

    def _build_bundle(self) -> StatisticsBundle:
        return build_statistics_bundle(
            self.audit_dir,
            tuple(
                PredictionSource(path)
                for path in self._prediction_paths(self.prediction_dir)
            ),
        )

    @staticmethod
    def _lock_path(output_dir: Path) -> Path:
        return output_dir.parent / f".{output_dir.name}.lock"

    @staticmethod
    def _staging_paths(output_dir: Path) -> list[Path]:
        return list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))

    def _assert_complete_output(self, output_dir: Path) -> None:
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {"statistics_report.json", "targets.jsonl", "predictions"},
        )
        self.assertEqual(len(self._strict_jsonl(output_dir / "targets.jsonl")), 10)
        prediction_dir = output_dir / "predictions"
        self.assertEqual(
            {path.name for path in prediction_dir.iterdir()},
            {
                f"{method}_seed_{seed}{suffix}"
                for method, seed in _FORMAL_RUN_KEYS
                for suffix in (".jsonl", ".run_config.json")
            },
        )
        for path in prediction_dir.glob("*.jsonl"):
            self.assertEqual(len(self._strict_jsonl(path)), 8)
        report = self._strict_json(
            (output_dir / "statistics_report.json").read_text(encoding="utf-8")
        )
        self.assertIsInstance(report, dict)
        self.assertEqual(len(report["predictions"]), 20)  # type: ignore[index]

    @classmethod
    def _cli_argv(
        cls,
        audit_dir: Path,
        prediction_dir: Path,
        output_dir: Path,
        *,
        reverse_sources: bool,
    ) -> list[str]:
        paths = cls._prediction_paths(prediction_dir)
        if reverse_sources:
            paths.reverse()
        argv = ["--audit-dir", str(audit_dir)]
        for path in paths:
            argv.extend(("--prediction-file", str(path)))
        argv.extend(("--output-dir", str(output_dir)))
        return argv

    @staticmethod
    def _assert_no_null_or_digest_keys(test: unittest.TestCase, value: object) -> None:
        digest_fragments = ("hash", "digest", "sha", "md5")

        def visit(item: object) -> None:
            test.assertIsNotNone(item)
            if isinstance(item, dict):
                for key, child in item.items():
                    test.assertFalse(
                        any(fragment in key.lower() for fragment in digest_fragments),
                        key,
                    )
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)

    @staticmethod
    def _normalize_source_paths(payload: dict[str, object]) -> None:
        for source in payload["audit_sources"]:  # type: ignore[union-attr]
            source["path"] = f"/<audit>/{Path(source['path']).name}"
        for source in payload["predictions"]:  # type: ignore[union-attr]
            source["source_path"] = f"/<prediction>/{Path(source['source_path']).name}"
            source["run_config_source_path"] = f"/<prediction>/{Path(source['run_config_source_path']).name}"

    def test_cli_publishes_complete_recomputable_canonical_bundle(self) -> None:
        # Mutation caught: incomplete/noncanonical output or CLI/input-order coupling.
        output_dir = self.root / "published"
        self.assertEqual(
            statistics_cli_main(
                self._cli_argv(
                    self.audit_dir,
                    self.prediction_dir,
                    output_dir,
                    reverse_sources=False,
                )
            ),
            0,
        )

        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {"statistics_report.json", "targets.jsonl", "predictions"},
        )
        prediction_output_dir = output_dir / "predictions"
        expected_prediction_names = {
            f"{method}_seed_{seed}{suffix}" for method, seed in _FORMAL_RUN_KEYS
            for suffix in (".jsonl", ".run_config.json")
        }
        self.assertTrue(prediction_output_dir.is_dir())
        self.assertEqual(
            {path.name for path in prediction_output_dir.iterdir()},
            expected_prediction_names,
        )

        target_path = output_dir / "targets.jsonl"
        target_payloads = self._strict_jsonl(target_path)
        self._assert_no_null_or_digest_keys(self, target_payloads)
        self.assertTrue(
            all(
                set(row)
                == {
                    "partition",
                    "official_split",
                    "image_index",
                    "image_filename",
                    "question_index",
                    "relation",
                    "label",
                }
                for row in target_payloads
            )
        )
        self.assertEqual(
            tuple(
                (row["partition"], row["question_index"], row["label"])
                for row in target_payloads
            ),
            (
                ("train", 10, 0),
                ("train", 20, 1),
                ("base_test", 100, 0),
                ("base_test", 101, 1),
                ("base_test", 102, 0),
                ("base_test", 103, 1),
                ("base_test", 104, 0),
                ("base_test", 105, 1),
                ("base_test", 106, 0),
                ("base_test", 107, 1),
            ),
        )

        prediction_payloads: dict[tuple[str, int], list[dict[str, object]]] = {}
        for method, seed in _FORMAL_RUN_KEYS:
            path = prediction_output_dir / f"{method}_seed_{seed}.jsonl"
            rows = self._strict_jsonl(path)
            self._assert_no_null_or_digest_keys(self, rows)
            self.assertTrue(
                all(
                    set(row)
                    == {
                        "protocol_version",
                        "method",
                        "seed",
                        "question_index",
                        "label",
                        "logits",
                    }
                    for row in rows
                )
            )
            self.assertEqual(
                tuple(row["question_index"] for row in rows),
                (100, 101, 102, 103, 104, 105, 106, 107),
            )
            self.assertTrue(
                all(
                    row["protocol_version"] == "PUB-R1.1"
                    and row["method"] == method
                    and row["seed"] == seed
                    for row in rows
                )
            )
            prediction_payloads[(method, seed)] = rows

        report_path = output_dir / "statistics_report.json"
        report_bytes = report_path.read_bytes()
        self.assertTrue(report_bytes.endswith(b"\n"))
        self.assertFalse(report_bytes.endswith(b"\n\n"))
        report_payload = self._strict_json(report_bytes.decode("utf-8"))
        self.assertIsInstance(report_payload, dict)
        report_payload = report_payload  # type: ignore[assignment]
        self._assert_no_null_or_digest_keys(self, report_payload)
        self.assertEqual(
            set(report_payload),
            {
                "protocol_version",
                "report_schema_version",
                "environment",
                "audit_sources",
                "targets_path",
                "manifest_partition_counts",
                "predictions",
                "statistics",
            },
        )
        self.assertEqual(report_payload["targets_path"], "targets.jsonl")
        self.assertEqual(
            tuple(
                (
                    source["path"],
                    source["size_bytes"],
                    source["record_count"],
                )
                for source in report_payload["audit_sources"]
            ),
            (
                (
                    str((self.audit_dir / "image_partitions.json").resolve()),
                    (self.audit_dir / "image_partitions.json").stat().st_size,
                    5,
                ),
                (
                    str((self.audit_dir / "selected_questions.jsonl").resolve()),
                    (self.audit_dir / "selected_questions.jsonl").stat().st_size,
                    12,
                ),
                (
                    str((self.audit_dir / "audit_report.json").resolve()),
                    (self.audit_dir / "audit_report.json").stat().st_size,
                    1,
                ),
            ),
        )
        self.assertEqual(
            tuple(
                (
                    source["method"],
                    source["seed"],
                    source["source_path"],
                    source["size_bytes"],
                    source["row_count"],
                    source["canonical_path"],
                )
                for source in report_payload["predictions"]
            ),
            tuple(
                (
                    method,
                    seed,
                    str(
                        (
                            self.prediction_dir
                            / f"input-{method}-{seed}.jsonl"
                        ).resolve()
                    ),
                    (
                        self.prediction_dir / f"input-{method}-{seed}.jsonl"
                    ).stat().st_size,
                    8,
                    f"predictions/{method}_seed_{seed}.jsonl",
                )
                for method, seed in _FORMAL_RUN_KEYS
            ),
        )

        targets = tuple(
            TargetRow(
                partition=row["partition"],
                official_split=row["official_split"],
                image_index=row["image_index"],
                image_filename=row["image_filename"],
                question_index=row["question_index"],
                relation=row["relation"],
                label=row["label"],
            )
            for row in target_payloads
        )
        runs = tuple(
            RunPredictions(
                method=method,
                seed=seed,
                rows=tuple(
                    PredictionRow(
                        question_index=row["question_index"],
                        label=row["label"],
                        logits=tuple(row["logits"]),
                    )
                    for row in prediction_payloads[(method, seed)]
                ),
            )
            for method, seed in _FORMAL_RUN_KEYS
        )
        recomputed = compute_clevr_statistics(targets, runs)
        published_statistics = report_payload["statistics"]
        self.assertEqual(
            tuple(
                (
                    item["method"],
                    item["seed"],
                    item["metrics"]["question_count"],
                    item["metrics"]["correct_count"],
                    item["metrics"]["accuracy"],
                    item["metrics"]["macro_accuracy"],
                )
                for item in published_statistics["run_metrics"]
            ),
            tuple((method, seed, 8, 4, 0.5, 0.5) for method, seed in _FORMAL_RUN_KEYS),
        )
        for computed, published in zip(
            recomputed.run_metrics,
            published_statistics["run_metrics"],
            strict=True,
        ):
            self.assertEqual(computed.method, published["method"])
            self.assertEqual(computed.seed, published["seed"])
            self.assertAlmostEqual(
                published["metrics"]["cross_entropy"],
                math.log(2.0),
            )
            self.assertEqual(
                computed.metrics.cross_entropy,
                published["metrics"]["cross_entropy"],
            )
            self.assertEqual(computed.metrics.accuracy, published["metrics"]["accuracy"])
            self.assertEqual(
                computed.metrics.macro_accuracy,
                published["metrics"]["macro_accuracy"],
            )
            self.assertEqual(
                tuple(
                    (
                        cell["relation"],
                        cell["label"],
                        cell["count"],
                        cell["correct_count"],
                        cell["accuracy"],
                    )
                    for cell in published["metrics"]["cells"]
                ),
                (
                    ("left", 0, 1, 1, 1.0),
                    ("left", 1, 1, 0, 0.0),
                    ("right", 0, 1, 1, 1.0),
                    ("right", 1, 1, 0, 0.0),
                    ("front", 0, 1, 1, 1.0),
                    ("front", 1, 1, 0, 0.0),
                    ("behind", 0, 1, 1, 1.0),
                    ("behind", 1, 1, 0, 0.0),
                ),
            )
        self.assertEqual(
            tuple(item["values"] for item in published_statistics["method_summaries"]),
            ([0.5] * 5,) * 4,
        )
        self.assertEqual(published_statistics["b_minus_g"]["values"], [0.0] * 5)
        self.assertEqual(published_statistics["m10_minus_g"]["values"], [0.0] * 5)
        self.assertEqual(published_statistics["train_majority"]["cross_entropy"], "NA")
        self.assertEqual(published_statistics["bootstrap"]["replicates"], [0.0] * 2000)
        self.assertEqual(
            list(recomputed.bootstrap.replicates),
            published_statistics["bootstrap"]["replicates"],
        )

        reversed_audit = self.root / "audit-reversed"
        reversed_predictions = self.root / "inputs-reversed"
        reversed_audit.mkdir()
        reversed_predictions.mkdir()
        self._write_inputs(
            reversed_audit,
            reversed_predictions,
            reverse_rows=True,
        )
        reversed_output = self.root / "published-reversed"
        self.assertEqual(
            statistics_cli_main(
                self._cli_argv(
                    reversed_audit,
                    reversed_predictions,
                    reversed_output,
                    reverse_sources=True,
                )
            ),
            0,
        )

        self.assertEqual(
            target_path.read_bytes(),
            (reversed_output / "targets.jsonl").read_bytes(),
        )
        for name in expected_prediction_names:
            if name.endswith(".run_config.json"):
                first_config = json.loads((prediction_output_dir / name).read_bytes())
                second_config = json.loads((reversed_output / "predictions" / name).read_bytes())
                for config, audit in ((first_config, self.audit_dir), (second_config, reversed_audit)):
                    for filename, text in config["data"]["audit_artifacts"].items():
                        self.assertEqual(text, (audit / filename).read_bytes().decode())
                continue
            self.assertEqual(
                (prediction_output_dir / name).read_bytes(),
                (reversed_output / "predictions" / name).read_bytes(),
            )
        reversed_report = self._strict_json(
            (reversed_output / "statistics_report.json").read_text(encoding="utf-8")
        )
        self.assertIsInstance(reversed_report, dict)
        first_normalized = copy.deepcopy(report_payload)
        second_normalized = copy.deepcopy(reversed_report)
        self._normalize_source_paths(first_normalized)
        self._normalize_source_paths(second_normalized)  # type: ignore[arg-type]
        self.assertEqual(
            json.dumps(
                first_normalized,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            ).encode("utf-8") + b"\n",
            json.dumps(
                second_normalized,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            ).encode("utf-8") + b"\n",
        )

    def test_publisher_canonicalizes_valid_bundle_run_rows(self) -> None:
        # Mutation caught: assuming every valid direct bundle has loader-sorted rows.
        bundle = self._build_bundle()
        reversed_bundle = replace(
            bundle,
            runs=tuple(
                replace(run, rows=tuple(reversed(run.rows)))
                for run in bundle.runs
            ),
        )
        output_dir = self.root / "direct-bundle-output"

        clevr_reporting.publish_statistics_bundle(output_dir, reversed_bundle)

        for method, seed in _FORMAL_RUN_KEYS:
            rows = self._strict_jsonl(
                output_dir / "predictions" / f"{method}_seed_{seed}.jsonl"
            )
            self.assertEqual(
                tuple(row["question_index"] for row in rows),
                (100, 101, 102, 103, 104, 105, 106, 107),
            )

    def test_refuses_existing_file_directories_and_symlink_without_changes(self) -> None:
        # Mutation caught: overwriting or cleaning a target this invocation does not own.
        bundle = self._build_bundle()

        existing_file = self.root / "existing-file"
        existing_file.write_bytes(b"keep-file")
        with self.assertRaises(CLEVRReportError):
            clevr_reporting.publish_statistics_bundle(existing_file, bundle)
        self.assertEqual(existing_file.read_bytes(), b"keep-file")
        self.assertFalse(self._lock_path(existing_file).exists())
        self.assertEqual(self._staging_paths(existing_file), [])

        empty_dir = self.root / "existing-empty"
        empty_dir.mkdir()
        with self.assertRaises(CLEVRReportError):
            clevr_reporting.publish_statistics_bundle(empty_dir, bundle)
        self.assertEqual(list(empty_dir.iterdir()), [])
        self.assertFalse(self._lock_path(empty_dir).exists())
        self.assertEqual(self._staging_paths(empty_dir), [])

        nonempty_dir = self.root / "existing-nonempty"
        nonempty_dir.mkdir()
        marker = nonempty_dir / "keep.txt"
        marker.write_bytes(b"keep-directory")
        with self.assertRaises(CLEVRReportError):
            clevr_reporting.publish_statistics_bundle(nonempty_dir, bundle)
        self.assertEqual(marker.read_bytes(), b"keep-directory")
        self.assertFalse(self._lock_path(nonempty_dir).exists())
        self.assertEqual(self._staging_paths(nonempty_dir), [])

        symlink_target = self.root / "symlink-target"
        symlink_target.mkdir()
        symlink_marker = symlink_target / "keep.txt"
        symlink_marker.write_bytes(b"keep-symlink-target")
        output_symlink = self.root / "output-symlink"
        output_symlink.symlink_to(symlink_target, target_is_directory=True)
        with self.assertRaises(CLEVRReportError):
            clevr_reporting.publish_statistics_bundle(output_symlink, bundle)
        self.assertTrue(output_symlink.is_symlink())
        self.assertEqual(symlink_marker.read_bytes(), b"keep-symlink-target")
        self.assertFalse(self._lock_path(output_symlink).exists())
        self.assertEqual(self._staging_paths(output_symlink), [])

    def test_refuses_preexisting_lock_without_creating_output_or_staging(self) -> None:
        # Mutation caught: ignoring another cooperating publisher's ownership marker.
        output_dir = self.root / "locked-output"
        lock_path = self._lock_path(output_dir)
        lock_path.write_bytes(b"owned-by-another-publisher")

        with self.assertRaises(CLEVRReportError):
            clevr_reporting.publish_statistics_bundle(output_dir, self._build_bundle())

        self.assertFalse(output_dir.exists())
        self.assertEqual(lock_path.read_bytes(), b"owned-by-another-publisher")
        self.assertEqual(self._staging_paths(output_dir), [])

    def test_target_appearing_after_lock_is_preserved_and_aborts_commit(self) -> None:
        # Mutation caught: replacing a noncooperating target seen at the final recheck.
        output_dir = self.root / "late-output"
        lock_path = self._lock_path(output_dir)
        marker = output_dir / "keep.txt"
        observed_lock_modes: list[int] = []
        real_writer = clevr_reporting._write_json_lines
        first_write = True

        def create_late_target(path: Path, rows: object) -> None:
            nonlocal first_write
            if first_write:
                first_write = False
                observed_lock_modes.append(stat.S_IMODE(lock_path.stat().st_mode))
                output_dir.mkdir()
                marker.write_bytes(b"late-owner")
            real_writer(path, rows)  # type: ignore[arg-type]

        with mock.patch.object(
            clevr_reporting,
            "_write_json_lines",
            side_effect=create_late_target,
        ):
            with self.assertRaises(CLEVRReportError):
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertEqual(observed_lock_modes, [0o600])
        self.assertEqual(marker.read_bytes(), b"late-owner")
        self.assertFalse(lock_path.exists())
        self.assertEqual(self._staging_paths(output_dir), [])

    def test_precommit_writer_and_rename_failures_clean_owned_artifacts(self) -> None:
        # Mutation caught: leaking partial final/staging/lock before the commit point.
        bundle = self._build_bundle()

        def fail_jsonl(path: Path, rows: object) -> None:
            path.write_text("partial-jsonl\n", encoding="utf-8")
            raise OSError("injected canonical JSONL failure")

        output_dir = self.root / "jsonl-failure"
        with mock.patch.object(
            clevr_reporting,
            "_write_json_lines",
            side_effect=fail_jsonl,
        ):
            with self.assertRaisesRegex(OSError, "canonical JSONL"):
                clevr_reporting.publish_statistics_bundle(output_dir, bundle)
        self.assertFalse(output_dir.exists())
        self.assertFalse(self._lock_path(output_dir).exists())
        self.assertEqual(self._staging_paths(output_dir), [])

        def fail_report(path: Path, payload: object) -> None:
            path.write_text("{\"partial\": true}\n", encoding="utf-8")
            raise OSError("injected report JSON failure")

        output_dir = self.root / "report-failure"
        with mock.patch.object(
            clevr_reporting,
            "_write_json_document",
            side_effect=fail_report,
        ):
            with self.assertRaisesRegex(OSError, "report JSON"):
                clevr_reporting.publish_statistics_bundle(output_dir, bundle)
        self.assertFalse(output_dir.exists())
        self.assertFalse(self._lock_path(output_dir).exists())
        self.assertEqual(self._staging_paths(output_dir), [])

        output_dir = self.root / "rename-failure"
        with mock.patch.object(
            clevr_reporting.os,
            "rename",
            side_effect=OSError("injected rename failure"),
        ):
            with self.assertRaisesRegex(OSError, "rename"):
                clevr_reporting.publish_statistics_bundle(output_dir, bundle)
        self.assertFalse(output_dir.exists())
        self.assertFalse(self._lock_path(output_dir).exists())
        self.assertEqual(self._staging_paths(output_dir), [])

    def test_keyboard_interrupt_before_commit_cleans_owned_artifacts(self) -> None:
        # Mutation caught: limiting the transaction cleanup scope to Exception.
        output_dir = self.root / "keyboard-interrupt"

        def interrupt_jsonl(path: Path, rows: object) -> None:
            path.write_text("partial-before-interrupt\n", encoding="utf-8")
            raise KeyboardInterrupt("injected publication interrupt")

        with mock.patch.object(
            clevr_reporting,
            "_write_json_lines",
            side_effect=interrupt_jsonl,
        ):
            with self.assertRaisesRegex(
                KeyboardInterrupt,
                "publication interrupt",
            ):
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertFalse(output_dir.exists())
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertFalse(self._lock_path(output_dir).exists())

    def test_interrupt_after_real_rename_reports_committed_bundle(self) -> None:
        # Mutation caught: recording commit only after a successful rename return.
        output_dir = self.root / "rename-return-interrupt"
        lock_path = self._lock_path(output_dir)
        real_rename = clevr_reporting.os.rename

        def rename_then_interrupt(source: Path, target: Path) -> None:
            real_rename(source, target)
            raise KeyboardInterrupt("injected after successful rename")

        with mock.patch.object(
            clevr_reporting.os,
            "rename",
            side_effect=rename_then_interrupt,
        ):
            with self.assertRaises(clevr_reporting.ReportPublicationError) as caught:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertIs(caught.exception.committed, True)
        self.assertIn("committed", str(caught.exception))
        self.assertNotIn("not committed", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, KeyboardInterrupt)
        self.assertIn("after successful rename", str(caught.exception.__cause__))
        self._assert_complete_output(output_dir)
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertFalse(lock_path.exists())

    def test_partial_lock_token_interrupt_cleans_identity_owned_lock(self) -> None:
        # Mutation caught: requiring a complete token after token initialization failed.
        output_dir = self.root / "partial-lock-token"
        real_write = clevr_reporting.os.write
        write_count = 0

        def write_part_then_interrupt(fd: int, data: object) -> int:
            nonlocal write_count
            if write_count == 0:
                write_count += 1
                return real_write(fd, memoryview(data)[:7])  # type: ignore[arg-type]
            raise KeyboardInterrupt("injected partial lock-token interrupt")

        caught: BaseException | None = None
        with mock.patch.object(
            clevr_reporting.os,
            "write",
            side_effect=write_part_then_interrupt,
        ):
            try:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )
            except BaseException as exc:
                caught = exc

        self.assertIsInstance(caught, KeyboardInterrupt)
        self.assertIn("partial lock-token interrupt", str(caught))
        self.assertFalse(output_dir.exists())
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertFalse(self._lock_path(output_dir).exists())

    def test_interrupt_after_verification_open_closes_descriptor(self) -> None:
        # Mutation caught: acquiring the verification fd before its cleanup scope.
        output_dir = self.root / "verification-open-interrupt"
        lock_path = self._lock_path(output_dir)
        real_open = clevr_reporting.os.open
        verification_fd: int | None = None
        interrupt_armed = False

        def track_open(path: object, flags: int, mode: int = 0o777) -> int:
            nonlocal interrupt_armed, verification_fd
            opened_fd = real_open(path, flags, mode)  # type: ignore[arg-type]
            if not flags & clevr_reporting.os.O_CREAT and Path(path) == lock_path:
                verification_fd = opened_fd
                interrupt_armed = True
            return opened_fd

        def interrupt_next_unlink_line(
            frame: object,
            event: str,
            argument: object,
        ) -> object:
            del argument
            nonlocal interrupt_armed
            if (
                interrupt_armed
                and event == "line"
                and getattr(frame, "f_code").co_name == "_unlink_owned_lock"
            ):
                interrupt_armed = False
                raise KeyboardInterrupt("injected after verification open")
            return interrupt_next_unlink_line

        caught: BaseException | None = None
        sys.settrace(interrupt_next_unlink_line)
        try:
            with mock.patch.object(
                clevr_reporting.os,
                "open",
                side_effect=track_open,
            ):
                try:
                    clevr_reporting.publish_statistics_bundle(
                        output_dir,
                        self._build_bundle(),
                    )
                except BaseException as exc:
                    caught = exc
        finally:
            sys.settrace(None)

        self.assertIsInstance(caught, clevr_reporting.ReportPublicationError)
        assert isinstance(caught, clevr_reporting.ReportPublicationError)
        self.assertIs(caught.committed, True)
        self.assertIsInstance(caught.__cause__, KeyboardInterrupt)
        self.assertIn("after verification open", str(caught.__cause__))
        self.assertIsNotNone(verification_fd)
        assert verification_fd is not None
        descriptor_was_open = True
        try:
            clevr_reporting.os.fstat(verification_fd)
        except OSError as exc:
            self.assertEqual(exc.errno, errno.EBADF)
            descriptor_was_open = False
        finally:
            if descriptor_was_open:
                clevr_reporting.os.close(verification_fd)

        self.assertFalse(descriptor_was_open)
        self._assert_complete_output(output_dir)
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertTrue(lock_path.is_file())

    def test_short_verification_reads_accept_intact_lock_token(self) -> None:
        # Mutation caught: assuming one os.read fills the requested token buffer.
        output_dir = self.root / "short-verification-reads"
        lock_path = self._lock_path(output_dir)
        real_read = clevr_reporting.os.read

        def short_read(fd: int, byte_count: int) -> bytes:
            return real_read(fd, min(byte_count, 7))

        caught: BaseException | None = None
        with mock.patch.object(
            clevr_reporting.os,
            "read",
            side_effect=short_read,
        ):
            try:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )
            except BaseException as exc:
                caught = exc

        self.assertIsNone(caught)
        self._assert_complete_output(output_dir)
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertFalse(lock_path.exists())

    def test_replaced_lock_before_commit_is_preserved_and_reports_state(self) -> None:
        # Mutation caught: cleanup unlinking a replacement lock solely by pathname.
        output_dir = self.root / "precommit-replaced-lock"
        lock_path = self._lock_path(output_dir)
        replacement = b"replacement-publisher-before-commit"

        def replace_lock_and_fail(path: Path, rows: object) -> None:
            lock_path.unlink()
            lock_path.write_bytes(replacement)
            path.write_text("partial-before-replacement-failure\n", encoding="utf-8")
            raise OSError("injected writer failure after lock replacement")

        with mock.patch.object(
            clevr_reporting,
            "_write_json_lines",
            side_effect=replace_lock_and_fail,
        ):
            with self.assertRaises(clevr_reporting.ReportPublicationError) as caught:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertIs(caught.exception.committed, False)
        self.assertIn("not committed", str(caught.exception))
        self.assertIn("writer failure", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("ownership", str(caught.exception.__cause__))
        self.assertFalse(output_dir.exists())
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertEqual(lock_path.read_bytes(), replacement)

    def test_replaced_lock_after_commit_is_preserved_and_reports_state(self) -> None:
        # Mutation caught: post-rename cleanup unlinking another publisher's lock.
        output_dir = self.root / "postcommit-replaced-lock"
        lock_path = self._lock_path(output_dir)
        replacement = b"replacement-publisher-after-commit"
        real_rename = clevr_reporting.os.rename

        def rename_then_replace_lock(source: Path, target: Path) -> None:
            real_rename(source, target)
            lock_path.unlink()
            lock_path.write_bytes(replacement)

        with mock.patch.object(
            clevr_reporting.os,
            "rename",
            side_effect=rename_then_replace_lock,
        ):
            with self.assertRaises(clevr_reporting.ReportPublicationError) as caught:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertIs(caught.exception.committed, True)
        self.assertIn("committed", str(caught.exception))
        self.assertNotIn("not committed", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("ownership", str(caught.exception.__cause__))
        self._assert_complete_output(output_dir)
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertEqual(lock_path.read_bytes(), replacement)

    def test_lock_unlink_failure_before_commit_reports_not_committed(self) -> None:
        # Mutation caught: losing the primary failure or misreporting pre-rename state.
        output_dir = self.root / "precommit-unlink-failure"
        lock_path = self._lock_path(output_dir)
        real_unlink = Path.unlink

        def fail_lock_unlink(path: Path, *args: object, **kwargs: object) -> None:
            if path == lock_path:
                raise OSError("injected lock unlink failure")
            real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

        def fail_writer(path: Path, rows: object) -> None:
            path.write_text("partial\n", encoding="utf-8")
            raise OSError("injected writer failure")

        with mock.patch.object(Path, "unlink", new=fail_lock_unlink), mock.patch.object(
            clevr_reporting,
            "_write_json_lines",
            side_effect=fail_writer,
        ):
            with self.assertRaises(clevr_reporting.ReportPublicationError) as caught:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertIs(caught.exception.committed, False)
        self.assertIn("not committed", str(caught.exception))
        self.assertIn("writer failure", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("lock unlink failure", str(caught.exception.__cause__))
        self.assertFalse(output_dir.exists())
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertTrue(lock_path.is_file())

    def test_lock_unlink_failure_after_commit_reports_committed_bundle(self) -> None:
        # Mutation caught: rolling back a committed bundle or claiming it did not commit.
        output_dir = self.root / "postcommit-unlink-failure"
        lock_path = self._lock_path(output_dir)
        real_unlink = Path.unlink

        def fail_lock_unlink(path: Path, *args: object, **kwargs: object) -> None:
            if path == lock_path:
                raise OSError("injected lock unlink failure")
            real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(Path, "unlink", new=fail_lock_unlink):
            with self.assertRaises(clevr_reporting.ReportPublicationError) as caught:
                clevr_reporting.publish_statistics_bundle(
                    output_dir,
                    self._build_bundle(),
                )

        self.assertIs(caught.exception.committed, True)
        self.assertIn("committed", str(caught.exception))
        self.assertNotIn("not committed", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("lock unlink failure", str(caught.exception.__cause__))
        self._assert_complete_output(output_dir)
        self.assertEqual(self._staging_paths(output_dir), [])
        self.assertTrue(lock_path.is_file())


if __name__ == "__main__":
    unittest.main()
