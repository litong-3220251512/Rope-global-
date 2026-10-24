from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

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
    statistics_report_payload,
)
from src.clevr_statistics import PredictionRow, RunPredictions, TargetRow


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
                "official_split": "val",
                "image_index": 12,
                "image_filename": "CLEVR_val_000012.png",
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
        "official_split": "val",
        "image_index": 12,
        "image_filename": "CLEVR_val_000012.png",
        "image_path": "images/val/CLEVR_val_000012.png",
        "question_index": 30,
        "source_position": 0,
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
        "official_split": "val",
        "image_index": 12,
        "image_filename": "CLEVR_val_000012.png",
        "image_path": "images/val/CLEVR_val_000012.png",
        "question_index": 30,
        "source_position": 0,
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
            "official_split": "train",
            "image_index": 12,
            "image_filename": "CLEVR_train_000012.png",
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
                "image_index": 11,
                "image_filename": "CLEVR_val_000011.png",
                "image_path": "images/val/CLEVR_val_000011.png",
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
            source = PredictionSource(path)
            self.sources.append(source)
            self.source_by_key[(method, seed)] = source

    def _write_audit_bundle(self) -> None:
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
        self.assertEqual(report.report_schema_version, 1)
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
            {"report_schema_version": 2},
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

    def test_reordering_all_three_inputs_preserves_bundle_and_payload(self) -> None:
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
        self.assertEqual(second.report, first.report)
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


if __name__ == "__main__":
    unittest.main()
