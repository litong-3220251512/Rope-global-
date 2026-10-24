from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.clevr_reporting import (
    ArtifactEvidence,
    CLEVRReportError,
    PartitionCounts,
    load_audit_targets,
)
from src.clevr_statistics import TargetRow


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


if __name__ == "__main__":
    unittest.main()
