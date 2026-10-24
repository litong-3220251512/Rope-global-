"""Data-free tests for the frozen CLEVR PUB-R1.1 statistics contract."""

import math
import unittest
from dataclasses import FrozenInstanceError

from src.clevr_statistics import (
    CellMetrics,
    ClassificationMetrics,
    PredictionRow,
    RunMetrics,
    RunPredictions,
    TargetRow,
    evaluate_run,
)


def _target(
    *,
    partition: str = "base_test",
    official_split: str = "val",
    image_index: int = 3,
    question_index: int = 7,
    relation: str = "left",
    label: int = 0,
) -> TargetRow:
    return TargetRow(
        partition=partition,
        official_split=official_split,
        image_index=image_index,
        image_filename=f"CLEVR_{official_split}_{image_index:06d}.png",
        question_index=question_index,
        relation=relation,
        label=label,
    )


def _prediction(
    *,
    question_index: int = 7,
    label: int = 0,
    logits: tuple[float, float] = (1.0, -1.0),
) -> PredictionRow:
    return PredictionRow(
        question_index=question_index,
        label=label,
        logits=logits,
    )


def _complete_targets() -> tuple[TargetRow, ...]:
    rows = [
        _target(
            partition="train",
            official_split="train",
            image_index=1,
            question_index=1,
            relation="left",
            label=0,
        )
    ]
    question_index = 100
    for relation_index, relation in enumerate(
        ("left", "right", "front", "behind")
    ):
        for label in (0, 1):
            rows.append(
                _target(
                    image_index=10 + relation_index,
                    question_index=question_index,
                    relation=relation,
                    label=label,
                )
            )
            question_index += 1
    return tuple(rows)


def _zero_logit_run(
    targets: tuple[TargetRow, ...], *, method: str = "G", seed: int = 101
) -> RunPredictions:
    return RunPredictions(
        method=method,
        seed=seed,
        rows=tuple(
            _prediction(
                question_index=target.question_index,
                label=target.label,
                logits=(0.0, 0.0),
            )
            for target in targets
            if target.partition == "base_test"
        ),
    )


class RecordContractTests(unittest.TestCase):
    def test_input_records_are_frozen_and_require_immutable_tuple_fields(self) -> None:
        target = _target()
        prediction = _prediction()
        run = RunPredictions(method="G", seed=101, rows=(prediction,))

        with self.assertRaises(FrozenInstanceError):
            target.label = 1
        with self.assertRaises(FrozenInstanceError):
            prediction.logits = (0.0, 0.0)
        with self.assertRaises(FrozenInstanceError):
            run.rows = ()
        with self.assertRaisesRegex(ValueError, "logits.*tuple"):
            _prediction(logits=[1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "rows.*tuple"):
            RunPredictions(method="G", seed=101, rows=[prediction])

    def test_target_rejects_boolean_negative_and_mismatched_identifiers(self) -> None:
        invalid_kwargs = (
            {"image_index": True},
            {"image_index": -1},
            {"question_index": True},
            {"question_index": -1},
            {"image_index": 3, "official_split": "train", "partition": "base_test"},
            {"image_index": 3, "official_split": "val", "partition": "train"},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _target(**kwargs)

        with self.assertRaisesRegex(ValueError, "image_filename"):
            TargetRow(
                partition="base_test",
                official_split="val",
                image_index=3,
                image_filename="CLEVR_val_000004.png",
                question_index=7,
                relation="left",
                label=0,
            )

    def test_target_rejects_unknown_partition_relation_and_nonbinary_label(self) -> None:
        invalid_kwargs = (
            {"partition": "dev"},
            {"official_split": "test"},
            {"relation": "above"},
            {"label": True},
            {"label": -1},
            {"label": 2},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _target(**kwargs)

    def test_prediction_requires_two_finite_nonboolean_real_logits(self) -> None:
        invalid_logits = (
            (),
            (0.0,),
            (0.0, 1.0, 2.0),
            ((0.0,), 1.0),
            (True, 0.0),
            ("0", 1.0),
            (math.nan, 0.0),
            (math.inf, 0.0),
            (-math.inf, 0.0),
        )
        for logits in invalid_logits:
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                _prediction(logits=logits)

        for kwargs in (
            {"question_index": True},
            {"question_index": -1},
            {"label": True},
            {"label": 2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _prediction(**kwargs)

    def test_run_rejects_unknown_method_nonformal_seed_and_nonrows(self) -> None:
        row = _prediction()
        invalid = (
            ("g", 101, (row,)),
            ("G", True, (row,)),
            ("G", 100, (row,)),
            ("G", 101, (object(),)),
        )
        for method, seed, rows in invalid:
            with self.subTest(method=method, seed=seed, rows=rows), self.assertRaises(
                ValueError
            ):
                RunPredictions(method=method, seed=seed, rows=rows)

    def test_output_records_reject_inconsistent_counts_and_ranges(self) -> None:
        with self.assertRaises(ValueError):
            CellMetrics("left", 0, count=1, correct_count=2, accuracy=2.0)
        with self.assertRaises(ValueError):
            CellMetrics("left", 0, count=0, correct_count=0, accuracy=0.0)

        valid_cells = tuple(
            CellMetrics(relation, label, 1, 1, 1.0)
            for relation in ("left", "right", "front", "behind")
            for label in (0, 1)
        )
        with self.assertRaises(ValueError):
            ClassificationMetrics(8, 9, 9 / 8, 0.0, valid_cells, 1.0)
        with self.assertRaises(ValueError):
            ClassificationMetrics(8, 8, 1.0, -0.1, valid_cells, 1.0)
        with self.assertRaisesRegex(ValueError, "cells"):
            ClassificationMetrics(8, 8, 1.0, 0.0, valid_cells[:-1], 1.0)

        metrics = ClassificationMetrics(8, 8, 1.0, 0.0, valid_cells, 1.0)
        with self.assertRaises(ValueError):
            RunMetrics(method="G", seed=100, metrics=metrics)
        with self.assertRaises(ValueError):
            RunMetrics(method="G", seed=101, metrics=object())


class RunMetricTests(unittest.TestCase):
    def test_ties_micro_cross_entropy_and_all_eight_cells_match_literals(self) -> None:
        targets = _complete_targets()

        result = evaluate_run(targets, _zero_logit_run(targets))

        self.assertEqual((result.method, result.seed), ("G", 101))
        self.assertEqual(result.metrics.question_count, 8)
        self.assertEqual(result.metrics.correct_count, 4)
        self.assertEqual(result.metrics.accuracy, 0.5)
        self.assertAlmostEqual(result.metrics.cross_entropy, math.log(2.0))
        self.assertEqual(
            tuple(
                (cell.relation, cell.label, cell.count, cell.correct_count)
                for cell in result.metrics.cells
            ),
            (
                ("left", 0, 1, 1),
                ("left", 1, 1, 0),
                ("right", 0, 1, 1),
                ("right", 1, 1, 0),
                ("front", 0, 1, 1),
                ("front", 1, 1, 0),
                ("behind", 0, 1, 1),
                ("behind", 1, 1, 0),
            ),
        )
        self.assertEqual(
            tuple(cell.accuracy for cell in result.metrics.cells),
            (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        )
        self.assertEqual(result.metrics.macro_accuracy, 0.5)

    def test_empty_cells_are_retained_and_make_macro_na(self) -> None:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
            ),
            _target(question_index=2, relation="behind", label=1),
        )
        run = RunPredictions(
            "B", 102, (_prediction(question_index=2, label=1, logits=(0.0, 1.0)),)
        )

        metrics = evaluate_run(targets, run).metrics

        self.assertEqual(len(metrics.cells), 8)
        self.assertEqual(sum(cell.count for cell in metrics.cells), 1)
        self.assertEqual(metrics.cells[-1], CellMetrics("behind", 1, 1, 1, 1.0))
        self.assertTrue(all(cell.accuracy is None for cell in metrics.cells[:-1]))
        self.assertIsNone(metrics.macro_accuracy)

    def test_cross_entropy_is_stable_for_extreme_equal_and_separated_logits(self) -> None:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
            ),
            _target(image_index=2, question_index=2, label=0),
            _target(image_index=3, question_index=3, label=1),
        )
        run = RunPredictions(
            "M10",
            103,
            (
                _prediction(
                    question_index=2, label=0, logits=(1e308, 1e308)
                ),
                _prediction(
                    question_index=3, label=1, logits=(-1000.0, 1000.0)
                ),
            ),
        )

        metrics = evaluate_run(targets, run).metrics

        self.assertEqual(metrics.correct_count, 2)
        self.assertTrue(math.isfinite(metrics.cross_entropy))
        self.assertAlmostEqual(metrics.cross_entropy, math.log(2.0) / 2)

    def test_target_run_and_row_order_do_not_change_results_or_inputs(self) -> None:
        targets = _complete_targets()
        run = _zero_logit_run(targets)
        original_targets = tuple(targets)
        original_rows = tuple(run.rows)

        forward = evaluate_run(targets, run)
        reversed_result = evaluate_run(
            tuple(reversed(targets)),
            RunPredictions(run.method, run.seed, tuple(reversed(run.rows))),
        )

        self.assertEqual(reversed_result, forward)
        self.assertEqual(targets, original_targets)
        self.assertEqual(run.rows, original_rows)

    def test_duplicate_or_empty_authoritative_targets_are_rejected(self) -> None:
        targets = _complete_targets()
        run = _zero_logit_run(targets)
        cases = (
            (targets + (targets[0],), "duplicate target"),
            (tuple(row for row in targets if row.partition != "train"), "train"),
            (tuple(row for row in targets if row.partition != "base_test"), "base_test"),
        )
        for invalid_targets, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                evaluate_run(invalid_targets, run)

    def test_prediction_ids_must_exactly_cover_targets_once_with_matching_labels(self) -> None:
        targets = _complete_targets()
        run = _zero_logit_run(targets)
        extra = _prediction(question_index=999, label=0, logits=(0.0, 0.0))
        wrong_label = _prediction(
            question_index=run.rows[0].question_index,
            label=1 - run.rows[0].label,
            logits=(0.0, 0.0),
        )
        cases = (
            (RunPredictions("G", 101, run.rows + (run.rows[0],)), "duplicate"),
            (RunPredictions("G", 101, run.rows[:-1]), "missing"),
            (RunPredictions("G", 101, run.rows + (extra,)), "extra"),
            (
                RunPredictions("G", 101, (wrong_label,) + run.rows[1:]),
                "label",
            ),
        )
        for invalid_run, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                evaluate_run(targets, invalid_run)

    def test_unrepresentable_cross_entropy_is_rejected_not_clipped(self) -> None:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
            ),
            _target(image_index=2, question_index=2, label=1),
        )
        run = RunPredictions(
            "Q-only",
            105,
            (
                _prediction(
                    question_index=2,
                    label=1,
                    logits=(1e308, -1e308),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "cross_entropy"):
            evaluate_run(targets, run)


if __name__ == "__main__":
    unittest.main()
