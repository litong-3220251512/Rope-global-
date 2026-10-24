"""Data-free tests for the frozen CLEVR PUB-R1.1 statistics contract."""

import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from src.clevr_statistics import (
    CLEVRStatistics,
    CellMetrics,
    ClassificationMetrics,
    ClusterBootstrapSummary,
    METHODS,
    PairedSummary,
    PredictionRow,
    RunMetrics,
    RunPredictions,
    SEEDS,
    SeedSummary,
    TargetRow,
    compute_clevr_statistics,
    evaluate_run,
    evaluate_train_majority,
    summarize_paired_differences,
    summarize_seed_values,
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


def _targets_with_train_labels(labels: tuple[int, ...]) -> tuple[TargetRow, ...]:
    train = tuple(
        _target(
            partition="train",
            official_split="train",
            image_index=50 + index,
            question_index=50 + index,
            relation=("left", "right", "front", "behind")[index % 4],
            label=label,
        )
        for index, label in enumerate(labels)
    )
    base_test = tuple(
        target for target in _complete_targets() if target.partition == "base_test"
    )
    return train + base_test


def _formal_runs(
    targets: tuple[TargetRow, ...],
    predictions: dict[str, dict[int, int]],
) -> tuple[RunPredictions, ...]:
    base_test = tuple(
        target for target in targets if target.partition == "base_test"
    )
    return tuple(
        RunPredictions(
            method=method,
            seed=seed,
            rows=tuple(
                PredictionRow(
                    question_index=target.question_index,
                    label=target.label,
                    logits=(1.0, 0.0)
                    if predictions[method][target.question_index] == 0
                    else (0.0, 1.0),
                )
                for target in base_test
            ),
        )
        for method in METHODS
        for seed in SEEDS
    )


def _predictions_by_method(
    targets: tuple[TargetRow, ...],
    *,
    g: dict[int, int] | None = None,
    b: dict[int, int] | None = None,
) -> dict[str, dict[int, int]]:
    base_test = tuple(
        target for target in targets if target.partition == "base_test"
    )
    labels = {target.question_index: target.label for target in base_test}
    return {
        "G": labels if g is None else g,
        "B": {key: 1 - label for key, label in labels.items()} if b is None else b,
        "M10": {key: 0 for key in labels},
        "Q-only": {key: 0 for key in labels},
    }


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

    def test_mean_cross_entropy_avoids_intermediate_sum_overflow(self) -> None:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
            ),
            _target(image_index=2, question_index=2, relation="left", label=0),
            _target(image_index=3, question_index=3, relation="right", label=0),
        )
        run = RunPredictions(
            "G",
            101,
            (
                _prediction(
                    question_index=2, label=0, logits=(0.0, 1e308)
                ),
                _prediction(
                    question_index=3, label=0, logits=(0.0, 1e308)
                ),
            ),
        )

        metrics = evaluate_run(targets, run).metrics

        self.assertEqual(metrics.cross_entropy, 1e308)


class MajorityBaselineTests(unittest.TestCase):
    def test_tied_train_labels_predict_no_and_do_not_manufacture_cross_entropy(self) -> None:
        targets = _targets_with_train_labels((0, 0, 1, 1))

        metrics = evaluate_train_majority(targets)

        self.assertEqual(metrics.question_count, 8)
        self.assertEqual(metrics.correct_count, 4)
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertIsNone(metrics.cross_entropy)
        self.assertEqual(
            tuple(cell.accuracy for cell in metrics.cells),
            (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        )
        self.assertEqual(metrics.macro_accuracy, 0.5)

    def test_yes_train_majority_predicts_yes_for_every_base_test_question(self) -> None:
        targets = _targets_with_train_labels((1, 0, 1))

        metrics = evaluate_train_majority(targets)

        self.assertEqual(metrics.correct_count, 4)
        self.assertEqual(
            tuple(cell.accuracy for cell in metrics.cells),
            (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        )

    def test_test_labels_cannot_choose_the_majority_class(self) -> None:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
                label=1,
            ),
            _target(
                partition="train",
                official_split="train",
                image_index=2,
                question_index=2,
                label=1,
            ),
            _target(image_index=3, question_index=3, label=0),
            _target(image_index=4, question_index=4, label=0),
            _target(image_index=5, question_index=5, label=0),
        )

        metrics = evaluate_train_majority(targets)

        self.assertEqual(metrics.correct_count, 0)
        self.assertEqual(metrics.accuracy, 0.0)

    def test_target_reordering_does_not_change_majority_metrics(self) -> None:
        targets = _targets_with_train_labels((1, 0, 1))

        self.assertEqual(
            evaluate_train_majority(targets),
            evaluate_train_majority(tuple(reversed(targets))),
        )


class SeedSummaryTests(unittest.TestCase):
    def test_uses_formal_seed_order_sample_sd_and_fixed_t_interval(self) -> None:
        values = {
            105: 0.02,
            104: 0.01,
            103: 0.0,
            102: -0.01,
            101: -0.02,
        }

        summary = summarize_seed_values("G accuracy", values)

        expected_sd = math.sqrt(0.001 / 4)
        expected_half_width = 2.7764451051977987 * expected_sd / math.sqrt(5)
        self.assertIsInstance(summary, SeedSummary)
        self.assertEqual(summary.name, "G accuracy")
        self.assertEqual(summary.seeds, (101, 102, 103, 104, 105))
        self.assertEqual(summary.values, (-0.02, -0.01, 0.0, 0.01, 0.02))
        self.assertEqual(summary.mean, 0.0)
        self.assertAlmostEqual(summary.sample_sd, expected_sd)
        self.assertAlmostEqual(summary.interval[0], -expected_half_width)
        self.assertAlmostEqual(summary.interval[1], expected_half_width)
        with self.assertRaises(FrozenInstanceError):
            summary.mean = 1.0

    def test_constant_values_have_zero_sd_and_degenerate_interval(self) -> None:
        summary = summarize_seed_values(
            "constant", {seed: 0.25 for seed in (101, 102, 103, 104, 105)}
        )

        self.assertEqual(summary.sample_sd, 0.0)
        self.assertEqual(summary.interval, (0.25, 0.25))

    def test_rejects_nonformal_seed_sets_names_and_nonfinite_values(self) -> None:
        valid = {seed: 0.5 for seed in (101, 102, 103, 104, 105)}
        cases = (
            ("valid", {seed: value for seed, value in valid.items() if seed != 105}),
            ("valid", valid | {106: 0.5}),
            ("valid", valid | {True: 0.5}),
            ("", valid),
            ("   ", valid),
            ("valid", valid | {101: True}),
            ("valid", valid | {101: math.nan}),
            ("valid", valid | {101: math.inf}),
        )
        for name, values in cases:
            with self.subTest(name=name, keys=tuple(values)), self.assertRaises(
                ValueError
            ):
                summarize_seed_values(name, values)

        with self.assertRaises(ValueError):
            summarize_seed_values("valid", [(101, 0.5)])

    def test_paired_summary_subtracts_values_within_each_seed(self) -> None:
        minuend = {
            101: 0.48,
            102: 0.49,
            103: 0.50,
            104: 0.51,
            105: 0.52,
        }
        subtrahend = {seed: 0.5 for seed in (101, 102, 103, 104, 105)}

        summary = summarize_paired_differences(
            "B-G", minuend=minuend, subtrahend=subtrahend
        )

        self.assertIsInstance(summary, PairedSummary)
        self.assertEqual(summary.name, "B-G")
        for observed, expected in zip(
            summary.values, (-0.02, -0.01, 0.0, 0.01, 0.02), strict=True
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(summary.mean, 0.0)

    def test_paired_summary_rejects_mismatched_seed_sets(self) -> None:
        complete = {seed: 0.5 for seed in (101, 102, 103, 104, 105)}
        incomplete = {seed: 0.5 for seed in (101, 102, 103, 104)}

        with self.assertRaisesRegex(ValueError, "seed"):
            summarize_paired_differences("B-G", complete, incomplete)

    def test_direct_summary_records_reject_float_seed_aliases(self) -> None:
        seed_summary = summarize_seed_values(
            "G accuracy", {seed: 0.5 for seed in SEEDS}
        )
        paired_summary = summarize_paired_differences(
            "B-G",
            {seed: 0.5 for seed in SEEDS},
            {seed: 0.5 for seed in SEEDS},
        )
        float_seeds = tuple(float(seed) for seed in SEEDS)

        with self.assertRaisesRegex(ValueError, "seed"):
            SeedSummary(
                seed_summary.name,
                float_seeds,
                seed_summary.values,
                seed_summary.mean,
                seed_summary.sample_sd,
                seed_summary.interval,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            PairedSummary(
                paired_summary.name,
                float_seeds,
                paired_summary.values,
                paired_summary.mean,
                paired_summary.sample_sd,
                paired_summary.interval,
            )


class FormalStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = _targets_with_train_labels((0, 0, 1, 1))
        self.runs = _formal_runs(
            self.targets, _predictions_by_method(self.targets)
        )

    def test_builds_fixed_run_grid_method_summaries_and_paired_differences(self) -> None:
        result = compute_clevr_statistics(self.targets, self.runs)

        self.assertIsInstance(result, CLEVRStatistics)
        self.assertEqual(len(result.run_metrics), 20)
        self.assertEqual(
            tuple((run.method, run.seed) for run in result.run_metrics),
            tuple((method, seed) for method in METHODS for seed in SEEDS),
        )
        self.assertEqual(
            tuple(summary.name for summary in result.method_summaries),
            ("G accuracy", "B accuracy", "M10 accuracy", "Q-only accuracy"),
        )
        self.assertEqual(
            tuple(summary.values for summary in result.method_summaries),
            ((1.0,) * 5, (0.0,) * 5, (0.5,) * 5, (0.5,) * 5),
        )
        self.assertEqual(result.b_minus_g.values, (-1.0,) * 5)
        self.assertEqual(result.m10_minus_g.values, (-0.5,) * 5)
        self.assertEqual(result.train_majority.accuracy, 0.5)
        with self.assertRaises(FrozenInstanceError):
            result.run_metrics = ()

    def test_rejects_missing_duplicate_extra_and_nonrun_grid_entries(self) -> None:
        cases = (
            (self.runs[:-1], "missing"),
            (self.runs + (self.runs[0],), "duplicate"),
            (self.runs + (object(),), "RunPredictions"),
        )
        for runs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                compute_clevr_statistics(self.targets, runs)


class ClusterBootstrapTests(unittest.TestCase):
    maxDiff = 1000

    def _unequal_image_fixture(
        self,
    ) -> tuple[tuple[TargetRow, ...], tuple[RunPredictions, ...]]:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
                label=0,
            ),
            _target(image_index=10, question_index=100, label=0),
            _target(image_index=20, question_index=101, label=0),
            _target(image_index=20, question_index=102, label=0),
            _target(image_index=20, question_index=103, label=0),
        )
        g = {100: 0, 101: 1, 102: 1, 103: 1}
        b = {100: 1, 101: 0, 102: 0, 103: 0}
        runs = _formal_runs(
            targets, _predictions_by_method(targets, g=g, b=b)
        )
        return targets, runs

    def test_common_image_weights_preserve_all_questions_and_match_literals(self) -> None:
        targets, runs = self._unequal_image_fixture()

        result = compute_clevr_statistics(targets, runs)

        self.assertIsInstance(result.bootstrap, ClusterBootstrapSummary)
        self.assertEqual(result.bootstrap.seed, 31001)
        self.assertEqual(result.bootstrap.draws, 2000)
        self.assertEqual(result.bootstrap.quantile_method, "linear")
        self.assertEqual(len(result.bootstrap.replicates), 2000)
        self.assertEqual(result.bootstrap.replicates[:5], (1.0, 0.5, 1.0, 0.5, 1.0))
        expected_interval = np.quantile(
            np.asarray(result.bootstrap.replicates),
            (0.025, 0.975),
            method="linear",
        )
        self.assertEqual(
            result.bootstrap.interval,
            (float(expected_interval[0]), float(expected_interval[1])),
        )
        self.assertEqual(result.b_minus_g.mean, 0.5)
        incorrect_equal_image_difference = ((0.0 + 1.0) / 2) - (
            (1.0 + 0.0) / 2
        )
        self.assertEqual(incorrect_equal_image_difference, 0.0)
        self.assertNotEqual(result.b_minus_g.mean, incorrect_equal_image_difference)

    def test_single_image_makes_every_replicate_equal_the_main_difference(self) -> None:
        targets = (
            _target(
                partition="train",
                official_split="train",
                image_index=1,
                question_index=1,
                label=0,
            ),
            _target(image_index=10, question_index=100, label=0),
            _target(image_index=10, question_index=101, label=1),
        )
        g = {100: 0, 101: 1}
        b = {100: 1, 101: 0}
        runs = _formal_runs(
            targets, _predictions_by_method(targets, g=g, b=b)
        )

        result = compute_clevr_statistics(targets, runs)

        self.assertEqual(result.b_minus_g.mean, -1.0)
        self.assertEqual(set(result.bootstrap.replicates), {-1.0})
        self.assertEqual(result.bootstrap.interval, (-1.0, -1.0))

    def test_all_input_reordering_is_invariant_and_global_rng_is_untouched(self) -> None:
        targets, runs = self._unequal_image_fixture()
        np.random.seed(8675309)
        state_before = np.random.get_state()

        forward = compute_clevr_statistics(targets, runs)
        state_after_forward = np.random.get_state()
        reversed_runs = tuple(
            RunPredictions(run.method, run.seed, tuple(reversed(run.rows)))
            for run in reversed(runs)
        )
        reversed_result = compute_clevr_statistics(
            tuple(reversed(targets)), reversed_runs
        )
        state_after_reversed = np.random.get_state()

        self.assertEqual(reversed_result, forward)
        for observed in (state_after_forward, state_after_reversed):
            self.assertEqual(observed[0], state_before[0])
            self.assertTrue(np.array_equal(observed[1], state_before[1]))
            self.assertEqual(observed[2:], state_before[2:])

    def test_replicates_average_distinct_predictions_from_all_five_seeds(self) -> None:
        targets, _ = self._unequal_image_fixture()
        labels = {100: 0, 101: 0, 102: 0, 103: 0}
        g_by_seed = {
            101: {100: 0, 101: 1, 102: 1, 103: 1},
            102: {100: 0, 101: 0, 102: 0, 103: 0},
            103: {100: 1, 101: 1, 102: 1, 103: 1},
            104: {100: 1, 101: 0, 102: 1, 103: 1},
            105: {100: 0, 101: 0, 102: 0, 103: 1},
        }
        b_by_seed = {
            101: {100: 1, 101: 0, 102: 0, 103: 0},
            102: {100: 1, 101: 1, 102: 1, 103: 1},
            103: {100: 0, 101: 0, 102: 0, 103: 0},
            104: {100: 0, 101: 1, 102: 0, 103: 1},
            105: {100: 1, 101: 1, 102: 1, 103: 0},
        }
        base_test = tuple(
            target for target in targets if target.partition == "base_test"
        )
        runs = tuple(
            RunPredictions(
                method,
                seed,
                tuple(
                    PredictionRow(
                        target.question_index,
                        target.label,
                        (1.0, 0.0)
                        if (
                            g_by_seed[seed]
                            if method == "G"
                            else b_by_seed[seed]
                            if method == "B"
                            else labels
                        )[target.question_index]
                        == 0
                        else (0.0, 1.0),
                    )
                    for target in base_test
                ),
            )
            for method in METHODS
            for seed in SEEDS
        )

        result = compute_clevr_statistics(targets, runs)

        weights = np.random.Generator(np.random.PCG64(31001)).multinomial(
            2, (0.5, 0.5), size=2000
        )
        self.assertEqual(
            {tuple(int(value) for value in row) for row in weights},
            {(0, 2), (1, 1), (2, 0)},
        )
        questions_by_image = ((100,), (101, 102, 103))
        expected_replicates = []
        for image_weights in weights:
            sampled_questions = tuple(
                question_index
                for image_position, question_indices in enumerate(questions_by_image)
                for _ in range(int(image_weights[image_position]))
                for question_index in question_indices
            )
            seed_differences = []
            for seed in SEEDS:
                g_accuracy = sum(
                    g_by_seed[seed][question_index] == labels[question_index]
                    for question_index in sampled_questions
                ) / len(sampled_questions)
                b_accuracy = sum(
                    b_by_seed[seed][question_index] == labels[question_index]
                    for question_index in sampled_questions
                ) / len(sampled_questions)
                seed_differences.append(b_accuracy - g_accuracy)
            expected_replicates.append(sum(seed_differences) / 5)

        for index, (observed, expected) in enumerate(
            zip(result.bootstrap.replicates, expected_replicates, strict=True)
        ):
            self.assertAlmostEqual(
                observed,
                expected,
                msg=f"bootstrap replicate {index} ignored a formal seed",
            )


if __name__ == "__main__":
    unittest.main()
