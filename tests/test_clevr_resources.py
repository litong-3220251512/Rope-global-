"""CPU and fake-clock/event checks; never initialize a real CUDA context."""

import importlib
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import torch

import test_clevr_runner as runner_tests


class ResourceMeterTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("src.clevr_resources")

    def test_cpu_wall_time_and_throughput_never_touch_cuda(self):
        with patch.object(self.module, "perf_counter", side_effect=[10.0, 18.0]), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA touched")), \
                patch.object(torch.cuda, "Event", side_effect=AssertionError("CUDA touched")), \
                patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("CUDA touched")):
            meter = self.module.CLEVRResourceMeter(torch.device("cpu"))
            with meter.compute():
                pass
            record = meter.finish(256)
        self.assertEqual(record.e2e_seconds, 8.0)
        self.assertEqual(record.as_dict()["samples_per_second"], 32.0)
        self.assertIsNone(record.cuda_compute_seconds)
        self.assertIsNone(record.peak_allocated_bytes)

    def test_cuda_regions_sum_ms_exclude_input_gaps_and_synchronize_before_read(self):
        trace = []
        device = torch.device("cuda:2")
        stream = object()
        events = []

        class Event:
            def __init__(self, *, enable_timing):
                self.index = len(events)
                events.append(self)
                assert enable_timing
            def record(self, target):
                assert target is stream
                trace.append(("record", self.index))
            def elapsed_time(self, end):
                assert trace[-1] == "sync"
                assert end.index == self.index + 1
                return 250.0 if self.index == 0 else 750.0

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.module, "perf_counter", side_effect=[10.0, 18.0]))
            sync = stack.enter_context(patch.object(torch.cuda, "synchronize", side_effect=lambda d: trace.append("sync")))
            reset = stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats"))
            stack.enter_context(patch.object(torch.cuda, "Event", Event))
            current = stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=stream))
            peak = stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=1024))
            reserved = stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=2048))
            meter = self.module.CLEVRResourceMeter(device)
            with meter.compute():
                trace.append("forward")
            trace.append("input loading")
            with meter.compute():
                trace.append("optimizer")
            record = meter.finish(256)
        self.assertEqual(trace, ["sync", ("record", 0), "forward", ("record", 1),
                                 "input loading", ("record", 2), "optimizer", ("record", 3), "sync"])
        self.assertEqual(record.cuda_compute_seconds, 1.0)
        self.assertEqual(record.peak_allocated_bytes, 1024)
        self.assertEqual(record.peak_reserved_bytes, 2048)
        reset.assert_called_once_with(device)
        peak.assert_called_once_with(device)
        reserved.assert_called_once_with(device)
        self.assertEqual(sync.call_count, 2)
        self.assertTrue(all(call.args == (device,) for call in current.call_args_list))

    def test_summary_weights_throughput_and_does_not_reuse_old_costs(self):
        with patch.object(self.module, "perf_counter", side_effect=[1.0, 13.0, 20.0, 21.0]):
            summary = self.module.CLEVRResourceSummary(shared_setup_seconds=3.0)
            summary.add("training", self.module.CLEVRResourceUsage(128, 2.0, 0.25, 100, 200))
            summary.add("training", self.module.CLEVRResourceUsage(128, 6.0, 0.75, 300, 400))
            summary.reused_evaluations += 1
            result = summary.snapshot(succeeded=True)
            resumed = self.module.CLEVRResourceSummary().snapshot(succeeded=False)
        self.assertEqual(result["training"]["samples_per_second"], 32.0)
        self.assertEqual(result["cuda_compute_hours"], 1.0 / 3600)
        self.assertEqual(result["peak_allocated_bytes"], 300)
        self.assertEqual(result["arm_wall_seconds"], 12.0)
        self.assertEqual(result["shared_setup_seconds"], 3.0)
        self.assertEqual(resumed["training"]["samples"], 0)
        self.assertIsNone(resumed["cuda_compute_hours"])
        self.assertFalse(resumed["attempt_succeeded"])
        self.assertTrue(resumed["completed_operations_only"])

    def test_zero_duration_is_unavailable_without_magic_floor(self):
        record = self.module.CLEVRResourceUsage(128, 0.0, None, None, None)
        self.assertIsNone(record.as_dict()["samples_per_second"])


class RunnerResourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_tests.RunnerCase()
        self.fixture.setUp()

    def test_update_and_short_evaluation_record_actual_sample_counts(self):
        result = self.fixture.update()
        self.assertEqual(result.resources.samples, 128)
        self.assertIsNone(result.resources.cuda_compute_seconds)
        self.fixture.dataset = runner_tests.TinyDataset(size=17)
        result = self.fixture.evaluate()
        self.assertEqual(result.resources.samples, 17)
        self.assertIsNone(result.resources.peak_allocated_bytes)

    def test_runner_marks_compute_regions_without_including_dataset_loading(self):
        from src.clevr_resources import CLEVRResourceUsage
        active = False
        spans = []
        testcase = self

        class Meter:
            def __init__(self, device):
                pass
            @contextmanager
            def compute(self):
                nonlocal active
                testcase.assertFalse(active)
                active = True
                spans.append("compute")
                try:
                    yield
                finally:
                    active = False
            def finish(self, samples):
                testcase.assertFalse(active)
                return CLEVRResourceUsage(samples, 1.0, None, None, None)

        class Dataset(runner_tests.TinyDataset):
            def __getitem__(self, index):
                testcase.assertFalse(active, "data loading is inside a compute span")
                return super().__getitem__(index)

        self.fixture.dataset = Dataset()
        self.fixture.model.observe = lambda: self.assertTrue(active)
        step = self.fixture.optimizer.step
        def observed_step(*args, **kwargs):
            self.assertTrue(active)
            return step(*args, **kwargs)
        with patch.object(self.fixture.runner, "CLEVRResourceMeter", Meter), \
                patch.object(self.fixture.optimizer, "step", side_effect=observed_step):
            result = self.fixture.update()
        self.assertEqual(len(spans), 9)  # 8 micro-batches + clipping/update
        self.assertEqual(result.resources.samples, 128)


class OrchestrationResourceTests(unittest.TestCase):
    def setUp(self):
        self.f = runner_tests.ProductionTrainingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.module = importlib.import_module("src.clevr_resources")

    def test_pilot_journals_training_and_evaluation_costs_separately(self):
        f = self.f
        def measured_update(**kwargs):
            return replace(f._update(**kwargs),
                           resources=self.module.CLEVRResourceUsage(128, 2.0, None, None, None))
        evaluate = f._evaluate
        def measured_evaluation(**kwargs):
            return replace(evaluate(**kwargs), resources=self.module.CLEVRResourceUsage(
                len(kwargs["dataset"]), 1.0, None, None, None))
        f._evaluate = measured_evaluation
        with f._patches(phase="pilot", methods=("G",), seed=9001, update=measured_update):
            f._train(phase="pilot", seed=9001, methods=("G",), resumes={})
        events = runner_tests._OrchestrationArtifacts.opened[0][0].events
        summary = events[-1][2]["resources"]
        self.assertEqual(summary["training"]["samples"], 800 * 128)
        self.assertEqual(summary["training"]["e2e_seconds"], 1600.0)
        self.assertEqual(summary["evaluation"]["samples"], len(f.dev_rgb))
        self.assertEqual(summary["evaluation"]["operations"], 1)
        self.assertEqual(summary["unmeasured_operations"], 0)
        self.assertTrue(summary["attempt_succeeded"])
        self.assertEqual(events[1][2]["resources"]["samples"], 128)
        self.assertEqual(events[-2][2]["resources"]["samples"], len(f.dev_rgb))

    def test_resuming_completed_predictions_does_not_charge_old_evaluations(self):
        f = self.f
        checkpoint = f._checkpoint(phase="pilot", seed=9001, method="G", update=800)
        f._write_prediction(phase="pilot", seed=9001, method="G", partition="dev",
                            update=800, dataset=f.dev_rgb)
        with f._patches(phase="pilot", methods=("G",), seed=9001,
                        load=f._load_at(800, method="G", seed=9001)):
            f._train(phase="pilot", seed=9001, methods=("G",), resumes={"G": checkpoint})
        events = runner_tests._OrchestrationArtifacts.opened[0][0].events
        summary = events[-1][2]["resources"]
        self.assertEqual(summary["training"]["operations"], 0)
        self.assertEqual(summary["evaluation"]["operations"], 0)
        self.assertEqual(summary["reused_evaluations"], 1)
        self.assertIsNone(events[-2][2]["resources"])
        self.assertFalse(f.eval_calls)

    def test_failure_retains_completed_costs_and_marks_incomplete_attempt(self):
        f = self.f
        calls = 0
        def interrupted_update(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption")
            return replace(f._update(**kwargs),
                           resources=self.module.CLEVRResourceUsage(128, 2.0, None, None, None))
        with f._patches(phase="pilot", methods=("G",), seed=9001, update=interrupted_update):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                f._train(phase="pilot", seed=9001, methods=("G",), resumes={})
        event = runner_tests._OrchestrationArtifacts.opened[0][0].events[-1]
        self.assertEqual(event[1], "failure")
        summary = event[2]["resources"]
        self.assertEqual(summary["training"]["samples"], 128)
        self.assertFalse(summary["attempt_succeeded"])
        self.assertTrue(summary["completed_operations_only"])
