"""Synthetic CPU tests for immutable PUB-R1.1 run evidence."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import torch

if importlib.util.find_spec("src.clevr_run_artifacts") is not None:
    from src import clevr_run_artifacts as artifacts
else:
    artifacts = None


class RunSpecTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(artifacts, "run artifact module must exist")

    def test_exact_phase_seed_budget_and_methods(self):
        triples = [("smoke", 9001, 1), ("pilot", 9001, 800)]
        triples += [("formal", seed, 8000) for seed in range(101, 106)]
        for phase, seed, budget in triples:
            for method in ("G", "B", "M10", "Q-only"):
                with self.subTest(phase=phase, seed=seed, method=method):
                    spec = artifacts.CLEVRRunSpec(phase, method, seed)
                    self.assertEqual(spec.target_updates, budget)
                    self.assertEqual(spec.evidence_role,
                                     "formal" if phase == "formal" else "engineering-only")

    def test_invalid_scalars_aliases_and_confirmation_are_rejected(self):
        cases = [(True, "G", 9001), ("formal", "G", True),
                 ("formal", "G", np.int64(101)), (np.str_("formal"), "G", 101),
                 ("formal", np.str_("G"), 101), ("formal", "g", 101),
                 ("confirmation", "G", 101), ("formal", "confirmation", 101),
                 ("formal", "G", 9001), ("pilot", "G", 101),
                 ("smoke", "G", 101), ("other", "G", 101)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(artifacts.CLEVRRunArtifactError):
                artifacts.CLEVRRunSpec(*args)

    def test_no_configurable_budget_or_mutable_spec(self):
        spec = artifacts.CLEVRRunSpec("formal", "B", 101)
        self.assertEqual([f.name for f in fields(spec)], ["phase", "method", "seed"])
        self.assertFalse(hasattr(spec, "__dict__"))
        with self.assertRaises(TypeError):
            artifacts.CLEVRRunSpec("formal", "B", 101, target_updates=7999)
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            spec.target_updates = 7999
        with self.assertRaises(FrozenInstanceError):
            spec.seed = 102


class ArtifactFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs"
        self.spec = artifacts.CLEVRRunSpec("formal", "G", 101)
        self.config = {"batch_size": 128, "nested": {"flags": [True, None, 0.5]}}

    def open_run(self, *, spec=None, resume=False, config=None):
        run = artifacts.CLEVRRunArtifacts.open(
            output_root=self.root, spec=spec or self.spec,
            expanded_config=self.config if config is None else config, resume=resume)
        self.addCleanup(run.close)
        return run

    def run_dir(self):
        return self.root / "formal" / "seed_101" / "G"


class ConfigTests(ArtifactFixture):
    def test_fresh_and_resume_preserve_complete_private_config(self):
        run = self.open_run()
        path = self.run_dir() / "run_config.json"
        self.assertEqual(run.run_dir, self.run_dir())
        original = path.read_bytes()
        payload = json.loads(original)
        self.assertEqual(payload["batch_size"], 128)
        self.assertEqual(payload["phase"], "formal")
        self.assertEqual(payload["target_updates"], 8000)
        self.assertEqual(payload["evidence_role"], "formal")
        self.assertEqual(payload["protocol_version"], "PUB-R1.1")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.config["nested"]["flags"][0] = False
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run(resume=True)
        self.config["nested"]["flags"][0] = True
        self.open_run(resume=True)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run()
        self.assertEqual(path.read_bytes(), original)

    def test_missing_resume_target_config_and_substitutions_are_rejected(self):
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run(resume=True)
        self.assertFalse(self.root.exists())
        target = self.run_dir()
        target.parent.mkdir(parents=True)
        target.write_text("old evidence")
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run(resume=True)
        target.unlink()
        elsewhere = Path(self.temp.name) / "elsewhere"
        elsewhere.mkdir()
        target.symlink_to(elsewhere, target_is_directory=True)
        for resume in (False, True):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                self.open_run(resume=resume)
        self.assertEqual(list(elsewhere.iterdir()), [])
        target.unlink()
        target.mkdir()
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run(resume=True)
        (target / "run_config.json").symlink_to(elsewhere / "config")
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run(resume=True)

    def test_strict_config_and_frozen_field_conflicts_before_publication(self):
        bad = [{"x": float("nan")}, {"x": float("inf")}, {1: "key"},
               {"x": np.int64(1)}, {"x": np.float64(1)}, {"x": (1, 2)},
               {"x": object()}, {"target_updates": 7999}, {"seed": True},
               {"phase": "confirmation"}, {"method": "B"}]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(artifacts.CLEVRRunArtifactError):
                self.open_run(config=config)
            self.assertFalse(self.root.exists())
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            self.open_run(resume=1)

    def test_config_type_sensitive_resume_and_invalid_json(self):
        self.open_run(config={"value": 1})
        path = self.run_dir() / "run_config.json"
        for config in ({"value": True}, {"value": 1.0}):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                self.open_run(resume=True, config=config)
        for data in (b'{"x":NaN}', b'{"x":1,"x":2}', b'[]', b'broken'):
            path.write_bytes(data)
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                self.open_run(resume=True)

    def test_link_faults_never_expose_partial_config_or_overwrite_winner(self):
        real_link = os.link
        for mode in ("before", "after", "winner"):
            with self.subTest(mode=mode):
                self.root = Path(self.temp.name) / mode
                path = self.run_dir() / "run_config.json"
                def link(source, target):
                    self.assertEqual(Path(source).parent, path.parent)
                    self.assertEqual(stat.S_IMODE(Path(source).stat().st_mode), 0o600)
                    json.loads(Path(source).read_bytes())
                    if mode == "winner":
                        Path(target).write_bytes(b"winner")
                        raise FileExistsError("winner")
                    if mode == "after":
                        real_link(source, target)
                    raise OSError("link fault")
                with mock.patch.object(artifacts.os, "link", side_effect=link):
                    with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                        self.open_run()
                self.assertIs(caught.exception.committed, mode == "after")
                self.assertEqual(list(path.parent.glob(".*staging-*")), [])
                if mode == "before":
                    self.assertFalse(path.exists())
                elif mode == "winner":
                    self.assertEqual(path.read_bytes(), b"winner")
                else:
                    self.assertEqual(json.loads(path.read_bytes())["seed"], 101)


class JournalTests(ArtifactFixture):
    def test_monotonic_attempts_strict_lines_flush_and_idempotent_close(self):
        first = self.open_run()
        self.assertTrue(hasattr(first, "append_event"))
        journal = self.run_dir() / "attempts" / "attempt_0001.jsonl"
        self.assertEqual(first.attempt, 1)
        self.assertEqual(first.journal_path, journal)
        self.assertEqual(stat.S_IMODE(journal.stat().st_mode), 0o600)
        real_flush = first._journal.flush
        flushed = []
        def flush():
            real_flush()
            flushed.append(journal.read_bytes())
        with mock.patch.object(first._journal, "flush", side_effect=flush):
            first.append_event(completed_updates=0, event="started", details={"resume": False})
            first.append_event(completed_updates=1, event="update", details={"loss": 0.5})
        self.assertEqual(len(flushed), 2)
        lines = journal.read_bytes().splitlines()
        self.assertEqual(len(lines), 2)
        first_row = json.loads(lines[0])
        self.assertEqual(first_row, {"protocol_version": "PUB-R1.1", "phase": "formal",
                                    "method": "G", "seed": 101, "attempt": 1,
                                    "completed_updates": 0, "event": "started",
                                    "details": {"resume": False}})
        for line in lines:
            self.assertNotIn(b" ", line)
            self.assertEqual(set(json.loads(line)), set(first_row))
        first.close()
        first.close()
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            first.append_event(completed_updates=1, event="late", details={})
        second = self.open_run(resume=True)
        self.assertEqual(second.attempt, 2)
        self.assertEqual(second.journal_path.name, "attempt_0002.jsonl")
        second.append_event(completed_updates=0, event="resumed", details={})
        self.assertEqual(journal.read_bytes().splitlines(), lines)
        second.close()
        second.journal_path.unlink()  # Gap does not allow reuse of an earlier file.
        (journal.parent / "attempt_0009.jsonl").touch()
        third = self.open_run(resume=True)
        self.assertEqual(third.attempt, 10)

    def test_concurrent_opens_have_exclusive_attempt_files(self):
        initial = self.open_run()
        self.assertTrue(hasattr(initial, "attempt"))
        barrier = threading.Barrier(2)
        def worker(_):
            barrier.wait()
            run = artifacts.CLEVRRunArtifacts.open(output_root=self.root, spec=self.spec,
                                                   expanded_config=self.config, resume=True)
            try:
                run.append_event(completed_updates=0, event="started", details={})
                return run.attempt, run.journal_path
            finally:
                run.close()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(worker, range(2)))
        self.assertEqual(sorted(attempt for attempt, _ in results), [2, 3])
        for attempt, path in results:
            self.assertEqual(json.loads(path.read_text())["attempt"], attempt)

    def test_invalid_events_do_not_write_any_bytes(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "append_event"))
        cases = [(True, "event", {}), (np.int64(1), "event", {}),
                 (-1, "event", {}), (8001, "event", {}), (0, "", {}),
                 (0, np.str_("event"), {}), (0, "event", []),
                 (0, "event", {"bad": float("nan")})]
        for update, event, details in cases:
            with self.subTest(update=update, event=event), self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.append_event(completed_updates=update, event=event, details=details)
            self.assertEqual(run.journal_path.read_bytes(), b"")

    def test_paths_validate_update_and_do_not_create_checkpoint_files(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "checkpoint_path"))
        for update, filename in ((0, "update_0000.pt"), (1, "update_0001.pt"),
                                 (8000, "update_8000.pt")):
            path = run.checkpoint_path(update)
            self.assertEqual(path, self.run_dir() / "checkpoints" / filename)
            self.assertTrue(path.parent.is_dir())
            self.assertFalse(path.exists())
        for bad in (True, np.int64(1), -1, 8001, 1.0):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.checkpoint_path(bad)
        for partition in ("train", "dev"):
            path = run.prediction_path(partition, 8000)
            self.assertEqual(path, self.run_dir() / "predictions" / partition / "update_8000.jsonl")
            self.assertFalse(path.exists())
            path.touch()
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.prediction_path(partition, 8000)
        for partition in ("base_test", "confirmation", "other", np.str_("dev")):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.prediction_path(partition, 8000)
        run.close()
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            run.checkpoint_path(0)


class CommonStateTests(ArtifactFixture):
    def state(self):
        return OrderedDict([("weight", torch.tensor([[1.0, 2.0]], requires_grad=True)),
                            ("counter", torch.tensor(3, dtype=torch.int64))])

    def test_restricted_ordered_cpu_snapshot_and_cross_method_verification(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "ensure_common_state"))
        source = self.state()
        real_save = torch.save
        snapshots = []
        def save(payload, output):
            snapshot = payload["common_state"]
            self.assertEqual(list(snapshot), ["weight", "counter"])
            for key, value in snapshot.items():
                self.assertEqual(value.device.type, "cpu")
                self.assertFalse(value.requires_grad)
                self.assertNotEqual(value.data_ptr(), source[key].data_ptr())
            snapshots.append(snapshot)
            with torch.no_grad():
                source["weight"].add_(100)
            return real_save(payload, output)
        with mock.patch.object(artifacts.torch, "save", side_effect=save):
            path = run.ensure_common_state(source)
        self.assertEqual(path, self.run_dir().parent / "common_state.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload), {"protocol_version", "run_seed", "environment", "common_state"})
        self.assertEqual(payload["protocol_version"], "PUB-R1.1")
        self.assertEqual(payload["run_seed"], 101)
        self.assertEqual(payload["environment"], {
            "python": platform.python_version(), "numpy": str(np.__version__),
            "torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda)})
        self.assertTrue(torch.equal(payload["common_state"]["weight"], torch.tensor([[1.0, 2.0]])))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        original = path.read_bytes()
        other = self.open_run(spec=artifacts.CLEVRRunSpec("formal", "B", 101))
        real_load = torch.load
        with mock.patch.object(artifacts.torch, "load", wraps=real_load) as loader:
            self.assertEqual(other.ensure_common_state(self.state()), path)
        self.assertTrue(all(call.kwargs.get("weights_only") is True for call in loader.call_args_list))
        self.assertEqual(path.read_bytes(), original)

    def test_mismatched_order_shape_dtype_values_and_bad_inputs_do_not_overwrite(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "ensure_common_state"))
        path = run.ensure_common_state(self.state())
        original = path.read_bytes()
        states = [OrderedDict(reversed(list(self.state().items()))),
                  {"weight": torch.tensor([1.0, 2.0]), "counter": torch.tensor(3)},
                  {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float64), "counter": torch.tensor(3)},
                  {"weight": torch.tensor([[1.0, 4.0]]), "counter": torch.tensor(3)},
                  {"weight": torch.tensor([[1.0, 2.0]])}, {},
                  {True: torch.tensor(1)}, {"bad": 1}, {"bad": torch.empty(2, device="meta")}]
        for source in states:
            with self.subTest(keys=list(source)), self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.ensure_common_state(source)
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list((self.run_dir() / "checkpoints").glob("*")), [])

    def test_malformed_untrusted_and_environment_mismatch_are_hard_errors(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "ensure_common_state"))
        path = run.ensure_common_state(self.state())
        good = torch.load(path, weights_only=True)
        cases = [{}, {**good, "protocol_version": "other"}, {**good, "run_seed": True},
                 {**good, "run_seed": 102}, {**good, "environment": {"python": "different"}},
                 {**good, "common_state": {"weight": "untrusted"}},
                 {**good, "extra": 1}, {**good, "untrusted": Path("untrusted")}]
        for payload in cases:
            with self.subTest(keys=list(payload)):
                torch.save(payload, path)
                before = path.read_bytes()
                with self.assertRaises(artifacts.CLEVRRunArtifactError):
                    run.ensure_common_state(self.state())
                self.assertEqual(path.read_bytes(), before)
        path.write_bytes(b"not a torch payload")
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            run.ensure_common_state(self.state())

    def test_concurrent_publishers_accept_only_an_exact_winner(self):
        for equal in (True, False):
            with self.subTest(equal=equal):
                self.root = Path(self.temp.name) / str(equal)
                first = self.open_run()
                self.assertTrue(hasattr(first, "ensure_common_state"))
                second = self.open_run(spec=artifacts.CLEVRRunSpec("formal", "B", 101))
                barrier = threading.Barrier(2)
                real_link = os.link
                def link(source, target):
                    if Path(target).name == "common_state.pt":
                        barrier.wait(timeout=10)
                    real_link(source, target)
                states = [self.state(), self.state()]
                if not equal:
                    states[1]["counter"] += 1
                def worker(index):
                    try:
                        return (first, second)[index].ensure_common_state(states[index])
                    except artifacts.CLEVRRunArtifactError as exc:
                        return exc
                with mock.patch.object(artifacts.os, "link", side_effect=link):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        results = list(executor.map(worker, range(2)))
                errors = [item for item in results if isinstance(item, Exception)]
                self.assertEqual(len(errors), 0 if equal else 1)
                self.assertEqual(list(self.run_dir().parent.glob(".*staging-*")), [])

    def test_writer_and_link_faults_leave_no_partial_common_state(self):
        for fault in ("save", "link"):
            with self.subTest(fault=fault):
                self.root = Path(self.temp.name) / fault
                run = self.open_run()
                self.assertTrue(hasattr(run, "ensure_common_state"))
                owner, name = (artifacts.torch, "save") if fault == "save" else (artifacts.os, "link")
                with mock.patch.object(owner, name, side_effect=OSError(fault)):
                    with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                        run.ensure_common_state(self.state())
                self.assertIs(caught.exception.committed, False)
                self.assertFalse((self.run_dir().parent / "common_state.pt").exists())
                self.assertEqual(list(self.run_dir().parent.glob(".*staging-*")), [])


class _FaultFile:
    """Delegate real I/O, injecting one explicit publication boundary fault."""
    def __init__(self, output, fault):
        self.output = output
        self.fault = fault

    @property
    def closed(self):
        return self.output.closed

    def write(self, value):
        if self.fault == "write":
            self.output.write(value[:3])
            raise OSError("partial write")
        return self.output.write(value)

    def flush(self):
        if self.fault == "flush":
            raise OSError("flush")
        return self.output.flush()

    def close(self):
        self.output.close()
        if self.fault == "close":
            raise OSError("close")


class PredictionTests(ArtifactFixture):
    def rows(self):
        return [artifacts.CLEVRPredictionRecord(20, 1, (-0.25, 0.5)),
                artifacts.CLEVRPredictionRecord(0, 0, (1.0, 1.0))]

    def test_record_is_frozen_strict_binary_and_finite(self):
        self.assertTrue(hasattr(artifacts, "CLEVRPredictionRecord"))
        row = self.rows()[0]
        self.assertFalse(hasattr(row, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            row.label = 0
        bad = [(True, 0, (1.0, 2.0)), (np.int64(2), 0, (1.0, 2.0)),
               (-1, 0, (1.0, 2.0)), (1, True, (1.0, 2.0)), (1, 2, (1.0, 2.0)),
               (1, np.int64(1), (1.0, 2.0)), (1, 0, [1.0, 2.0]),
               (1, 0, (1.0,)), (1, 0, (1.0, 2.0, 3.0)), (1, 0, (1, 2.0)),
               (1, 0, (np.float64(1), 2.0)), (1, 0, (float("nan"), 2.0)),
               (1, 0, (1.0, float("inf")))]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(artifacts.CLEVRRunArtifactError):
                artifacts.CLEVRPredictionRecord(*args)

    def test_sorted_exact_six_key_rows_canonical_filename_and_no_overwrite(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "publish_predictions"))
        rows = self.rows()
        path = run.publish_predictions(partition="base_test", completed_updates=8000, rows=rows)
        self.assertEqual(path, self.run_dir() / "predictions" / "base_test" / "G_seed_101.jsonl")
        self.assertEqual([row.question_index for row in rows], [20, 0])
        parsed = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(parsed, [
            {"protocol_version": "PUB-R1.1", "method": "G", "seed": 101,
             "question_index": 0, "label": 0, "logits": [1.0, 1.0]},
            {"protocol_version": "PUB-R1.1", "method": "G", "seed": 101,
             "question_index": 20, "label": 1, "logits": [-0.25, 0.5]}])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        before = path.read_bytes()
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            run.publish_predictions(partition="base_test", completed_updates=8000, rows=rows)
        self.assertEqual(path.read_bytes(), before)

    def test_protocol_evaluation_boundaries_and_confirmation_gate(self):
        self.assertTrue(hasattr(artifacts, "CLEVRPredictionRecord"))
        for phase, seed, allowed in (
                ("smoke", 9001, []),
                ("pilot", 9001, [("dev", 800)]),
                ("formal", 101, [("dev", k) for k in range(1000, 8001, 1000)]
                 + [("train", 8000), ("base_test", 8000)])):
            self.root = Path(self.temp.name) / phase
            run = self.open_run(spec=artifacts.CLEVRRunSpec(phase, "G", seed))
            for partition, update in allowed:
                path = run.publish_predictions(partition=partition, completed_updates=update, rows=self.rows())
                self.assertTrue(path.is_file())
                if partition != "base_test":
                    self.assertEqual(path.name, f"update_{update:04d}.jsonl")
            for partition, update in (("confirmation", 8000), ("other", 8000),
                                      ("dev", 0), ("dev", 1), ("dev", 7999),
                                      ("train", 800), ("base_test", 800),
                                      ("base_test", 7999), ("dev", True),
                                      (np.str_("dev"), 1000)):
                with self.subTest(phase=phase, partition=partition, update=update):
                    with self.assertRaises(artifacts.CLEVRRunArtifactError):
                        run.publish_predictions(partition=partition, completed_updates=update, rows=self.rows())

    def test_duplicate_ids_empty_bad_rows_and_serialization_never_publish(self):
        run = self.open_run()
        self.assertTrue(hasattr(run, "publish_predictions"))
        rows = self.rows()
        for bad in ([], [rows[0], rows[0]], [{}], "rows", iter(rows)):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.publish_predictions(partition="dev", completed_updates=1000, rows=bad)
        with mock.patch.object(artifacts.json, "dumps", side_effect=ValueError("serialization")):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.publish_predictions(partition="dev", completed_updates=1000, rows=rows)
        self.assertFalse((self.run_dir() / "predictions").exists())

    def test_write_flush_close_link_and_cleanup_failures_report_commit_status(self):
        for fault in ("write", "flush", "close", "link", "link_after", "unlink", "unlink_after"):
            with self.subTest(fault=fault):
                self.root = Path(self.temp.name) / fault
                run = self.open_run()
                self.assertTrue(hasattr(run, "publish_predictions"))
                target = self.run_dir() / "predictions" / "dev" / "update_1000.jsonl"
                real_fdopen, real_link, real_unlink = os.fdopen, os.link, os.unlink
                def fdopen(fd, mode):
                    return _FaultFile(real_fdopen(fd, mode), fault)
                def link(source, target):
                    if fault == "link_after":
                        real_link(source, target)
                    if fault in ("link", "link_after"):
                        raise OSError(fault)
                    real_link(source, target)
                def unlink(path):
                    if fault == "unlink_after":
                        real_unlink(path)
                    if fault in ("unlink", "unlink_after"):
                        raise OSError(fault)
                    real_unlink(path)
                with mock.patch.object(artifacts.os, "fdopen", side_effect=fdopen), \
                     mock.patch.object(artifacts.os, "link", side_effect=link), \
                     mock.patch.object(artifacts.os, "unlink", side_effect=unlink):
                    with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                        run.publish_predictions(partition="dev", completed_updates=1000, rows=self.rows())
                committed = fault in ("link_after", "unlink", "unlink_after")
                self.assertIs(caught.exception.committed, committed)
                self.assertEqual(target.exists(), committed)
                stages = list(target.parent.glob(".*staging-*"))
                self.assertEqual(len(stages), 1 if fault == "unlink" else 0)
                if committed:
                    self.assertEqual(len(target.read_bytes().splitlines()), 2)
                    for line in target.read_bytes().splitlines():
                        self.assertEqual(len(json.loads(line)), 6)


class PublicationEdgeTests(ArtifactFixture):
    def test_short_write_never_commits_config_or_predictions(self):
        real_fdopen = os.fdopen
        def fdopen(fd, mode):
            output = _FaultFile(real_fdopen(fd, mode), None)
            real_write = output.write
            output.write = lambda content: real_write(content[:3])
            return output
        for kind in ("config", "predictions"):
            with self.subTest(kind=kind):
                self.root = Path(self.temp.name) / kind
                if kind == "predictions":
                    run = self.open_run()
                    target = self.run_dir() / "predictions" / "dev" / "update_1000.jsonl"
                else:
                    target = self.run_dir() / "run_config.json"
                with mock.patch.object(artifacts.os, "fdopen", side_effect=fdopen):
                    with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                        if kind == "config":
                            self.open_run()
                        else:
                            run.publish_predictions(partition="dev", completed_updates=1000,
                                                    rows=[artifacts.CLEVRPredictionRecord(0, 0, (1.0, 0.0))])
                self.assertIs(caught.exception.committed, False)
                self.assertFalse(target.exists())
                self.assertEqual(list(target.parent.glob(".*staging-*")), [])

    def test_short_journal_write_poisoned_attempt_cannot_append_again(self):
        run = self.open_run()
        real_write = run._journal.write
        with mock.patch.object(run._journal, "write", side_effect=lambda line: real_write(line[:3])):
            with self.assertRaises(artifacts.CLEVRRunArtifactError):
                run.append_event(completed_updates=0, event="start", details={})
        with self.assertRaises(artifacts.CLEVRRunArtifactError):
            run.append_event(completed_updates=0, event="retry", details={})
        run.close()
        self.assertEqual(run.journal_path.read_bytes(), b'{"a')

    def test_foreign_staging_is_never_linked_or_deleted(self):
        run = self.open_run()
        real_fdopen = os.fdopen
        foreign = []
        def fdopen(fd, mode):
            output = _FaultFile(real_fdopen(fd, mode), None)
            close = output.close
            def replace_after_close():
                close()
                staging = next((self.run_dir() / "predictions" / "dev").glob(".*staging-*"))
                # Keep the owned inode alive to rule out inode-number reuse.
                os.link(staging, staging.parent / "owned-retained")
                staging.unlink()
                staging.write_bytes(b"foreign")
                foreign.append(staging)
            output.close = replace_after_close
            return output
        with mock.patch.object(artifacts.os, "fdopen", side_effect=fdopen):
            with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                run.publish_predictions(partition="dev", completed_updates=1000,
                                        rows=[artifacts.CLEVRPredictionRecord(0, 0, (1.0, 0.0))])
        self.assertIs(caught.exception.committed, False)
        self.assertEqual(foreign[0].read_bytes(), b"foreign")
        self.assertTrue(any("ownership" in note for note in caught.exception.__notes__))
        self.assertFalse((foreign[0].parent / "update_1000.jsonl").exists())

    def test_unreadable_target_after_link_error_reports_unknown_commit(self):
        run = self.open_run()
        real_link, real_lstat = os.link, Path.lstat
        target = self.run_dir() / "predictions" / "dev" / "update_1000.jsonl"
        linked = False
        def link(source, destination):
            nonlocal linked
            real_link(source, destination)
            linked = True
            raise OSError("link returned an error after commit")
        def lstat(path, *args, **kwargs):
            if linked and path == target:
                raise PermissionError("target unavailable for reconciliation")
            return real_lstat(path, *args, **kwargs)
        with mock.patch.object(artifacts.os, "link", side_effect=link), \
             mock.patch.object(Path, "lstat", lstat):
            with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                run.publish_predictions(partition="dev", completed_updates=1000,
                                        rows=[artifacts.CLEVRPredictionRecord(0, 0, (1.0, 0.0))])
        self.assertIsNone(caught.exception.committed)
        self.assertTrue(target.is_file())
        self.assertEqual(list(target.parent.glob(".*staging-*")), [])
        self.assertTrue(any("reconciliation" in note for note in caught.exception.__notes__))

    def test_setup_faults_recover_descriptors_and_owned_staging(self):
        for fault in ("fstat", "fchmod", "fdopen"):
            with self.subTest(fault=fault):
                self.root = Path(self.temp.name) / fault
                run = self.open_run()
                real = getattr(os, fault)
                calls = 0
                def fail_once(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise OSError(fault)
                    return real(*args, **kwargs)
                with mock.patch.object(artifacts.os, fault, side_effect=fail_once):
                    with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                        run.ensure_common_state({"weight": torch.tensor([1.0])})
                self.assertIs(caught.exception.committed, False)
                self.assertFalse((self.run_dir().parent / "common_state.pt").exists())
                self.assertEqual(list(self.run_dir().parent.glob(".*staging-*")), [])


class CommonShortWriteTests(ArtifactFixture):
    def test_torch_writer_short_return_cannot_publish_a_corrupt_archive(self):
        run = self.open_run()
        real_fdopen = os.fdopen
        def fdopen(fd, mode):
            output = _FaultFile(real_fdopen(fd, mode), None)
            real_write = output.write
            output.write = lambda content: real_write(content[:3])
            return output
        with mock.patch.object(artifacts.os, "fdopen", side_effect=fdopen):
            with self.assertRaises(artifacts.CLEVRRunArtifactError) as caught:
                run.ensure_common_state({"weight": torch.tensor([1.0])})
        self.assertIs(caught.exception.committed, False)
        path = self.run_dir().parent / "common_state.pt"
        self.assertFalse(path.exists(), "a truncated torch archive must never reach the final path")
        self.assertEqual(list(path.parent.glob(".*staging-*")), [])
