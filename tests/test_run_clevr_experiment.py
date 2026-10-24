"""CPU-only parser and dispatch tests for the explicit CLEVR entry point."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class CLEVRExperimentCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.audit = self.root / "audit"
        self.data = self.root / "data"
        self.output = self.root / "output"
        for path in (self.audit, self.data, self.output):
            path.mkdir()
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"injected")
        self.resume = self.root / "resume.pt"
        self.resume.write_bytes(b"injected")

    def tearDown(self):
        self.temporary.cleanup()

    def _module(self):
        return importlib.import_module("run_clevr_experiment")

    def _run(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = self._module().main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_import_has_no_dispatch_or_output_side_effect(self):
        sys.modules.pop("run_clevr_experiment", None)
        stdout, stderr = io.StringIO(), io.StringIO()
        runner = importlib.import_module("src.clevr_runner")
        with patch.object(runner, "train_clevr_seed") as train, \
                patch.object(runner, "evaluate_clevr_checkpoint") as evaluate, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            imported = importlib.import_module("run_clevr_experiment")
        self.assertTrue(callable(imported.main))
        train.assert_not_called()
        evaluate.assert_not_called()
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))

    def test_inspect_normalizes_methods_and_avoids_cuda_model_and_item_access(self):
        module = self._module()
        preflight = SimpleNamespace(methods=("G", "B"), vocab_size=17,
                                    train_datasets={"G": object(), "B": object()},
                                    dev_datasets={"G": object(), "B": object()})
        arguments = [
            "inspect", "--phase", "formal", "--seed", "101",
            "--method", "B", "--method", "G", "--audit-dir", str(self.audit),
            "--data-root", str(self.data), "--device-index", "2",
        ]
        with patch.object(module, "preflight_clevr_run", return_value=preflight) as inspect, \
                patch.object(module, "train_clevr_seed", side_effect=AssertionError("model access")), \
                patch.object(module, "evaluate_clevr_checkpoint", side_effect=AssertionError("evaluation")), \
                patch.object(module, "_setup_clevr_runtime", side_effect=AssertionError("CUDA access")):
            code, stdout, stderr = self._run(arguments)
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout), {
            "command": "inspect", "methods": ["G", "B"], "phase": "formal",
            "seed": 101, "status": "ok", "vocab_size": 17,
        })
        inspect.assert_called_once_with(
            spec_phase="formal", seed=101, methods=("G", "B"),
            audit_dir=self.audit, data_root=self.data, device_index=2,
        )

    def test_train_forwards_exact_normalized_explicit_values_and_resume_mapping(self):
        module = self._module()
        new_output = self.root / "new-output"
        arguments = [
            "train", "--phase", "pilot", "--seed", "9001",
            "--method", "Q-only", "--method", "G",
            "--audit-dir", str(self.audit), "--data-root", str(self.data),
            "--output-root", str(new_output), "--device-index", "1",
            "--resume", f"G={self.resume}",
        ]
        with patch.object(module, "train_clevr_seed") as train:
            code, stdout, stderr = self._run(arguments)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout), {"command": "train", "status": "complete"})
        train.assert_called_once_with(
            spec_phase="pilot", seed=9001, methods=("G", "Q-only"),
            audit_dir=self.audit, data_root=self.data, output_root=new_output,
            device_index=1, resume_checkpoints={"G": self.resume},
        )

    def test_evaluate_forwards_one_formal_method_and_explicit_checkpoint(self):
        module = self._module()
        published = self.root / "G_seed_101.jsonl"
        arguments = [
            "evaluate", "--phase", "formal", "--seed", "101", "--method", "G",
            "--audit-dir", str(self.audit), "--data-root", str(self.data),
            "--output-root", str(self.output), "--device-index", "0",
            "--checkpoint", str(self.checkpoint),
        ]
        with patch.object(module, "evaluate_clevr_checkpoint", return_value=published) as evaluate:
            code, stdout, stderr = self._run(arguments)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout), {
            "command": "evaluate", "predictions": str(published), "status": "complete",
        })
        evaluate.assert_called_once_with(
            spec_phase="formal", seed=101, method="G", audit_dir=self.audit,
            data_root=self.data, output_root=self.output, device_index=0,
            checkpoint=self.checkpoint, partition="base_test",
        )

    def test_validation_and_runtime_failures_are_json_stderr_and_nonzero(self):
        module = self._module()
        base_train = [
            "train", "--phase", "pilot", "--seed", "9001", "--method", "G",
            "--audit-dir", str(self.audit), "--data-root", str(self.data),
            "--output-root", str(self.output), "--device-index", "0",
        ]
        bad_seed = list(base_train)
        bad_seed[bad_seed.index("9001")] = "101"
        relative_audit = list(base_train)
        relative_audit[relative_audit.index(str(self.audit))] = "relative-audit"
        cases = (
            ["inspect"],
            base_train + ["--method", "G"],
            base_train + ["--resume", f"G={self.resume}", "--resume", f"G={self.resume}"],
            base_train + ["--resume", f"B={self.resume}"],
            bad_seed,
            relative_audit,
            base_train + ["--download"],
            base_train[:-1] + ["auto"],
            base_train[:-2] + ["--url", "https://example.invalid/data"],
            [
                "evaluate", "--phase", "pilot", "--seed", "9001", "--method", "G",
                "--audit-dir", str(self.audit), "--data-root", str(self.data),
                "--output-root", str(self.output), "--device-index", "0",
                "--checkpoint", str(self.checkpoint),
            ],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), \
                    patch.object(module, "train_clevr_seed") as train, \
                    patch.object(module, "evaluate_clevr_checkpoint") as evaluate:
                code, stdout, stderr = self._run(arguments)
                self.assertNotEqual(code, 0)
                self.assertEqual(stdout, "")
                error = json.loads(stderr)
                self.assertEqual(set(error), {"error", "type"})
                self.assertTrue(error["error"])
                train.assert_not_called()
                evaluate.assert_not_called()
        with patch.object(module, "train_clevr_seed", side_effect=RuntimeError("injected failure")):
            code, stdout, stderr = self._run(base_train)
        self.assertEqual((code, stdout), (1, ""))
        self.assertEqual(json.loads(stderr), {
            "error": "injected failure", "type": "RuntimeError",
        })

    def test_help_states_gpu_authorization_and_separate_base_test(self):
        help_text = self._module()._build_parser().format_help()
        self.assertIn("user-authorized GPU work", help_text)
        self.assertIn("base-test evaluation is separate", help_text)


if __name__ == "__main__":
    unittest.main()
