"""Cross-method data provenance regressions, using synthetic local artifacts."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from src.clevr_run_artifacts import CLEVRRunArtifacts, CLEVRRunSpec, CLEVRRunArtifactError
from src.clevr_reporting import CLEVRReportError, build_statistics_bundle, publish_statistics_bundle, PredictionSource
import test_clevr_reporting as reporting_tests


FILES = ("selected_questions.jsonl", "vocabulary.json", "audit_report.json", "image_partitions.json")


def run_config(method, seed, snapshot, root="/explicit/rgb"):
    return {"protocol_version": "PUB-R1.1", "phase": "formal", "method": method,
            "seed": seed, "target_updates": 8000, "evidence_role": "formal",
            "data": {"audit_artifacts": snapshot,
                     "data_root": None if method == "Q-only" else root}}


class PairedArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = {name: name + "\n" for name in FILES}

    def open_run(self, method, *, snapshot=None, seed=101, root="/explicit/rgb"):
        config = run_config(method, seed, self.snapshot if snapshot is None else snapshot, root)
        run = CLEVRRunArtifacts.open(output_root=self.root,
            spec=CLEVRRunSpec("formal", method, seed), expanded_config=config, resume=False)
        self.addCleanup(run.close)
        return run

    def test_same_size_vocabulary_change_rejected_between_fresh_arms(self):
        first = self.open_run("G")
        first.ensure_data_contract()
        changed = dict(self.snapshot, **{"vocabulary.json": "other_vocab\n"})
        second = self.open_run("B", snapshot=changed)
        with self.assertRaisesRegex(CLEVRRunArtifactError, "paired data"):
            second.ensure_data_contract()

    def test_qonly_first_and_matching_arms_across_seeds_share_contract(self):
        first = self.open_run("Q-only")
        path = first.ensure_data_contract()
        for method, seed in (("G", 101), ("B", 101), ("M10", 102)):
            self.assertEqual(self.open_run(method, seed=seed).ensure_data_contract(), path)

    def test_rgb_root_change_is_rejected(self):
        self.open_run("G").ensure_data_contract()
        with self.assertRaisesRegex(CLEVRRunArtifactError, "RGB root"):
            self.open_run("B", root="/different/rgb").ensure_data_contract()

    def test_preexisting_run_without_shared_contract_is_not_ignored(self):
        self.open_run("G")
        changed = dict(self.snapshot, **{"selected_questions.jsonl": "changed\n"})
        with self.assertRaisesRegex(CLEVRRunArtifactError, "paired data"):
            self.open_run("B", snapshot=changed).ensure_data_contract()

    def test_each_snapshot_component_is_bound_and_existing_file_is_preserved(self):
        for index, name in enumerate(FILES):
            with self.subTest(name=name):
                self.root = Path(self.temp.name) / str(index)
                path = self.open_run("G").ensure_data_contract()
                before = path.read_bytes()
                changed = dict(self.snapshot)
                changed[name] += "changed"
                with self.assertRaises(CLEVRRunArtifactError):
                    self.open_run("B", snapshot=changed).ensure_data_contract()
                self.assertEqual(path.read_bytes(), before)

    def test_concurrent_matching_first_writers_share_complete_contract(self):
        first, second = self.open_run("G"), self.open_run("B")
        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(lambda run: run.ensure_data_contract(), (first, second)))
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(json.loads(paths[0].read_text())["audit_artifacts"], self.snapshot)


class PairedReportingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = reporting_tests.StatisticsBundleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        (f.audit_dir / "vocabulary.json").write_text('{"tokens":["red"]}\n')
        self.snapshot = {name: (f.audit_dir / name).read_bytes().decode() for name in FILES}
        for (method, seed), source in f.source_by_key.items():
            source.path.with_suffix(".run_config.json").write_text(
                json.dumps(run_config(method, seed, self.snapshot)))

    def test_reporting_rejects_different_training_vocabulary(self):
        path = self.fixture.sources[5].path.with_suffix(".run_config.json")
        config = json.loads(path.read_text())
        config["data"]["audit_artifacts"]["vocabulary.json"] = '{"tokens":["tan"]}\n'
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(CLEVRReportError, "paired data"):
            self.fixture._build()

    def test_reporting_rejects_missing_training_provenance(self):
        self.fixture.sources[5].path.with_suffix(".run_config.json").unlink()
        with self.assertRaises(CLEVRReportError):
            self.fixture._build()

    def test_export_retains_configs_for_rechecking(self):
        bundle = self.fixture._build()
        output = self.fixture.root / "export"
        publish_statistics_bundle(output, bundle)
        sources = tuple(PredictionSource(output / item.canonical_path) for item in bundle.report.predictions)
        rebuilt = build_statistics_bundle(self.fixture.audit_dir, sources)
        self.assertEqual(rebuilt.report.statistics, bundle.report.statistics)
        self.assertEqual(rebuilt.report.report_schema_version, 2)

    def test_run_identity_and_rgb_root_mismatches_are_rejected(self):
        path = self.fixture.sources[5].path.with_suffix(".run_config.json")
        original = json.loads(path.read_text())
        for field, value in (("method", "G"), ("seed", 102), ("seed", True),
                             ("phase", "pilot"), ("target_updates", 1000)):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                changed[field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(CLEVRReportError):
                    self.fixture._build()
        changed = copy.deepcopy(original)
        changed["data"]["data_root"] = "/other/rgb"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(CLEVRReportError, "RGB roots"):
            self.fixture._build()

    def test_runner_paths_and_explicit_config_paths_are_supported(self):
        sources = []
        for index, source in enumerate(self.fixture.sources):
            old_config = source.path.with_suffix(".run_config.json")
            config = json.loads(old_config.read_text())
            run_dir = self.fixture.root / "runs" / "formal" / f"seed_{config['seed']}" / config["method"]
            prediction = run_dir / "predictions" / "base_test" / source.path.name
            prediction.parent.mkdir(parents=True)
            prediction.write_bytes(source.path.read_bytes())
            target = run_dir / "run_config.json"
            target.write_bytes(old_config.read_bytes())
            sources.append(PredictionSource(prediction, target if index % 2 else None))
        bundle = build_statistics_bundle(self.fixture.audit_dir, sources)
        self.assertEqual(len(bundle.runs), 20)
