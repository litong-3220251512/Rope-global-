"""Production optimizer/checkpoint interface regressions; CPU, no data."""

from pathlib import Path
import tempfile
import unittest

import torch

from src.clevr_checkpoint import save_clevr_checkpoint, load_clevr_checkpoint
from src.clevr_sampling import CLEVRIndexStream
from src.clevr_training import (
    CLEVRPairedInitialization, build_clevr_adamw, build_clevr_lr_scheduler,
)


class ProductionCheckpointTests(unittest.TestCase):
    def test_all_production_arms_roundtrip_update_zero(self):
        factory = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        for method in ("G", "B", "M10", "Q-only"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                model = factory.build(method)
                expected = {k: v.clone() for k, v in model.state_dict().items()}
                optimizer = build_clevr_adamw(model)
                scheduler = build_clevr_lr_scheduler(optimizer)
                sampler = CLEVRIndexStream(dataset_size=5, run_seed=101)
                path = Path(tmp) / "update_0000.pt"
                save_clevr_checkpoint(
                    path, method=method, run_seed=101, completed_updates=0,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    sampler=sampler, cuda_devices=(),
                )
                with torch.no_grad():
                    next(model.parameters()).add_(1)
                metadata = load_clevr_checkpoint(
                    path, expected_method=method, expected_run_seed=101,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    sampler=sampler, cuda_devices=(),
                )
                self.assertEqual(metadata.completed_updates, 0)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[name]), name)
                self.assertEqual(optimizer.state_dict()["state"], {})
                self.assertEqual(scheduler.last_epoch, 0)
                self.assertEqual(sampler.state_dict()["cursor"], 0)
