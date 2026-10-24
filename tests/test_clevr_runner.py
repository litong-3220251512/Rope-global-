"""Synthetic CPU tests for the committed CLEVR computation seam."""

import importlib
import importlib.util
import json
import math
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from copy import deepcopy
from unittest.mock import Mock, call, patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.clevr_dataset import (
    CLEVRSampleIdentity, collate_clevr_questions, collate_clevr_question_only,
)
from src.clevr_run_artifacts import (
    CLEVRPredictionRecord, CLEVRRunArtifactError, CLEVRRunArtifacts,
)
from src.clevr_sampling import CLEVRIndexStream, CLEVRSamplingError


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


class TinyDataset:
    def __init__(self, size=129, *, rgb=True, partition="train"):
        self.size, self.rgb, self.partition = size, rgb, partition
        self.accessed = []
        self.identities = []
        self.image = torch.full((3, 224, 224), 0.25)
        self.fail_item = None
        self.identity_override = None

    def __len__(self):
        return self.size

    def sample_identity(self, index):
        self.identities.append(index)
        if self.identity_override is not None:
            return self.identity_override(index)
        return CLEVRSampleIdentity(
            self.partition, "train" if self.partition != "base_test" else "val",
            index, f"fake_{index}.png", 1000 + self.size - index, index,
        )

    def __getitem__(self, index):
        if index == self.fail_item:
            raise RuntimeError("item failure")
        self.accessed.append(index)
        middle = [3 + index % 5] * (1 + index % 2)
        result = {
            "question_tokens": torch.tensor([1, *middle, 2]),
            "question_length": torch.tensor(2 + len(middle)),
            "answer": torch.tensor(int(index % 5 == 0)),
        }
        if self.rgb:
            result["image"] = self.image
        return result


class TinyAnswerer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.8, -0.6], [0.2, 0.4]]))
        self.bias = nn.Parameter(torch.tensor([0.1, -0.1]))
        self.patch = nn.Parameter(torch.tensor(0.3))
        self.calls = []
        self.output_grads = []
        self.fail_call = None
        self.output_transform = None
        self.observe = None

    def forward(self, image, tokens, lengths):
        self.calls.append((image is None, len(tokens), torch.is_inference_mode_enabled(), self.training))
        if self.observe is not None:
            self.observe()
        if len(self.calls) == self.fail_call:
            raise RuntimeError("forward failure")
        features = torch.stack((tokens[:, 1].float(), lengths.float()), dim=1)
        logits = features @ self.weight + self.bias
        if image is not None:
            logits = logits + self.patch * image[:, 0, 0, 0, None] * logits.new_tensor([1., -1.])
        if self.output_transform is not None:
            logits = self.output_transform(logits)
        if logits.requires_grad:
            logits.register_hook(lambda grad: self.output_grads.append(grad.detach().clone()))
        return logits


class ObservedAdamW(AdamW):
    def __init__(self, model, events):
        self.events = events
        self.step_lrs = []
        self.patch_grads = []
        self.model = model
        super().__init__([
            {"params": [model.weight, model.patch], "weight_decay": 0.1},
            {"params": [model.bias], "weight_decay": 0.0},
        ], lr=0.02)

    def zero_grad(self, set_to_none=True):
        self.events.append(("zero", set_to_none))
        return super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.events.append("optimizer")
        self.step_lrs.append(tuple(g["lr"] for g in self.param_groups))
        self.patch_grads.append(None if self.model.patch.grad is None else self.model.patch.grad.clone())
        return super().step(closure)


class ObservedScheduler(LambdaLR):
    def __init__(self, optimizer, events):
        self.events = events
        super().__init__(optimizer, lambda epoch: 1 / (epoch + 1))

    def step(self, epoch=None):
        self.events.append("scheduler")
        return super().step(epoch)


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("src.clevr_runner"),
                             "Task 3 computation engine is missing")
        self.runner = importlib.import_module("src.clevr_runner")
        self.events = []
        self.dataset = TinyDataset()
        self.model = TinyAnswerer()
        self.optimizer = ObservedAdamW(self.model, self.events)
        self.scheduler = ObservedScheduler(self.optimizer, self.events)
        self.events.clear()
        self.sampler = CLEVRIndexStream(dataset_size=129, run_seed=101)
        self.groups = []

    def collate(self, samples):
        self.groups.append(len(samples))
        return (collate_clevr_questions if self.dataset.rgb else collate_clevr_question_only)(samples)

    def update(self, **overrides):
        args = dict(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    sampler=self.sampler, dataset=self.dataset, collate_fn=self.collate,
                    method="G" if self.dataset.rgb else "Q-only", device=torch.device("cpu"))
        args.update(overrides)
        return self.runner.run_clevr_update(**args)

    def evaluate(self, **overrides):
        args = dict(model=self.model, dataset=self.dataset, collate_fn=self.collate,
                    method="G" if self.dataset.rgb else "Q-only", partition=self.dataset.partition,
                    device=torch.device("cpu"))
        args.update(overrides)
        return self.runner.evaluate_clevr_model(**args)


class UpdateTests(RunnerCase):
    def test_exact_accumulation_order_loss_gradients_and_commit(self):
        indices = self.sampler.preview(128)
        initial_sampler = deepcopy(self.sampler.state_dict())
        reference = TinyAnswerer()
        reference.load_state_dict(self.model.state_dict())
        ref_opt = ObservedAdamW(reference, [])
        ref_scheduler = ObservedScheduler(ref_opt, [])
        batch = collate_clevr_questions([self.dataset[i] for i in indices])
        logits = reference(batch["image"], batch["question_tokens"], batch["question_length"])
        individual = F.cross_entropy(logits, batch["answer"], reduction="none")
        # Independent scalar binary CE, avoiding the runner's reduction logic.
        scalar_losses = [math.log1p(math.exp(float(z[1] - z[0]))) - int(y) * float(z[1] - z[0])
                         for z, y in zip(logits.detach(), batch["answer"])]
        individual.mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
        ref_opt.step()
        ref_scheduler.step()
        ref_opt.zero_grad(set_to_none=True)
        self.dataset.accessed.clear()
        # A dirty prior gradient must be cleared before this call's backward.
        self.model.bias.grad = torch.ones_like(self.model.bias) * 123
        self.model.observe = lambda: self.assertEqual(self.sampler.state_dict(), initial_sampler)
        original_clip = torch.nn.utils.clip_grad_norm_
        clip_calls = []
        def clip(parameters, max_norm, **kwargs):
            self.events.append("clip")
            clip_calls.append((max_norm, kwargs))
            return original_clip(parameters, max_norm, **kwargs)
        original_commit = self.sampler.commit
        def commit(supplied):
            self.events.append("commit")
            self.assertEqual(tuple(supplied), indices)
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))
            self.assertEqual(self.sampler.state_dict(), initial_sampler)
            return original_commit(supplied)
        original_preview = self.sampler.preview
        with patch.object(self.sampler, "preview", wraps=original_preview) as preview, \
                patch.object(self.sampler, "commit", side_effect=commit), \
                patch("torch.nn.utils.clip_grad_norm_", side_effect=clip):
            result = self.update()
        # commit's own validation also previews the exact stream prefix.
        self.assertEqual(preview.call_args_list[0].args, (128,))
        self.assertTrue(all(call.args == (128,) for call in preview.call_args_list))
        self.assertEqual(self.dataset.accessed, list(indices))
        self.assertEqual(self.groups, [16] * 8)
        self.assertEqual(self.events, [("zero", True), "clip", "optimizer", "scheduler", ("zero", True), "commit"])
        self.assertEqual(clip_calls, [(1.0, {"error_if_nonfinite": True})])
        self.assertEqual(self.optimizer.step_lrs, [(0.02, 0.02)])
        self.assertEqual(result.learning_rate, 0.02)
        self.assertEqual([g["lr"] for g in self.optimizer.param_groups], [0.01, 0.01])
        self.assertEqual(result.dataset_indices, indices)
        self.assertEqual(result.question_indices, tuple(1129 - i for i in indices))
        self.assertAlmostEqual(result.loss_sum, sum(scalar_losses), places=4)
        self.assertAlmostEqual(result.loss_mean, sum(scalar_losses) / 128, places=6)
        self.assertAlmostEqual(result.gradient_norm, norm.item(), places=5)
        self.assertGreater(result.gradient_norm, 1.0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name])
        expected_grad = logits.detach().softmax(-1)
        expected_grad[torch.arange(128), batch["answer"]] -= 1
        torch.testing.assert_close(torch.cat(self.model.output_grads), expected_grad / 128)
        self.assertEqual(self.sampler.state_dict()["cursor"], 128)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(call == (False, 16, False, True) for call in self.model.calls))

    def test_qonly_unused_patch_has_no_gradient_or_weight_decay_update(self):
        self.dataset.rgb = False
        initial_patch = self.model.patch.detach().clone()
        self.model.patch.grad = torch.ones_like(self.model.patch)
        self.update()
        self.assertEqual(self.optimizer.patch_grads, [None])
        self.assertIsNone(self.model.patch.grad)
        self.assertNotIn(self.model.patch, self.optimizer.state)
        self.assertTrue(torch.equal(initial_patch, self.model.patch))
        self.assertTrue(all(call[0] for call in self.model.calls))

    def test_all_rgb_methods_share_the_call_contract(self):
        for method in ("G", "B", "M10"):
            with self.subTest(method=method):
                self.update(method=method)
        self.assertEqual(len(self.model.calls), 24)
        self.assertTrue(all(not call[0] for call in self.model.calls))

    def test_failure_boundaries_never_return_success_or_commit_early(self):
        for boundary in ("forward1", "forward8", "backward", "clip", "optimizer", "scheduler", "commit"):
            with self.subTest(boundary=boundary):
                self.setUp()
                state = deepcopy(self.sampler.state_dict())
                initial_weight = self.model.weight.detach().clone()
                if boundary.startswith("forward"):
                    self.model.fail_call = int(boundary[-1])
                if boundary == "backward":
                    def fail_grad(grad):
                        raise RuntimeError("backward failure")
                    self.model.weight.register_hook(fail_grad)
                from contextlib import ExitStack
                with ExitStack() as stack:
                    commit = stack.enter_context(patch.object(self.sampler, "commit", wraps=self.sampler.commit))
                    if boundary == "clip":
                        stack.enter_context(patch("torch.nn.utils.clip_grad_norm_", side_effect=RuntimeError(boundary)))
                    elif boundary in ("optimizer", "scheduler"):
                        owner = self.optimizer if boundary == "optimizer" else self.scheduler
                        original = owner.step
                        def after_effect(*args, **kwargs):
                            original(*args, **kwargs)
                            raise RuntimeError(boundary)
                        stack.enter_context(patch.object(owner, "step", side_effect=after_effect))
                    elif boundary == "commit":
                        commit.side_effect = RuntimeError(boundary)
                    with self.assertRaisesRegex(RuntimeError, "failure|clip|optimizer|scheduler|commit"):
                        self.update()
                    self.assertEqual(commit.call_count, int(boundary == "commit"))
                self.assertEqual(self.sampler.state_dict(), state)
                if boundary in ("optimizer", "scheduler", "commit"):
                    self.assertFalse(torch.equal(initial_weight, self.model.weight))
                else:
                    self.assertTrue(torch.equal(initial_weight, self.model.weight))
                self.assertEqual(self.events[0], ("zero", True))

    def test_commit_stream_mismatch_is_not_complete(self):
        def disturb():
            if len(self.model.calls) == 8:
                self.sampler.commit(self.sampler.preview(1))
        self.model.observe = disturb
        with self.assertRaises(CLEVRSamplingError):
            self.update()
        self.assertEqual(self.sampler.state_dict()["cursor"], 1)

    def test_post_effect_commit_failure_is_not_success_or_rollback(self):
        original = self.sampler.commit
        def fail_after(indices):
            original(indices)
            raise RuntimeError("post commit")
        with patch.object(self.sampler, "commit", side_effect=fail_after):
            with self.assertRaisesRegex(RuntimeError, "post commit"):
                self.update()
        self.assertEqual(self.sampler.state_dict()["cursor"], 128)

    def test_bad_method_lr_groups_and_dataset_size_reject_before_forward(self):
        cases = ("method", "different_lr", "nan_lr", "inf_lr", "one_group", "size")
        for case in cases:
            with self.subTest(case=case):
                self.setUp()
                args = {}
                if case == "method":
                    args["method"] = "C"
                elif case == "different_lr":
                    self.optimizer.param_groups[1]["lr"] = 0.03
                elif case in ("nan_lr", "inf_lr"):
                    for group in self.optimizer.param_groups:
                        group["lr"] = float(case[:3])
                elif case == "one_group":
                    self.optimizer.param_groups.pop()
                else:
                    self.dataset.size = 130
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self.update(**args)
                self.assertEqual(self.events, [("zero", True)])
                self.assertFalse(self.model.calls)
                self.assertEqual(self.sampler.state_dict()["cursor"], 0)

    def test_malformed_collated_batches_are_rejected(self):
        mutations = {
            "extra": lambda b: b.update(question_index=torch.zeros(16)),
            "missing": lambda b: b.pop("question_length"),
            "rgb_without_image": lambda b: b.pop("image"),
            "label_rank": lambda b: b.update(answer=b["answer"][:, None]),
            "label_dtype": lambda b: b.update(answer=b["answer"].float()),
            "label_range": lambda b: b.update(answer=torch.full((16,), 2)),
            "tokens_rank": lambda b: b.update(question_tokens=b["question_tokens"][:, None]),
            "tokens_dtype": lambda b: b.update(question_tokens=b["question_tokens"].float()),
            "length_shape": lambda b: b.update(question_length=b["question_length"][:1]),
            "length_dtype": lambda b: b.update(question_length=b["question_length"].float()),
            "length_range": lambda b: b.update(question_length=torch.zeros(16, dtype=torch.long)),
            "image_shape": lambda b: b.update(image=b["image"][:, :1]),
            "image_nan": lambda b: b.update(image=b["image"] * float("nan")),
            "value_type": lambda b: b.update(answer=[0] * 16),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                def corrupt(samples):
                    batch = self.collate(samples)
                    mutate(batch)
                    return batch
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self.update(collate_fn=corrupt)
                self.assertEqual(self.sampler.state_dict()["cursor"], 0)
        self.dataset.rgb = False
        with self.assertRaises(self.runner.CLEVRRunnerError):
            self.update(collate_fn=lambda s: {**self.collate(s), "image": self.dataset.image})

    def test_bad_logits_and_nonfinite_loss_or_gradients_reject(self):
        transforms = {
            "shape": lambda z: z[:, :1],
            "nan": lambda z: z * float("nan"),
            "inf": lambda z: z * float("inf"),
            # Finite FP32 inputs with an overflowing loss.
            "loss": lambda z: z * 0 + z.new_tensor([3e38, -3e38]),
        }
        for name, transform in transforms.items():
            with self.subTest(name=name):
                self.model.output_transform = transform
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self.update()
                self.assertEqual(self.sampler.state_dict()["cursor"], 0)
        self.model.output_transform = None
        handle = self.model.weight.register_hook(lambda g: g * float("nan"))
        try:
            with self.assertRaises(self.runner.CLEVRRunnerError):
                self.update()
        finally:
            handle.remove()
        self.assertFalse(self.optimizer.state)
        self.assertEqual(self.sampler.state_dict()["cursor"], 0)

    def test_invalid_identity_rejects_before_optimizer_or_commit(self):
        for value in (-1, True, 1.0):
            with self.subTest(value=value):
                self.dataset.identity_override = lambda i: CLEVRSampleIdentity("train", "train", i, "x", value, i)
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self.update()
                self.assertFalse(self.optimizer.state)
                self.assertEqual(self.sampler.state_dict()["cursor"], 0)


class EvaluationTests(RunnerCase):
    def setUp(self):
        super().setUp()
        self.dataset = TinyDataset(19, partition="dev")

    def test_ascending_uneven_sum_mean_accuracy_and_sorted_exact_rows(self):
        # First 18 have tied logits; last has a high-confidence wrong prediction.
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
            self.model.patch.zero_()
        def logits(z):
            if len(self.model.calls) == 2:
                z = z.clone()
                z[-1] = z.new_tensor([0., 4.])
            return z
        self.model.output_transform = logits
        result = self.evaluate()
        self.assertEqual(self.dataset.accessed, list(range(19)))
        self.assertEqual(self.dataset.identities, list(range(19)))
        self.assertEqual(self.groups, [16, 3])
        self.assertEqual(self.model.calls, [(False, 16, True, False), (False, 3, True, False)])
        self.assertTrue(self.model.training)
        self.assertEqual(result.partition, "dev")
        self.assertEqual(result.total_questions, 19)
        # Four true labels at 0,5,10,15 plus one wrong last prediction.
        self.assertEqual(result.correct, 14)
        self.assertEqual(result.accuracy, 14 / 19)
        total_loss = 18 * math.log(2) + math.log1p(math.exp(4))
        self.assertAlmostEqual(result.loss_sum, total_loss, places=5)
        self.assertAlmostEqual(result.cross_entropy, total_loss / 19, places=6)
        expected_rows = tuple(sorted((CLEVRPredictionRecord(
            1019 - i, int(i % 5 == 0), (0., 4. if i == 18 else 0.))
            for i in range(19)), key=lambda row: row.question_index))
        self.assertEqual(result.rows, expected_rows)
        self.assertTrue(all(type(v) is float and math.isfinite(v) for row in result.rows for v in row.logits))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_qonly_passes_none_and_preserves_eval_mode(self):
        self.dataset.rgb = False
        self.model.eval()
        result = self.evaluate()
        self.assertFalse(self.model.training)
        self.assertEqual(len(result.rows), 19)
        self.assertTrue(all(call[0] and call[2] and not call[3] for call in self.model.calls))

    def test_restores_mode_on_item_collation_forward_and_identity_failure(self):
        for training in (True, False):
            for boundary in ("item", "collate", "forward", "identity"):
                with self.subTest(training=training, boundary=boundary):
                    self.setUp()
                    self.model.train(training)
                    args = {}
                    if boundary == "item":
                        self.dataset.fail_item = 16
                    elif boundary == "collate":
                        def fail_collate(samples):
                            raise RuntimeError("collate failure")
                        args["collate_fn"] = fail_collate
                    elif boundary == "forward":
                        self.model.fail_call = 2
                    else:
                        def fail_identity(i):
                            raise RuntimeError("identity failure")
                        self.dataset.identity_override = fail_identity
                    with self.assertRaisesRegex(RuntimeError, "failure"):
                        self.evaluate(**args)
                    self.assertEqual(self.model.training, training)

    def test_empty_duplicate_partition_and_invalid_ids_reject(self):
        for case in ("empty", "duplicate", "partition", "negative", "bool", "confirmation"):
            with self.subTest(case=case):
                self.setUp()
                args = {}
                if case == "empty":
                    self.dataset.size = 0
                elif case == "confirmation":
                    args["partition"] = "confirmation"
                else:
                    def identity(i):
                        return CLEVRSampleIdentity(
                            "train" if case == "partition" else "dev", "train", i, "x",
                            {"duplicate": 7, "negative": -1, "bool": True}.get(case, i), i)
                    self.dataset.identity_override = identity
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self.evaluate(**args)
                self.assertTrue(self.model.training)

    def test_bad_contract_and_nonfinite_outputs_do_not_return_partial_result(self):
        for case in ("method", "shape", "nan", "loss", "labels", "keys", "qonly_image"):
            with self.subTest(case=case):
                self.setUp()
                args = {}
                if case == "method":
                    args["method"] = "bad"
                elif case in ("shape", "nan", "loss"):
                    self.model.output_transform = {
                        "shape": lambda z: z[:, :1],
                        "nan": lambda z: z * float("nan"),
                        "loss": lambda z: z * 0 + z.new_tensor([3e38, -3e38]),
                    }[case]
                else:
                    if case == "qonly_image":
                        args["method"] = "Q-only"
                    def corrupt(samples):
                        b = self.collate(samples)
                        if case == "labels":
                            b["answer"] = torch.full((len(samples),), 2)
                        elif case == "keys":
                            b["question_index"] = torch.zeros(len(samples))
                        return b
                    args["collate_fn"] = corrupt
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self.evaluate(**args)
                self.assertTrue(self.model.training)


class _PreflightDataset:
    constructions = []
    sizes = {"train": 3, "dev": 2}
    vocabularies = {"rgb": 17, "qonly": 17}
    identity_delta = 0

    def __init__(self, *, kind, partition, audit_dir, data_root=None):
        self.kind = kind
        self.partition = partition
        self.vocab_size = self.vocabularies[kind]
        self.constructions.append((kind, partition, audit_dir, data_root))

    def __len__(self):
        return self.sizes[self.partition]

    def sample_identity(self, index):
        delta = self.identity_delta if self.kind == "qonly" else 0
        return CLEVRSampleIdentity(
            self.partition, "train", index, f"image_{index}.png",
            100 + index + delta, index,
        )

    def __getitem__(self, index):
        raise AssertionError("preflight must not access model-facing tensors")


class ProductionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module("src.clevr_runner")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.audit = self.root / "audit"
        self.data = self.root / "data"
        self.audit.mkdir()
        self.data.mkdir()
        self.audit_contents = {
            "selected_questions.jsonl": '{"question_index":1}\n',
            "vocabulary.json": '{"token_to_index":{"x":4}}\n',
            "audit_report.json": '{"protocol_version":"PUB-R1.1"}\n',
        }
        for name, content in self.audit_contents.items():
            (self.audit / name).write_text(content, encoding="utf-8")
        _PreflightDataset.constructions = []
        _PreflightDataset.sizes = {"train": 3, "dev": 2}
        _PreflightDataset.vocabularies = {"rgb": 17, "qonly": 17}
        _PreflightDataset.identity_delta = 0

    def tearDown(self):
        self.temporary.cleanup()

    def _rgb(self, **kwargs):
        return _PreflightDataset(kind="rgb", **kwargs)

    def _qonly(self, **kwargs):
        return _PreflightDataset(kind="qonly", **kwargs)

    def preflight(self, **overrides):
        arguments = dict(
            spec_phase="formal", seed=101, methods=("B", "Q-only", "G"),
            audit_dir=self.audit, data_root=self.data, device_index=2,
        )
        arguments.update(overrides)
        with patch.object(self.runner, "CLEVRQuestionDataset", side_effect=self._rgb), \
                patch.object(self.runner, "CLEVRQuestionOnlyDataset", side_effect=self._qonly):
            return self.runner.preflight_clevr_run(**arguments)

    def test_normalizes_methods_and_constructs_only_train_dev_without_item_access(self):
        result = self.preflight()
        self.assertEqual(result.methods, ("G", "B", "Q-only"))
        self.assertEqual(tuple(result.specs), result.methods)
        self.assertEqual(result.vocab_size, 17)
        self.assertIs(result.train_datasets["G"], result.train_datasets["B"])
        self.assertIsNot(result.train_datasets["G"], result.train_datasets["Q-only"])
        self.assertEqual(
            _PreflightDataset.constructions,
            [
                ("rgb", "train", self.audit, self.data),
                ("rgb", "dev", self.audit, self.data),
                ("qonly", "train", self.audit, None),
                ("qonly", "dev", self.audit, None),
            ],
        )

    def test_qonly_requires_data_root_absent_and_never_constructs_rgb(self):
        result = self.preflight(methods=("Q-only",), data_root=None)
        self.assertEqual(result.methods, ("Q-only",))
        self.assertEqual(
            _PreflightDataset.constructions,
            [("qonly", "train", self.audit, None),
             ("qonly", "dev", self.audit, None)],
        )
        with self.assertRaises(self.runner.CLEVRRunnerError):
            self.preflight(methods=("Q-only",), data_root=self.data)

    def test_rejects_paths_methods_phase_seed_and_device_before_dataset_or_cuda(self):
        cases = (
            {"audit_dir": Path("relative")},
            {"audit_dir": self.root / "missing"},
            {"data_root": None},
            {"data_root": Path("relative")},
            {"methods": ()},
            {"methods": ("G", "G")},
            {"methods": ("G", "unknown")},
            {"spec_phase": "formal", "seed": 9001},
            {"device_index": -1},
            {"device_index": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                _PreflightDataset.constructions.clear()
                with patch.object(self.runner.torch, "cuda") as cuda:
                    with self.assertRaises((self.runner.CLEVRRunnerError, ValueError)):
                        self.preflight(**overrides)
                    self.assertFalse(_PreflightDataset.constructions)
                    self.assertFalse(cuda.mock_calls)

    def test_rejects_rgb_qonly_length_vocab_or_identity_disagreement_without_items(self):
        for case in ("length", "vocab", "identity"):
            with self.subTest(case=case):
                _PreflightDataset.sizes = {"train": 3, "dev": 2}
                _PreflightDataset.vocabularies = {"rgb": 17, "qonly": 17}
                _PreflightDataset.identity_delta = 0
                if case == "length":
                    _PreflightDataset.sizes = {"train": 3, "dev": 2, "qonly": 4}
                    original_len = _PreflightDataset.__len__
                    def mismatched_len(instance):
                        if instance.kind == "qonly" and instance.partition == "train":
                            return 4
                        return original_len(instance)
                    context = patch.object(_PreflightDataset, "__len__", mismatched_len)
                else:
                    context = patch.object(_PreflightDataset, "__len__", _PreflightDataset.__len__)
                if case == "vocab":
                    _PreflightDataset.vocabularies["qonly"] = 18
                if case == "identity":
                    _PreflightDataset.identity_delta = 1
                with context, self.assertRaises(self.runner.CLEVRRunnerError):
                    self.preflight(methods=("G", "Q-only"))

    def test_runtime_requires_exact_cuda_and_disables_both_tf32_backends(self):
        calls = []
        class Flag:
            allow_tf32 = True
        class CudnnFlag:
            allow_tf32 = True
            deterministic = False
            benchmark = True
        matmul = Flag()
        cudnn = CudnnFlag()
        with patch.object(self.runner.torch.cuda, "is_available", return_value=True) as available, \
                patch.object(self.runner.torch.cuda, "device_count", return_value=3) as count, \
                patch.object(self.runner.torch.cuda, "set_device", side_effect=lambda i: calls.append(i)), \
                patch.object(self.runner.torch.backends.cuda, "matmul", matmul), \
                patch.object(self.runner.torch.backends, "cudnn", cudnn):
            device = self.runner._setup_clevr_runtime(2)
        self.assertEqual(device, torch.device("cuda:2"))
        self.assertEqual(calls, [2])
        self.assertTrue(available.called)
        self.assertTrue(count.called)
        self.assertFalse(matmul.allow_tf32)
        self.assertFalse(cudnn.allow_tf32)
        with patch.object(self.runner.torch.cuda, "is_available", return_value=False), \
                patch.object(self.runner.torch.cuda, "set_device") as set_device:
            with self.assertRaises(self.runner.CLEVRRunnerError):
                self.runner._setup_clevr_runtime(0)
            set_device.assert_not_called()

    def test_expanded_config_is_exact_auditable_and_contains_no_hash(self):
        class MatmulFlag:
            allow_tf32 = False
        class CudnnFlag:
            allow_tf32 = False
            deterministic = True
            benchmark = False
        with patch.object(self.runner.torch.cuda, "get_device_name", return_value="Injected GPU"), \
                patch.object(self.runner.torch, "are_deterministic_algorithms_enabled", return_value=True), \
                patch.object(self.runner.torch, "is_deterministic_algorithms_warn_only_enabled", return_value=False), \
                patch.object(self.runner.torch.backends.cuda, "matmul", MatmulFlag()), \
                patch.object(self.runner.torch.backends, "cudnn", CudnnFlag()):
            config = self.runner._expanded_clevr_config(
                device_index=2, audit_dir=self.audit, data_root=self.data,
                vocab_size=17, train_size=3,
            )
        self.assertEqual(config, {
            "protocol_version": "PUB-R1.1",
            "data": {
                "audit_dir": str(self.audit), "data_root": str(self.data),
                "train_size": 3, "vocab_size": 17,
                "audit_artifacts": self.audit_contents,
            },
            "packages": {
                "python": "3.11.14", "numpy": "2.2.6", "torch": "2.6.0+cu124",
                "torchvision": "0.21.0+cu124", "timm": "1.0.24", "pillow": "12.0.0",
                "cuda_runtime": "12.4",
            },
            "model": {
                "image_shape": [3, 224, 224], "patch_size": 16, "grid_size": [14, 14],
                "embed_dim": 192, "depth": 6, "heads": 3, "head_dim": 64,
                "mlp_dim": 768, "num_classes": 2, "question_embedding_dim": 64,
                "question_hidden_dim": 192, "question_layers": 1,
                "question_bidirectional": False, "attention_backend": "manual",
                "layer_norm_epsilon": 1e-6, "layer_scale_initial": 1e-4,
                "dropout": 0.0, "attention_dropout": 0.0, "drop_path": 0.0,
                "fixed_lambda": 1.0, "global_theta": 100.0, "mixed_theta": 10.0,
                "m10_seed_offset": 20000,
            },
            "training": {
                "effective_batch_size": 128, "micro_batch_size": 16,
                "accumulation_steps": 8, "optimizer": "AdamW", "betas": [0.9, 0.999],
                "epsilon": 1e-8, "weight_decay": 0.05, "base_learning_rate": 3e-4,
                "minimum_learning_rate": 1e-6, "warmup_updates": 400,
                "scheduler_total_updates": 8000, "loss": "cross_entropy",
                "label_smoothing": 0.0, "gradient_clip_norm": 1.0,
                "sampler": "numpy.random.PCG64", "sampler_seed_offset": 10000,
                "checkpoint_interval_updates": 1000,
            },
            "runtime": {
                "device": "cuda:2", "device_index": 2, "device_name": "Injected GPU",
                "precision": "float32", "amp": False,
                "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                "deterministic_algorithms": True, "deterministic_warn_only": False,
                "cudnn_deterministic": True, "cudnn_benchmark": False,
            },
        })
        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)
        self.assertFalse(any("hash" in key.lower() for key in keys(config)))


class _OrchestrationDataset:
    def __init__(self, name, size=257):
        self.name = name
        self.size = size
        self.vocab_size = 17
        self.partition = ("base_test" if "base" in name else
                          "dev" if "dev" in name else "train")

    def __len__(self):
        return self.size

    def sample_identity(self, index):
        return CLEVRSampleIdentity(
            self.partition, "val" if self.partition == "base_test" else "train",
            index, f"image_{index}.png", 10_000 + index, index,
        )

    def __getitem__(self, index):
        return {"answer": torch.tensor(index % 2, dtype=torch.int64)}


class _OrchestrationModel:
    def __init__(self, method, trace):
        self.method = method
        self.trace = trace
        self.parameter = nn.Parameter(torch.tensor(1.0))

    def to(self, device):
        self.trace.append(("to", self.method, str(device)))
        return self

    def train(self, mode=True):
        self.trace.append(("train", self.method, mode))
        return self

    def parameters(self):
        return (self.parameter,)


class _OrchestrationFactory:
    instances = []
    trace = []

    def __init__(self, *, vocab_size, run_seed):
        self.vocab_size = vocab_size
        self.run_seed = run_seed
        self.builds = []
        self.instances.append(self)
        self.trace.append(("factory", vocab_size, run_seed))

    def common_state_dict(self):
        self.trace.append("common-state")
        return OrderedDict((("shared", torch.tensor([3.0])),))

    def build(self, method):
        self.trace.append(("build", method))
        self.builds.append(method)
        return _OrchestrationModel(method, self.trace)


class _OrchestrationSampler:
    def __init__(self, *, dataset_size, run_seed):
        self.dataset_size = dataset_size
        self.run_seed = run_seed
        self.completed = 0


class _OrchestrationArtifacts:
    opened = []
    trace = []
    root = None
    fail_failure_event = False

    @classmethod
    def open(cls, *, output_root, spec, expanded_config, resume):
        artifact = cls()
        artifact.spec = spec
        artifact.run_dir = output_root / spec.phase / f"seed_{spec.seed}" / spec.method
        artifact.events = []
        artifact.predictions = []
        artifact.closed = False
        cls.opened.append((artifact, resume, expanded_config))
        cls.trace.append(("open", spec.method, resume))
        return artifact

    def ensure_common_state(self, state):
        snapshot = OrderedDict((key, value.clone()) for key, value in state.items())
        self.trace.append(("ensure-common", self.spec.method, snapshot))
        return self.run_dir.parent / "common_state.pt"

    def append_event(self, *, completed_updates, event, details):
        if event == "failure" and self.fail_failure_event:
            raise RuntimeError("journal failure")
        self.events.append((completed_updates, event, details))
        self.trace.append(("event", self.spec.method, completed_updates, event))

    def checkpoint_path(self, update):
        self.trace.append(("checkpoint-path", self.spec.method, update))
        return self.run_dir / "checkpoints" / f"update_{update:04d}.pt"

    def publish_predictions(self, *, partition, completed_updates, rows):
        snapshot = tuple(rows)
        self.predictions.append((partition, completed_updates, snapshot))
        self.trace.append(("publish", self.spec.method, partition, completed_updates))
        parent = self.run_dir / "predictions" / partition
        parent.mkdir(parents=True, exist_ok=True)
        if partition == "base_test":
            path = parent / f"{self.spec.method}_seed_{self.spec.seed}.jsonl"
        else:
            path = parent / f"update_{completed_updates:04d}.jsonl"
        if path.exists() or path.is_symlink():
            raise RuntimeError("prediction target already exists")
        content = "".join(json.dumps({
            "protocol_version": "PUB-R1.1", "method": self.spec.method,
            "seed": self.spec.seed, "question_index": row.question_index,
            "label": row.label, "logits": list(row.logits),
        }, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
                          for row in snapshot)
        path.write_text(content, encoding="utf-8")
        return path

    def close(self):
        self.closed = True
        self.trace.append(("close", self.spec.method))


class ProductionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module("src.clevr_runner")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.audit = self.root / "audit"
        self.data = self.root / "data"
        self.output = self.root / "output"
        for path in (self.audit, self.data, self.output):
            path.mkdir()
        self.audit_contents = {
            "selected_questions.jsonl": "alpha\n",
            "vocabulary.json": "vocab\n",
            "audit_report.json": "audit\n",
        }
        for name, content in self.audit_contents.items():
            (self.audit / name).write_text(content, encoding="utf-8")
        self.train_rgb = _OrchestrationDataset("train-rgb")
        self.dev_rgb = _OrchestrationDataset("dev-rgb", 19)
        self.train_q = _OrchestrationDataset("train-q")
        self.dev_q = _OrchestrationDataset("dev-q", 19)
        _OrchestrationFactory.instances = []
        _OrchestrationFactory.trace = []
        _OrchestrationArtifacts.opened = []
        _OrchestrationArtifacts.trace = _OrchestrationFactory.trace
        _OrchestrationArtifacts.fail_failure_event = False
        self.saved = []
        self.loaded = []
        self.update_counts = {}
        self.eval_calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def _preflight(self, phase, methods, seed):
        specs = {method: self.runner.CLEVRRunSpec(phase, method, seed) for method in methods}
        train = {method: self.train_q if method == "Q-only" else self.train_rgb
                 for method in methods}
        dev = {method: self.dev_q if method == "Q-only" else self.dev_rgb
               for method in methods}
        return self.runner.CLEVRRunPreflight(methods, specs, train, dev, 17)

    def _save(self, path, **kwargs):
        self.assertTrue(all(parameter.grad is None for parameter in kwargs["model"].parameters()))
        expected = kwargs["completed_updates"]
        self.assertEqual(kwargs["sampler"].completed, expected)
        self.saved.append((path, kwargs))
        return SimpleNamespace(completed_updates=expected)

    def _load(self, path, **kwargs):
        completed = 7999 if "formal" in str(path) else 799
        kwargs["sampler"].completed = completed
        metadata = SimpleNamespace(
            path=str(path), method=kwargs["expected_method"],
            run_seed=kwargs["expected_run_seed"], completed_updates=completed,
            dataset_size=kwargs["sampler"].dataset_size,
            cuda_devices=kwargs["cuda_devices"],
        )
        self.loaded.append((path, kwargs, metadata))
        _OrchestrationFactory.trace.append(("load", kwargs["expected_method"], completed))
        return metadata

    def _update(self, **kwargs):
        method = kwargs["method"]
        count = self.update_counts.get(method, kwargs["sampler"].completed) + 1
        self.update_counts[method] = count
        kwargs["sampler"].completed = count
        self.assertIs(
            kwargs["collate_fn"],
            collate_clevr_question_only if method == "Q-only" else collate_clevr_questions,
        )
        identity = 10 if method == "Q-only" else 20
        return self.runner.CLEVRUpdateResult(
            tuple(range(128)), tuple(range(identity, identity + 128)),
            64.0, 0.5, 0.75, 3e-4,
        )

    def _evaluate(self, **kwargs):
        self.eval_calls.append(kwargs)
        partition = kwargs["partition"]
        dataset = kwargs["dataset"]
        _OrchestrationFactory.trace.append(("evaluate", kwargs["method"], partition))
        rows = tuple(
            self.runner.CLEVRPredictionRecord(
                dataset.sample_identity(index).question_index,
                index % 2,
                (2.0, 0.0) if index % 2 == 0 else (0.0, 2.0),
            )
            for index in range(len(dataset))
        )
        return self.runner.CLEVREvaluationResult(
            partition, len(dataset), len(dataset), 1.0, 1.0 / len(dataset), 1.0,
            rows,
        )

    def _patches(
        self, *, phase, methods, seed, load=None, update=None,
        use_real_config=False, artifact_class=None,
    ):
        preflight = self._preflight(phase, methods, seed)
        stack = ExitStack()
        stack.enter_context(patch.object(self.runner, "preflight_clevr_run", return_value=preflight))
        stack.enter_context(patch.object(self.runner, "_setup_clevr_runtime", return_value=torch.device("cpu")))
        self.config_mock = None
        if not use_real_config:
            self.config_mock = stack.enter_context(
                patch.object(self.runner, "_expanded_clevr_config", return_value={"frozen": True})
            )
        stack.enter_context(patch.object(self.runner, "CLEVRPairedInitialization", _OrchestrationFactory))
        stack.enter_context(patch.object(
            self.runner, "CLEVRRunArtifacts", artifact_class or _OrchestrationArtifacts,
        ))
        stack.enter_context(patch.object(self.runner, "build_clevr_adamw", side_effect=lambda model: SimpleNamespace(model=model)))
        stack.enter_context(patch.object(self.runner, "build_clevr_lr_scheduler", side_effect=lambda optimizer: SimpleNamespace(optimizer=optimizer)))
        stack.enter_context(patch.object(self.runner, "CLEVRIndexStream", _OrchestrationSampler))
        stack.enter_context(patch.object(self.runner, "save_clevr_checkpoint", side_effect=self._save))
        stack.enter_context(patch.object(self.runner, "load_clevr_checkpoint", side_effect=load or self._load))
        stack.enter_context(patch.object(self.runner, "run_clevr_update", side_effect=update or self._update))
        stack.enter_context(patch.object(self.runner, "evaluate_clevr_model", side_effect=self._evaluate))
        stack.enter_context(patch.object(self.runner.random, "seed", side_effect=lambda seed: _OrchestrationFactory.trace.append(("python-seed", seed))))
        stack.enter_context(patch.object(self.runner.np.random, "seed", side_effect=lambda seed: _OrchestrationFactory.trace.append(("numpy-seed", seed))))
        cpu_generator = SimpleNamespace(
            manual_seed=lambda seed: _OrchestrationFactory.trace.append(("torch-seed", seed))
        )
        stack.enter_context(patch.object(self.runner.torch.random, "default_generator", cpu_generator))
        stack.enter_context(patch.object(
            self.runner.torch, "manual_seed",
            side_effect=AssertionError("fresh arm must not seed every CUDA device"),
        ))
        stack.enter_context(patch.object(self.runner.torch.cuda, "manual_seed", side_effect=lambda seed: _OrchestrationFactory.trace.append(("cuda-seed", seed))))
        return stack

    def _train(self, *, phase, seed, methods, resumes):
        return self.runner.train_clevr_seed(
            spec_phase=phase, seed=seed, methods=methods, audit_dir=self.audit,
            data_root=None if methods == ("Q-only",) else self.data,
            output_root=self.output, device_index=0, resume_checkpoints=resumes,
        )

    def _checkpoint(self, *, phase, seed, method, update):
        path = (self.output / phase / f"seed_{seed}" / method /
                "checkpoints" / f"update_{update:04d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"injected")
        return path

    @staticmethod
    def _load_at(update, *, method, seed):
        def load(path, **kwargs):
            kwargs["sampler"].completed = update
            return SimpleNamespace(
                method=method, run_seed=seed, completed_updates=update,
                dataset_size=kwargs["sampler"].dataset_size,
                cuda_devices=kwargs["cuda_devices"],
            )
        return load

    def _write_prediction(self, *, phase, seed, method, partition, update, dataset):
        path = (self.output / phase / f"seed_{seed}" / method /
                "predictions" / partition / f"update_{update:04d}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(len(dataset)):
            label = index % 2
            rows.append({
                "protocol_version": "PUB-R1.1", "method": method, "seed": seed,
                "question_index": dataset.sample_identity(index).question_index,
                "label": label,
                "logits": [2.0, 0.0] if label == 0 else [0.0, 2.0],
            })
        path.write_text("".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ), encoding="utf-8")
        return path

    def test_fresh_seed_uses_one_factory_reseeds_each_arm_and_commits_update_zero_then_final(self):
        methods = ("G", "Q-only")
        with self._patches(phase="smoke", methods=methods, seed=9001):
            self._train(phase="smoke", seed=9001, methods=methods, resumes={})
        self.assertEqual(len(_OrchestrationFactory.instances), 1)
        self.assertEqual(_OrchestrationFactory.instances[0].builds, ["G", "Q-only"])
        for method in methods:
            trace = _OrchestrationFactory.trace
            build = trace.index(("build", method))
            ensure = next(i for i, event in enumerate(trace)
                          if event[:2] == ("ensure-common", method))
            update = next(i for i, event in enumerate(trace)
                          if event[:4] == ("event", method, 1, "update"))
            final_checkpoint = trace.index(("checkpoint-path", method, 1))
            self.assertLess(ensure, update)
            self.assertLess(build, update)
            self.assertLess(update, final_checkpoint)
        seed_events = [event for event in _OrchestrationFactory.trace
                       if isinstance(event, tuple) and event[0].endswith("seed")]
        self.assertEqual(seed_events, [
            ("python-seed", 9001), ("numpy-seed", 9001),
            ("torch-seed", 9001), ("cuda-seed", 9001),
        ] * 2)
        self.assertEqual(
            [(entry[1]["method"], entry[1]["completed_updates"]) for entry in self.saved],
            [("G", 0), ("G", 1), ("Q-only", 0), ("Q-only", 1)],
        )
        self.assertEqual(self.config_mock.call_args_list, [
            call(device_index=0, audit_dir=self.audit, data_root=self.data,
                 vocab_size=17, train_size=257,
                 audit_artifacts=self.audit_contents),
            call(device_index=0, audit_dir=self.audit, data_root=None,
                 vocab_size=17, train_size=257,
                 audit_artifacts=self.audit_contents),
        ])
        for artifact, resume, _ in _OrchestrationArtifacts.opened:
            self.assertFalse(resume)
            self.assertEqual([event[1] for event in artifact.events],
                             ["start", "update", "complete"])
            self.assertTrue(artifact.closed)
        self.assertFalse(self.eval_calls)

    def test_fresh_pilot_evaluates_dev_only_at_800_and_publishes_summary(self):
        with self._patches(phase="pilot", methods=("G",), seed=9001):
            self._train(phase="pilot", seed=9001, methods=("G",), resumes={})
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual(self.update_counts, {"G": 800})
        self.assertEqual([entry[1]["completed_updates"] for entry in self.saved], [0, 800])
        self.assertEqual([call["partition"] for call in self.eval_calls], ["dev"])
        self.assertEqual([(p, u) for p, u, _ in artifact.predictions], [("dev", 800)])
        self.assertEqual(artifact.events[-2][1:], ("evaluation", artifact.events[-2][2]))
        self.assertEqual(artifact.events[-1][1], "complete")

    def test_fresh_training_accepts_an_explicit_absolute_new_output_root(self):
        self.output = self.root / "new-output"
        self.assertFalse(self.output.exists())
        with self._patches(phase="smoke", methods=("G",), seed=9001):
            self._train(phase="smoke", seed=9001, methods=("G",), resumes={})
        self.assertEqual(len(_OrchestrationArtifacts.opened), 1)

    def test_resume_constructs_then_loads_and_runs_8000_without_update_zero(self):
        checkpoint = self._checkpoint(
            phase="formal", seed=101, method="B", update=7999,
        )
        with self._patches(phase="formal", methods=("B",), seed=101):
            self._train(phase="formal", seed=101, methods=("B",), resumes={"B": checkpoint})
        trace = _OrchestrationFactory.trace
        self.assertLess(trace.index(("build", "B")), trace.index(("load", "B", 7999)))
        self.assertEqual(len(self.loaded), 1)
        self.assertEqual(self.update_counts, {"B": 8000})
        self.assertEqual([entry[1]["completed_updates"] for entry in self.saved], [8000])
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertTrue(_OrchestrationArtifacts.opened[0][1])
        self.assertEqual([event[1] for event in artifact.events],
                         ["resume", "update", "evaluation", "evaluation", "complete"])
        self.assertEqual([call["partition"] for call in self.eval_calls], ["dev", "train"])
        self.assertEqual([(p, u) for p, u, _ in artifact.predictions],
                         [("dev", 8000), ("train", 8000)])
        self.assertFalse(any(event[0] == "checkpoint-path" and event[2] == 0 for event in trace))

    def test_resume_at_target_runs_no_update_and_finishes_required_evaluation(self):
        checkpoint = self._checkpoint(
            phase="pilot", seed=9001, method="G", update=800,
        )
        def load_target(path, **kwargs):
            kwargs["sampler"].completed = 800
            return SimpleNamespace(
                method="G", run_seed=9001, completed_updates=800,
                dataset_size=kwargs["sampler"].dataset_size,
                cuda_devices=kwargs["cuda_devices"],
            )
        def forbidden_update(**kwargs):
            raise AssertionError("target checkpoint must not run another update")
        with self._patches(
            phase="pilot", methods=("G",), seed=9001,
            load=load_target, update=forbidden_update,
        ):
            self._train(
                phase="pilot", seed=9001, methods=("G",),
                resumes={"G": checkpoint},
            )
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertFalse(self.saved)
        self.assertEqual([call["partition"] for call in self.eval_calls], ["dev"])
        self.assertEqual([event[1] for event in artifact.events],
                         ["resume", "evaluation", "complete"])

    def test_resume_intermediate_checkpoint_completes_missing_dev_before_next_update(self):
        checkpoint = self._checkpoint(
            phase="formal", seed=101, method="G", update=1000,
        )
        def load_boundary(path, **kwargs):
            kwargs["sampler"].completed = 1000
            return SimpleNamespace(
                method="G", run_seed=101, completed_updates=1000,
                dataset_size=kwargs["sampler"].dataset_size,
                cuda_devices=kwargs["cuda_devices"],
            )
        def stop_after_boundary(**kwargs):
            _OrchestrationFactory.trace.append(("update-attempt", "G", 1001))
            raise RuntimeError("stop after recovered boundary")
        with self._patches(
            phase="formal", methods=("G",), seed=101,
            load=load_boundary, update=stop_after_boundary,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after recovered boundary"):
                self._train(
                    phase="formal", seed=101, methods=("G",),
                    resumes={"G": checkpoint},
                )
        self.assertEqual([call["partition"] for call in self.eval_calls], ["dev"])
        self.assertLess(
            _OrchestrationFactory.trace.index(("evaluate", "G", "dev")),
            _OrchestrationFactory.trace.index(("update-attempt", "G", 1001)),
        )
        self.assertTrue((self.output / "formal" / "seed_101" / "G" /
                         "predictions" / "dev" / "update_1000.jsonl").is_file())

    def test_resume_final_before_dev_completes_both_evaluations(self):
        checkpoint = self._checkpoint(
            phase="formal", seed=101, method="G", update=8000,
        )
        with self._patches(
            phase="formal", methods=("G",), seed=101,
            load=self._load_at(8000, method="G", seed=101),
        ):
            self._train(
                phase="formal", seed=101, methods=("G",),
                resumes={"G": checkpoint},
            )
        self.assertEqual([call["partition"] for call in self.eval_calls],
                         ["dev", "train"])
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual([event[1] for event in artifact.events],
                         ["resume", "evaluation", "evaluation", "complete"])

    def test_resume_final_between_dev_and_train_reuses_dev_then_completes_train(self):
        checkpoint = self._checkpoint(
            phase="formal", seed=101, method="G", update=8000,
        )
        self._write_prediction(
            phase="formal", seed=101, method="G", partition="dev", update=8000,
            dataset=self.dev_rgb,
        )
        with self._patches(
            phase="formal", methods=("G",), seed=101,
            load=self._load_at(8000, method="G", seed=101),
        ):
            self._train(
                phase="formal", seed=101, methods=("G",),
                resumes={"G": checkpoint},
            )
        self.assertEqual([call["partition"] for call in self.eval_calls], ["train"])
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual([event[1] for event in artifact.events],
                         ["resume", "evaluation", "evaluation", "complete"])

    def test_resume_final_after_both_predictions_only_appends_completion(self):
        checkpoint = self._checkpoint(
            phase="formal", seed=101, method="G", update=8000,
        )
        for partition, dataset in (("dev", self.dev_rgb), ("train", self.train_rgb)):
            self._write_prediction(
                phase="formal", seed=101, method="G", partition=partition,
                update=8000, dataset=dataset,
            )
        with self._patches(
            phase="formal", methods=("G",), seed=101,
            load=self._load_at(8000, method="G", seed=101),
        ):
            self._train(
                phase="formal", seed=101, methods=("G",),
                resumes={"G": checkpoint},
            )
        self.assertFalse(self.eval_calls)
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual([event[1] for event in artifact.events],
                         ["resume", "evaluation", "evaluation", "complete"])

    def test_resume_pilot_with_published_dev_is_idempotent(self):
        checkpoint = self._checkpoint(
            phase="pilot", seed=9001, method="G", update=800,
        )
        self._write_prediction(
            phase="pilot", seed=9001, method="G", partition="dev", update=800,
            dataset=self.dev_rgb,
        )
        with self._patches(
            phase="pilot", methods=("G",), seed=9001,
            load=self._load_at(800, method="G", seed=9001),
        ):
            self._train(
                phase="pilot", seed=9001, methods=("G",),
                resumes={"G": checkpoint},
            )
        self.assertFalse(self.eval_calls)
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual([event[1] for event in artifact.events],
                         ["resume", "evaluation", "complete"])

    def test_publish_commit_then_failure_propagates_and_later_resume_reuses_file(self):
        publication_failure = RuntimeError("post-commit publication failure")

        class CommitThenFailArtifacts(_OrchestrationArtifacts):
            def publish_predictions(self, **kwargs):
                super().publish_predictions(**kwargs)
                raise publication_failure

        with self._patches(
            phase="pilot", methods=("G",), seed=9001,
            artifact_class=CommitThenFailArtifacts,
        ):
            with self.assertRaises(RuntimeError) as caught:
                self._train(
                    phase="pilot", seed=9001, methods=("G",), resumes={},
                )
        self.assertIs(caught.exception, publication_failure)
        first_attempt = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual(first_attempt.events[-1][1], "failure")
        self.assertNotIn("complete", [event[1] for event in first_attempt.events])
        prediction = (self.output / "pilot" / "seed_9001" / "G" /
                      "predictions" / "dev" / "update_0800.jsonl")
        self.assertTrue(prediction.is_file())

        checkpoint = self._checkpoint(
            phase="pilot", seed=9001, method="G", update=800,
        )
        with self._patches(
            phase="pilot", methods=("G",), seed=9001,
            load=self._load_at(800, method="G", seed=9001),
            artifact_class=CommitThenFailArtifacts,
        ):
            self._train(
                phase="pilot", seed=9001, methods=("G",),
                resumes={"G": checkpoint},
            )
        self.assertEqual(len(self.eval_calls), 1)
        second_attempt = _OrchestrationArtifacts.opened[1][0]
        self.assertFalse(second_attempt.predictions)
        self.assertEqual([event[1] for event in second_attempt.events],
                         ["resume", "evaluation", "complete"])

    def test_resume_rejects_published_prediction_with_changed_label(self):
        checkpoint = self._checkpoint(
            phase="pilot", seed=9001, method="G", update=800,
        )
        path = self._write_prediction(
            phase="pilot", seed=9001, method="G", partition="dev", update=800,
            dataset=self.dev_rgb,
        )
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["label"] = 1 - rows[0]["label"]
        path.write_text("".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ), encoding="utf-8")
        update = Mock(side_effect=AssertionError("invalid boundary must not update"))
        with self._patches(
            phase="pilot", methods=("G",), seed=9001,
            load=self._load_at(800, method="G", seed=9001), update=update,
        ):
            with self.assertRaisesRegex(
                self.runner.CLEVRRunnerError,
                "prediction identities or labels disagree",
            ):
                self._train(
                    phase="pilot", seed=9001, methods=("G",),
                    resumes={"G": checkpoint},
                )
        self.assertFalse(self.eval_calls)
        update.assert_not_called()

    def test_resume_rejects_smoke_checkpoint_for_pilot_before_load_or_update(self):
        checkpoint = self._checkpoint(
            phase="smoke", seed=9001, method="G", update=1,
        )
        load = Mock(side_effect=AssertionError("foreign checkpoint must not load"))
        update = Mock(side_effect=AssertionError("foreign checkpoint must not update"))
        with self._patches(
            phase="pilot", methods=("G",), seed=9001, load=load, update=update,
        ):
            with self.assertRaises(self.runner.CLEVRRunnerError):
                self._train(
                    phase="pilot", seed=9001, methods=("G",),
                    resumes={"G": checkpoint},
                )
        load.assert_not_called()
        update.assert_not_called()

    def test_resume_rejects_pilot_checkpoint_for_smoke_before_load_or_update(self):
        checkpoint = self._checkpoint(
            phase="pilot", seed=9001, method="G", update=0,
        )
        load = Mock(side_effect=AssertionError("foreign checkpoint must not load"))
        update = Mock(side_effect=AssertionError("foreign checkpoint must not update"))
        with self._patches(
            phase="smoke", methods=("G",), seed=9001, load=load, update=update,
        ):
            with self.assertRaises(self.runner.CLEVRRunnerError):
                self._train(
                    phase="smoke", seed=9001, methods=("G",),
                    resumes={"G": checkpoint},
                )
        load.assert_not_called()
        update.assert_not_called()

    def test_resume_rejects_same_size_audit_mutation_before_checkpoint_load(self):
        with patch.object(self.runner.torch.cuda, "get_device_name", return_value="GPU"), \
                self._patches(
                    phase="smoke", methods=("G",), seed=9001,
                    use_real_config=True, artifact_class=CLEVRRunArtifacts,
                ):
            self._train(phase="smoke", seed=9001, methods=("G",), resumes={})
        checkpoint = self._checkpoint(
            phase="smoke", seed=9001, method="G", update=1,
        )
        manifest = self.audit / "selected_questions.jsonl"
        original_size = manifest.stat().st_size
        manifest.write_text("omega\n", encoding="utf-8")
        self.assertEqual(manifest.stat().st_size, original_size)
        self.assertEqual(len(self.train_rgb), 257)
        self.assertEqual(self.train_rgb.vocab_size, 17)
        load = Mock(side_effect=AssertionError("mismatched data must not load"))
        update = Mock(side_effect=AssertionError("mismatched data must not update"))
        with patch.object(self.runner.torch.cuda, "get_device_name", return_value="GPU"), \
                self._patches(
                    phase="smoke", methods=("G",), seed=9001, load=load,
                    update=update, use_real_config=True,
                    artifact_class=CLEVRRunArtifacts,
                ):
            with self.assertRaisesRegex(CLEVRRunArtifactError, "resume config disagrees"):
                self._train(
                    phase="smoke", seed=9001, methods=("G",),
                    resumes={"G": checkpoint},
                )
        load.assert_not_called()
        update.assert_not_called()

    def test_resume_mapping_and_metadata_mismatch_fail_before_update(self):
        checkpoint = self._checkpoint(
            phase="pilot", seed=9001, method="G", update=799,
        )
        cases = (
            {"G": Path("relative.pt")},
            {"unknown": checkpoint},
            {"G": self.root / "missing.pt"},
        )
        for resumes in cases:
            with self.subTest(resumes=resumes), \
                    self._patches(phase="pilot", methods=("G",), seed=9001):
                with self.assertRaises(self.runner.CLEVRRunnerError):
                    self._train(phase="pilot", seed=9001, methods=("G",), resumes=resumes)
        def mismatched(path, **kwargs):
            return SimpleNamespace(method="G", run_seed=9001, completed_updates=801,
                                   dataset_size=257, cuda_devices=(0,))
        load = Mock(side_effect=mismatched)
        with self._patches(phase="pilot", methods=("G",), seed=9001, load=load):
            with self.assertRaises(self.runner.CLEVRRunnerError):
                self._train(phase="pilot", seed=9001, methods=("G",),
                            resumes={"G": checkpoint})
        load.assert_called_once()
        self.assertFalse(self.update_counts)

    def test_failure_event_is_best_effort_and_primary_baseexception_is_reraised(self):
        primary = KeyboardInterrupt("stop now")
        def fail_update(**kwargs):
            raise primary
        with self._patches(phase="smoke", methods=("G",), seed=9001, update=fail_update):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self._train(phase="smoke", seed=9001, methods=("G",), resumes={})
        self.assertIs(caught.exception, primary)
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertEqual(artifact.events[-1][1], "failure")
        self.assertEqual(artifact.events[-1][2]["type"], "KeyboardInterrupt")
        self.assertTrue(artifact.closed)
        _OrchestrationArtifacts.opened.clear()
        _OrchestrationArtifacts.fail_failure_event = True
        primary = RuntimeError("training failed")
        with self._patches(
            phase="smoke", methods=("G",), seed=9001,
            update=lambda **kwargs: (_ for _ in ()).throw(primary),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self._train(phase="smoke", seed=9001, methods=("G",), resumes={})
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("journal failure" in note for note in primary.__notes__))


class ProductionFormalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module("src.clevr_runner")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.audit = self.root / "audit"
        self.data = self.root / "data"
        self.output = self.root / "output"
        for path in (self.audit, self.data, self.output):
            path.mkdir()
        for name, content in {
            "selected_questions.jsonl": "questions\n",
            "vocabulary.json": "vocabulary\n",
            "audit_report.json": "report\n",
        }.items():
            (self.audit / name).write_text(content, encoding="utf-8")
        self.checkpoint = self._checkpoint_for("G")
        self.trace = []
        _OrchestrationFactory.instances = []
        _OrchestrationFactory.trace = self.trace
        _OrchestrationArtifacts.opened = []
        _OrchestrationArtifacts.trace = self.trace
        _OrchestrationArtifacts.fail_failure_event = False
        self.train = _OrchestrationDataset("train", 257)
        self.base = _OrchestrationDataset("base_test", 23)
        self.base_items = 0

    def tearDown(self):
        self.temporary.cleanup()

    def _checkpoint_for(self, method):
        path = (self.output / "formal" / "seed_101" / method /
                "checkpoints" / "update_8000.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"injected")
        return path

    def _dataset(self, *, data_root, audit_dir, partition):
        self.trace.append(("dataset", "rgb", partition))
        if partition == "train":
            return self.train
        if partition == "base_test":
            return self.base
        raise AssertionError(f"forbidden partition {partition}")

    def _q_dataset(self, *, audit_dir, partition):
        self.trace.append(("dataset", "qonly", partition))
        if partition == "train":
            return self.train
        if partition == "base_test":
            return self.base
        raise AssertionError(f"forbidden partition {partition}")

    def _load(self, path, **kwargs):
        self.trace.append(("load", kwargs["expected_method"], 8000))
        return SimpleNamespace(
            method=kwargs["expected_method"], run_seed=kwargs["expected_run_seed"],
            completed_updates=8000, dataset_size=kwargs["sampler"].dataset_size,
            cuda_devices=kwargs["cuda_devices"], path=str(path),
        )

    def _evaluate(self, **kwargs):
        self.trace.append(("evaluate", kwargs["method"], kwargs["partition"]))
        self.assertIs(kwargs["dataset"], self.base)
        return self.runner.CLEVREvaluationResult(
            "base_test", 1, 1, 0.25, 0.25, 1.0,
            (self.runner.CLEVRPredictionRecord(8, 1, (0.0, 2.0)),),
        )

    def _patches(
        self, *, setup=None, load=None, artifact_class=None,
        use_real_config=False,
    ):
        stack = ExitStack()
        stack.enter_context(patch.object(self.runner, "CLEVRQuestionDataset", side_effect=self._dataset))
        stack.enter_context(patch.object(self.runner, "CLEVRQuestionOnlyDataset", side_effect=self._q_dataset))
        stack.enter_context(patch.object(self.runner, "_setup_clevr_runtime",
                                         side_effect=setup, return_value=torch.device("cpu")))
        if not use_real_config:
            stack.enter_context(patch.object(
                self.runner, "_expanded_clevr_config", return_value={"frozen": True},
            ))
        stack.enter_context(patch.object(self.runner, "CLEVRPairedInitialization", _OrchestrationFactory))
        stack.enter_context(patch.object(self.runner, "CLEVRRunArtifacts", artifact_class or _OrchestrationArtifacts))
        stack.enter_context(patch.object(self.runner, "build_clevr_adamw", side_effect=lambda model: SimpleNamespace(model=model)))
        stack.enter_context(patch.object(self.runner, "build_clevr_lr_scheduler", side_effect=lambda optimizer: SimpleNamespace(optimizer=optimizer)))
        stack.enter_context(patch.object(self.runner, "CLEVRIndexStream", _OrchestrationSampler))
        stack.enter_context(patch.object(self.runner, "load_clevr_checkpoint", side_effect=load or self._load))
        stack.enter_context(patch.object(self.runner, "evaluate_clevr_model", side_effect=self._evaluate))
        return stack

    def _run(self, **overrides):
        arguments = dict(
            spec_phase="formal", seed=101, method="G", audit_dir=self.audit,
            data_root=self.data, output_root=self.output, device_index=0,
            checkpoint=self.checkpoint, partition="base_test",
        )
        arguments.update(overrides)
        if ("checkpoint" not in overrides
                and arguments["method"] in ("G", "B", "M10", "Q-only")):
            arguments["checkpoint"] = self._checkpoint_for(arguments["method"])
        return self.runner.evaluate_clevr_checkpoint(**arguments)

    def test_verifies_train_state_then_constructs_base_test_evaluates_once_and_publishes(self):
        with self._patches():
            path = self._run()
        self.assertEqual(len(_OrchestrationFactory.instances), 1)
        self.assertEqual(_OrchestrationFactory.instances[0].builds, ["G"])
        self.assertEqual([event for event in self.trace if event[0] == "dataset"],
                         [("dataset", "rgb", "train"),
                          ("dataset", "rgb", "base_test")])
        self.assertLess(self.trace.index(("load", "G", 8000)),
                        self.trace.index(("dataset", "rgb", "base_test")))
        ensure_index = next(i for i, event in enumerate(self.trace)
                            if event[:2] == ("ensure-common", "G"))
        self.assertLess(ensure_index, self.trace.index(("load", "G", 8000)))
        self.assertEqual([event for event in self.trace if event[0] == "evaluate"],
                         [("evaluate", "G", "base_test")])
        artifact = _OrchestrationArtifacts.opened[0][0]
        self.assertTrue(_OrchestrationArtifacts.opened[0][1])
        self.assertEqual([(p, u) for p, u, _ in artifact.predictions],
                         [("base_test", 8000)])
        self.assertEqual(
            path,
            artifact.run_dir / "predictions" / "base_test" / "G_seed_101.jsonl",
        )
        self.assertEqual([event[1] for event in artifact.events],
                         ["evaluation", "complete"])
        self.assertTrue(artifact.closed)

    def test_qonly_builds_only_question_datasets_and_requires_no_data_root(self):
        with self._patches():
            self._run(method="Q-only", data_root=None)
        self.assertEqual([event for event in self.trace if event[0] == "dataset"],
                         [("dataset", "qonly", "train"),
                          ("dataset", "qonly", "base_test")])

    def test_rejects_checkpoint_outside_current_formal_run_before_load(self):
        foreign = self.root / "foreign" / "update_8000.pt"
        foreign.parent.mkdir()
        foreign.write_bytes(b"injected")
        load = Mock(side_effect=AssertionError("foreign checkpoint must not load"))
        with self._patches(load=load), self.assertRaises(self.runner.CLEVRRunnerError):
            self._run(checkpoint=foreign)
        load.assert_not_called()

    def test_evaluation_rejects_same_size_audit_mutation_before_checkpoint_load(self):
        method = "B"
        spec = self.runner.CLEVRRunSpec("formal", method, 101)
        with patch.object(self.runner.torch.cuda, "get_device_name", return_value="GPU"):
            config = self.runner._expanded_clevr_config(
                device_index=0, audit_dir=self.audit, data_root=self.data,
                vocab_size=17, train_size=len(self.train),
            )
        CLEVRRunArtifacts.open(
            output_root=self.output, spec=spec, expanded_config=config, resume=False,
        ).close()
        manifest = self.audit / "selected_questions.jsonl"
        original_size = manifest.stat().st_size
        manifest.write_text("altered!!\n", encoding="utf-8")
        self.assertEqual(manifest.stat().st_size, original_size)
        load = Mock(side_effect=AssertionError("mismatched data must not load"))
        with patch.object(self.runner.torch.cuda, "get_device_name", return_value="GPU"), \
                self._patches(
                    load=load, artifact_class=CLEVRRunArtifacts,
                    use_real_config=True,
                ), self.assertRaisesRegex(
                    CLEVRRunArtifactError, "resume config disagrees",
                ):
            self._run(method=method)
        load.assert_not_called()

    def test_rejects_nonformal_confirmation_bad_paths_and_existing_output_before_base(self):
        cases = (
            {"spec_phase": "pilot", "seed": 9001},
            {"spec_phase": "smoke", "seed": 9001},
            {"seed": 9001},
            {"method": "bad"},
            {"partition": "confirmation"},
            {"checkpoint": Path("relative.pt")},
            {"checkpoint": self.root / "missing.pt"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.trace.clear()
                with self._patches(), self.assertRaises(self.runner.CLEVRRunnerError):
                    self._run(**overrides)
                self.assertFalse(any(event[0] == "dataset" and event[2] == "base_test"
                                     for event in self.trace))
        target = (self.output / "formal" / "seed_101" / "G" /
                  "predictions" / "base_test")
        target.mkdir(parents=True)
        (target / "G_seed_101.jsonl").write_text("existing", encoding="utf-8")
        self.trace.clear()
        with self._patches(), self.assertRaises(self.runner.CLEVRRunnerError):
            self._run()
        self.assertFalse(any(event[0] == "dataset" and event[2] == "base_test"
                             for event in self.trace))

    def test_unknown_cuda_config_common_state_and_checkpoint_mismatch_stop_before_base(self):
        cases = []
        cases.append(("cuda", dict(setup=RuntimeError("unknown cuda"))))
        class BadArtifacts(_OrchestrationArtifacts):
            @classmethod
            def open(cls, **kwargs):
                raise RuntimeError("config mismatch")
        cases.append(("config", dict(artifact_class=BadArtifacts)))
        class BadCommonArtifacts(_OrchestrationArtifacts):
            def ensure_common_state(self, state):
                raise RuntimeError("common-state mismatch")
        cases.append(("common-state", dict(artifact_class=BadCommonArtifacts)))
        def bad_load(path, **kwargs):
            return SimpleNamespace(
                method=kwargs["expected_method"], run_seed=kwargs["expected_run_seed"],
                completed_updates=7999, dataset_size=kwargs["sampler"].dataset_size,
                cuda_devices=kwargs["cuda_devices"],
            )
        cases.append(("checkpoint", dict(load=bad_load)))
        for name, patch_args in cases:
            with self.subTest(name=name):
                self.trace.clear()
                with self._patches(**patch_args), self.assertRaises((RuntimeError, self.runner.CLEVRRunnerError)):
                    self._run()
                self.assertFalse(any(event[0] == "dataset" and event[2] == "base_test"
                                     for event in self.trace))

if __name__ == "__main__":
    unittest.main()
