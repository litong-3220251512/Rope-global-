"""CPU-only tests for the frozen PUB-R1.1 training configuration."""

from __future__ import annotations

from collections import OrderedDict
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.clevr_model import CLEVRMixedPosition
from src.clevr_training import (
    CLEVRPairedInitialization,
    CLEVRTrainingConfigurationError,
    build_clevr_adamw,
)


def _common_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value)
        for name, value in model.state_dict().items()
        if not name.startswith("position.")
    )


class CLEVRPairedInitializationTests(unittest.TestCase):
    def test_builds_all_arms_from_one_exact_common_state(self) -> None:
        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        common = initialization.common_state_dict()

        self.assertEqual(initialization.vocab_size, 8)
        self.assertEqual(initialization.run_seed, 101)
        self.assertEqual(initialization.m10_seed, 20101)
        self.assertIsInstance(common, OrderedDict)
        self.assertGreater(len(common), 0)
        self.assertFalse(any(name.startswith("position.") for name in common))

        for mode in ("G", "B", "M10", "Q-only"):
            with self.subTest(mode=mode):
                model_common = _common_state(initialization.build(mode))
                self.assertEqual(tuple(model_common), tuple(common))
                for name, expected in common.items():
                    actual = model_common[name]
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(torch.equal(actual, expected), name)

    def test_m10_uses_exact_independent_seed_outside_common_state(self) -> None:
        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        mixed = initialization.build("M10")
        expected_position = CLEVRMixedPosition(seed=20101)

        self.assertNotIn("position.freqs", initialization.common_state_dict())
        self.assertEqual(mixed.position.freqs.shape, expected_position.freqs.shape)
        self.assertEqual(mixed.position.freqs.dtype, expected_position.freqs.dtype)
        self.assertTrue(torch.equal(mixed.position.freqs, expected_position.freqs))

    def test_snapshots_and_built_models_do_not_alias(self) -> None:
        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        first = initialization.common_state_dict()
        second = initialization.common_state_dict()
        name = "backbone.cls_token"
        original = first[name].clone()

        self.assertNotEqual(first[name].data_ptr(), second[name].data_ptr())
        first[name].add_(1.0)
        self.assertTrue(torch.equal(second[name], original))
        self.assertTrue(
            torch.equal(initialization.common_state_dict()[name], original)
        )

        global_model = initialization.build("G")
        model_tensor = global_model.state_dict()[name]
        self.assertNotEqual(
            model_tensor.data_ptr(),
            initialization.common_state_dict()[name].data_ptr(),
        )
        with torch.no_grad():
            global_model.backbone.cls_token.add_(2.0)

        self.assertTrue(
            torch.equal(initialization.common_state_dict()[name], original)
        )
        local_model = initialization.build("B")
        self.assertTrue(torch.equal(local_model.state_dict()[name], original))
        self.assertNotEqual(
            local_model.state_dict()[name].data_ptr(),
            global_model.state_dict()[name].data_ptr(),
        )

    def test_rejects_invalid_inputs_and_seed_overflow_before_rng_change(
        self,
    ) -> None:
        torch.manual_seed(31)
        before = torch.random.get_rng_state().clone()
        invalid_constructors = (
            {"vocab_size": True, "run_seed": 101},
            {"vocab_size": 8.0, "run_seed": 101},
            {"vocab_size": np.int64(8), "run_seed": 101},
            {"vocab_size": 3, "run_seed": 101},
            {"vocab_size": 8, "run_seed": True},
            {"vocab_size": 8, "run_seed": 101.0},
            {"vocab_size": 8, "run_seed": np.int64(101)},
            {"vocab_size": 8, "run_seed": -1},
            {"vocab_size": 8, "run_seed": (1 << 64) - 20_000},
        )
        for arguments in invalid_constructors:
            with self.subTest(arguments=arguments):
                with self.assertRaises(CLEVRTrainingConfigurationError):
                    CLEVRPairedInitialization(**arguments)  # type: ignore[arg-type]
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

        boundary = CLEVRPairedInitialization(
            vocab_size=8, run_seed=(1 << 64) - 1 - 20_000
        )
        self.assertEqual(boundary.m10_seed, (1 << 64) - 1)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        before_invalid_modes = torch.random.get_rng_state().clone()
        for mode in (None, 1, np.str_("G"), "unknown"):
            with self.subTest(mode=mode):
                with self.assertRaises(CLEVRTrainingConfigurationError):
                    initialization.build(mode)  # type: ignore[arg-type]
        self.assertTrue(
            torch.equal(torch.random.get_rng_state(), before_invalid_modes)
        )

    def test_factory_and_builds_leave_global_rng_streams_unchanged(self) -> None:
        random.seed(41)
        np.random.seed(42)
        torch.random.default_generator.manual_seed(43)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        def unexpected_seed(_seed: int) -> None:
            raise AssertionError("paired initialization touched accelerator RNG")

        with (
            patch.object(torch.cuda, "manual_seed_all", side_effect=unexpected_seed),
            patch.object(torch.mps, "manual_seed", side_effect=unexpected_seed),
            patch.object(torch.xpu, "manual_seed_all", side_effect=unexpected_seed),
        ):
            initialization = CLEVRPairedInitialization(
                vocab_size=8, run_seed=101
            )
            initialization.common_state_dict()
            for mode in ("G", "B", "M10", "Q-only"):
                initialization.build(mode)

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))


class CLEVRAdamWTests(unittest.TestCase):
    def test_groups_cover_every_trainable_parameter_once_with_names(self) -> None:
        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        expected_counts = {
            "G": {"no_decay": 67, "decay": 29},
            "M10": {"no_decay": 68, "decay": 29},
            "Q-only": {"no_decay": 67, "decay": 29},
        }

        for mode, counts in expected_counts.items():
            with self.subTest(mode=mode):
                model = initialization.build(mode)
                optimizer = build_clevr_adamw(model)
                named = OrderedDict(
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                )
                seen_ids: list[int] = []
                groups = {
                    group["group_name"]: group for group in optimizer.param_groups
                }

                self.assertEqual(set(groups), {"decay", "no_decay"})
                for group_name, group in groups.items():
                    parameters = group["params"]
                    parameter_names = group["param_names"]
                    parameter_ids = {id(parameter) for parameter in parameters}
                    self.assertEqual(len(parameters), counts[group_name])
                    self.assertEqual(len(parameters), len(parameter_names))
                    for name, parameter in zip(parameter_names, parameters):
                        self.assertIn(name, named)
                        self.assertIs(parameter, named[name])
                        seen_ids.append(id(parameter))
                    expected_order = tuple(
                        name
                        for name, parameter in named.items()
                        if id(parameter) in parameter_ids
                    )
                    self.assertEqual(tuple(parameter_names), expected_order)

                self.assertEqual(len(seen_ids), len(set(seen_ids)))
                self.assertEqual(
                    set(seen_ids),
                    {id(parameter) for parameter in named.values()},
                )
                saved_groups = {
                    group["group_name"]: group
                    for group in optimizer.state_dict()["param_groups"]
                }
                for group_name in groups:
                    self.assertEqual(
                        tuple(saved_groups[group_name]["param_names"]),
                        tuple(groups[group_name]["param_names"]),
                    )

    def test_protocol_parameter_categories_have_exact_decay_and_hyperparameters(
        self,
    ) -> None:
        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        global_model = initialization.build("G")
        optimizer = build_clevr_adamw(global_model)
        decay_by_name = {
            name: group["weight_decay"]
            for group in optimizer.param_groups
            for name in group["param_names"]
        }

        decay_names = (
            "backbone.patch_embed.proj.weight",
            "backbone.blocks.0.attn.qkv.weight",
            "backbone.blocks.0.attn.proj.weight",
            "backbone.blocks.0.mlp.fc1.weight",
            "backbone.blocks.0.mlp.fc2.weight",
            "backbone.head.weight",
            "question_encoder.embedding.weight",
            "question_encoder.gru.weight_ih_l0",
            "question_encoder.gru.weight_hh_l0",
        )
        no_decay_names = (
            "backbone.cls_token",
            "backbone.patch_embed.proj.bias",
            "backbone.blocks.0.gamma_1",
            "backbone.blocks.0.gamma_2",
            "backbone.blocks.0.norm1.weight",
            "backbone.blocks.0.norm1.bias",
            "backbone.blocks.0.attn.qkv.bias",
            "backbone.blocks.0.attn.proj.bias",
            "backbone.head.bias",
            "question_encoder.gru.bias_ih_l0",
            "question_encoder.gru.bias_hh_l0",
        )
        for name in decay_names:
            with self.subTest(name=name):
                self.assertEqual(decay_by_name[name], 0.05)
        for name in no_decay_names:
            with self.subTest(name=name):
                self.assertEqual(decay_by_name[name], 0.0)
        for name in global_model.state_dict():
            if name.endswith(".bias") or ".bias_" in name:
                with self.subTest(all_bias=name):
                    self.assertEqual(decay_by_name[name], 0.0)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 3e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertEqual(
            {group["weight_decay"] for group in optimizer.param_groups},
            {0.0, 0.05},
        )
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], 3e-4)
            self.assertEqual(group["betas"], (0.9, 0.999))
            self.assertEqual(group["eps"], 1e-8)

        mixed = initialization.build("M10")
        mixed_optimizer = build_clevr_adamw(mixed)
        mixed_decay = {
            name: group["weight_decay"]
            for group in mixed_optimizer.param_groups
            for name in group["param_names"]
        }
        self.assertEqual(mixed_decay["position.freqs"], 0.0)

    def test_qonly_none_gradient_leaves_patch_projection_unchanged_and_untracked(
        self,
    ) -> None:
        initialization = CLEVRPairedInitialization(vocab_size=8, run_seed=101)
        model = initialization.build("Q-only")
        optimizer = build_clevr_adamw(model)
        projection = model.backbone.patch_embed.proj
        weight_before = projection.weight.detach().clone()
        bias_before = projection.bias.detach().clone()
        question_tokens = torch.tensor(((1, 4, 2, 0),), dtype=torch.int64)
        question_length = torch.tensor((3,), dtype=torch.int64)

        logits = model(None, question_tokens, question_length)
        torch.nn.functional.cross_entropy(logits, torch.tensor((1,))).backward()
        self.assertIsNone(projection.weight.grad)
        self.assertIsNone(projection.bias.grad)

        optimizer.step()

        self.assertTrue(torch.equal(projection.weight, weight_before))
        self.assertTrue(torch.equal(projection.bias, bias_before))
        self.assertNotIn(projection.weight, optimizer.state)
        self.assertNotIn(projection.bias, optimizer.state)

    def test_rejects_non_answerer_input(self) -> None:
        for value in (None, object(), torch.nn.Linear(2, 2)):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(CLEVRTrainingConfigurationError):
                    build_clevr_adamw(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
