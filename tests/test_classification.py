"""Scratch classification regressions for shared initialization and attention."""

import unittest
from copy import deepcopy
from types import MethodType

import torch
from torch.nn import functional as F

from src.classification import RoPEClassifier, sdpa_attention
from src.local_rope import build_axial_cis
from src.segmentation import differentiable_attention
from models.vit_rope import RoPEAttention


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261015)

    def test_zero_reset_matches_unmodified_official_classifier(self):
        image = torch.randn(2, 3, 32, 32)
        for mode in ("G", "B", "C"):
            with self.subTest(mode=mode):
                model = RoPEClassifier(mode=mode, depth=2, init_scale=0.1).eval()
                reference = deepcopy(model.backbone).eval()
                for block in reference.blocks:
                    block.attn.forward = MethodType(RoPEAttention.forward, block.attn)
                with torch.no_grad():
                    expected = reference(image)
                    actual = model(image, lam=0.0)
                self.assertEqual(tuple(actual.shape), (2, 100))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_common_state_is_independent_and_strictly_loads_every_non_pe_weight(self):
        global_model = RoPEClassifier(depth=2)
        initial = global_model.shared_state_dict()
        for mode in ("B", "C"):
            with self.subTest(mode=mode):
                local_model = RoPEClassifier(mode=mode, depth=2)
                local_model.load_shared_state_dict(initial)
                self.assertEqual(set(initial), set(local_model.backbone.state_dict()))
                for name, value in local_model.backbone.state_dict().items():
                    torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
                self.assertEqual(list(local_model.position.parameters()), [])
                self.assertEqual(
                    sum(p.numel() for p in local_model.parameters()),
                    sum(p.numel() for p in global_model.parameters()),
                )
                self.assertEqual(local_model.position.lambdas.tolist(),
                                 [1.0] if mode == "B" else [1.0, 1.0])
        with torch.no_grad():
            global_model.backbone.head.weight.add_(1.0)
        self.assertFalse(torch.equal(initial["head.weight"], global_model.backbone.head.weight))

        invalid = dict(initial)
        del invalid["head.weight"]
        with self.assertRaises(RuntimeError):
            global_model.load_shared_state_dict(invalid)
        invalid = dict(initial)
        invalid["unexpected"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            global_model.load_shared_state_dict(invalid)
        with self.assertRaises(RuntimeError):
            RoPEClassifier(depth=2, num_classes=10).load_shared_state_dict(initial)

    def test_training_updates_cls_patch_embed_and_all_blocks_but_not_fixed_lambda(self):
        for backend in ("manual", "sdpa"):
            with self.subTest(backend=backend):
                model = RoPEClassifier(mode="B", depth=2, attention_backend=backend)
                image = torch.randn(3, 3, 32, 32)
                target = torch.tensor([1, 22, 79])
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                before_cls = model.backbone.cls_token.detach().clone()
                before_head = model.backbone.head.weight.detach().clone()
                F.cross_entropy(model(image), target).backward()
                parameters = [model.backbone.cls_token,
                              model.backbone.patch_embed.proj.weight,
                              model.backbone.head.weight]
                parameters.extend(block.attn.qkv.weight for block in model.backbone.blocks)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
                optimizer.step()
                self.assertFalse(torch.equal(before_cls, model.backbone.cls_token))
                self.assertFalse(torch.equal(before_head, model.backbone.head.weight))
                self.assertEqual(model.position.lambdas.tolist(), [1.0])
                self.assertEqual(list(model.position.parameters()), [])

    def test_fixed_reset_changes_classifier_with_identical_non_pe_weights(self):
        image = torch.randn(2, 3, 32, 32)
        global_model = RoPEClassifier(depth=2, init_scale=0.1).eval()
        local_model = RoPEClassifier(mode="B", depth=2, init_scale=0.1).eval()
        local_model.load_shared_state_dict(global_model.shared_state_dict())
        with torch.no_grad():
            global_output = global_model(image)
            local_output = local_model(image)
            reset_zero_output = local_model(image, lam=0.0)
        torch.testing.assert_close(global_output, reset_zero_output, rtol=0, atol=0)
        self.assertGreater((local_output - global_output).abs().max().item(), 1e-6)

    def test_architecture_configuration_and_state_roundtrip(self):
        for width, heads in ((192, 3), (384, 6)):
            with self.subTest(width=width):
                model = RoPEClassifier(mode="C", image_size=32, patch_size=4,
                                       embed_dim=width, depth=2, heads=heads,
                                       num_classes=17).eval()
                restored = RoPEClassifier(mode="C", image_size=32, patch_size=4,
                                          embed_dim=width, depth=2, heads=heads,
                                          num_classes=17).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                image = torch.randn(1, 3, 32, 32)
                with torch.no_grad():
                    expected = model(image)
                    actual = restored(image)
                self.assertEqual(tuple(actual.shape), (1, 17))
                self.assertEqual(len(model.backbone.blocks), 2)
                self.assertEqual(model.backbone.patch_embed.num_patches, 64)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_invalid_architecture_before_training(self):
        for kwargs in ({"image_size": 30, "patch_size": 4}, {"patch_size": 0},
                       {"heads": 0}, {"embed_dim": 193, "heads": 3},
                       {"depth": 0}, {"num_classes": 0},
                       {"mode": "C", "embed_dim": 120, "heads": 3},
                       {"attention_backend": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RoPEClassifier(**kwargs)


class SDPAAttentionTests(unittest.TestCase):
    def compare_backends(self, device):
        torch.manual_seed(824)
        manual = RoPEAttention(dim=128, num_heads=2, qkv_bias=True,
                               attn_drop=0.25, proj_drop=0.0).to(device).eval()
        sdpa = deepcopy(manual)
        inputs = torch.randn(2, 65, 128, device=device)
        weights = torch.randn_like(inputs)
        for mode in ("G", "B", "C"):
            with self.subTest(device=device, mode=mode):
                manual.zero_grad(set_to_none=True)
                sdpa.zero_grad(set_to_none=True)
                manual_inputs = inputs.detach().clone().requires_grad_()
                sdpa_inputs = inputs.detach().clone().requires_grad_()
                manual_lambda = torch.tensor(0.7, device=device, requires_grad=True)
                sdpa_lambda = torch.tensor(0.7, device=device, requires_grad=True)
                manual_output = differentiable_attention(
                    manual, manual_inputs,
                    build_axial_cis(8, 8, mode=mode, lam=manual_lambda, device=device))
                sdpa_output = sdpa_attention(
                    sdpa, sdpa_inputs,
                    build_axial_cis(8, 8, mode=mode, lam=sdpa_lambda, device=device))
                torch.testing.assert_close(sdpa_output, manual_output, rtol=1e-4, atol=2e-6)
                (manual_output * weights).mean().backward()
                (sdpa_output * weights).mean().backward()
                torch.testing.assert_close(sdpa_inputs.grad, manual_inputs.grad,
                                           rtol=2e-4, atol=2e-7)
                for (_, manual_parameter), (_, sdpa_parameter) in zip(
                        manual.named_parameters(), sdpa.named_parameters()):
                    torch.testing.assert_close(sdpa_parameter.grad, manual_parameter.grad,
                                               rtol=2e-4, atol=2e-7)
                if mode != "G":
                    torch.testing.assert_close(sdpa_lambda.grad, manual_lambda.grad,
                                               rtol=2e-4, atol=2e-7)

    def test_cpu_outputs_and_gradients_match_manual_with_eval_dropout_disabled(self):
        self.compare_backends("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_outputs_and_gradients_match_manual_with_eval_dropout_disabled(self):
        self.compare_backends("cuda")

    def test_sdpa_training_attention_dropout_changes_output(self):
        torch.manual_seed(825)
        attention = RoPEAttention(dim=128, num_heads=2, qkv_bias=True,
                                  attn_drop=0.5, proj_drop=0.0)
        inputs = torch.randn(2, 17, 128)
        phases = build_axial_cis(4, 4)
        first = sdpa_attention(attention.train(), inputs, phases)
        second = sdpa_attention(attention, inputs, phases)
        self.assertFalse(torch.equal(first, second))
        first = sdpa_attention(attention.eval(), inputs, phases)
        second = sdpa_attention(attention, inputs, phases)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
