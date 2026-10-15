"""CPU checks for the scratch runner's optimization and pairing contracts."""

import unittest

import torch
from timm.data import Mixup
from torch.nn import functional as F

from run_cifar_scratch import (
    digest_tensors,
    learning_rate,
    optimizer_groups,
    seed_everything,
)
from src.classification import RoPEClassifier


class CIFARTrainingTests(unittest.TestCase):
    def test_learning_rate_warms_up_then_reaches_cosine_minimum_on_final_update(self):
        values = [learning_rate(step, 7, 2, peak=1e-3, minimum=1e-5, warmup_start=1e-6)
                  for step in range(7)]
        self.assertAlmostEqual(values[0], 1e-6)
        self.assertAlmostEqual(values[1], 0.0005005)
        self.assertAlmostEqual(values[2], 0.001)
        self.assertAlmostEqual(values[4], 0.000505)
        self.assertAlmostEqual(values[-1], 1e-5)
        self.assertTrue(all(a < b for a, b in zip(values[:2], values[1:3])))
        self.assertTrue(all(a > b for a, b in zip(values[2:-1], values[3:])))

    def test_learning_rate_without_warmup_starts_at_peak(self):
        self.assertAlmostEqual(learning_rate(0, 5, 0, 1e-3, minimum=1e-5), 1e-3)
        self.assertAlmostEqual(learning_rate(2, 5, 0, 1e-3, minimum=1e-5), 0.000505)
        self.assertAlmostEqual(learning_rate(4, 5, 0, 1e-3, minimum=1e-5), 1e-5)

    def test_recipe_default_learning_rate_endpoints(self):
        self.assertAlmostEqual(learning_rate(0, 10, 2, peak=5e-4), 5e-7, places=12)
        self.assertAlmostEqual(learning_rate(2, 10, 2, peak=5e-4), 5e-4, places=12)
        self.assertAlmostEqual(learning_rate(9, 10, 2, peak=5e-4), 5e-6, places=12)

    def test_optimizer_includes_each_trainable_parameter_once_and_exempts_cls(self):
        model = RoPEClassifier(mode="B", depth=1)
        model.backbone.head.bias.requires_grad_(False)
        groups = optimizer_groups(model, 0.05)
        grouped = [parameter for group in groups for parameter in group["params"]]
        expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual({id(parameter) for parameter in grouped},
                         {id(parameter) for parameter in expected})
        self.assertNotIn(id(model.position.fixed_lambda), {id(parameter) for parameter in grouped})

        before_cls = model.backbone.cls_token.detach().clone()
        before_patch = model.backbone.patch_embed.proj.weight.detach().clone()
        before_norm = model.backbone.norm.weight.detach().clone()
        optimizer = torch.optim.AdamW(groups, lr=0.1)
        for parameter in grouped:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        torch.testing.assert_close(model.backbone.cls_token, before_cls, rtol=0, atol=0)
        torch.testing.assert_close(model.backbone.norm.weight, before_norm, rtol=0, atol=0)
        torch.testing.assert_close(model.backbone.patch_embed.proj.weight,
                                   before_patch * 0.995, rtol=1e-6, atol=1e-8)
        self.assertEqual(model.position.lambdas.tolist(), [1.0])

    def test_shared_weights_and_seed_pair_mixup_dropout_and_optimizer_updates_at_zero_reset(self):
        seed_everything(20261015)
        global_model = RoPEClassifier(depth=2, init_scale=0.1, drop_path_rate=0.2)
        initial = global_model.shared_state_dict()
        local_model = RoPEClassifier(mode="B", depth=2, init_scale=0.1, drop_path_rate=0.2)
        local_model.load_shared_state_dict(initial)
        self.assertEqual(digest_tensors(initial), digest_tensors(local_model.shared_state_dict()))
        image = torch.randn(4, 3, 32, 32)
        target = torch.tensor([0, 31, 55, 99])
        observed = []
        for model in (global_model, local_model):
            seed_everything(20271022)
            optimizer = torch.optim.AdamW(optimizer_groups(model, 0.05), lr=1e-3)
            mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
                          label_smoothing=0.1, num_classes=100)
            mixed_image, mixed_target = mixup(image.clone(), target.clone())
            logits = model.train()(mixed_image, lam=0.0)
            loss = -(mixed_target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            observed.append((mixed_image, mixed_target, logits.detach(), loss.detach()))
        for global_value, local_value in zip(*observed):
            torch.testing.assert_close(global_value, local_value, rtol=0, atol=0)
        self.assertEqual(digest_tensors(global_model.shared_state_dict()),
                         digest_tensors(local_model.shared_state_dict()))
        self.assertNotEqual(digest_tensors(initial), digest_tensors(global_model.shared_state_dict()))


if __name__ == "__main__":
    unittest.main()
