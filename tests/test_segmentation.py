import unittest

import torch

from src.segmentation import RoPESegmenter


class SegmentationIntegrationTests(unittest.TestCase):
    def test_zero_reset_preserves_official_backbone_cls(self):
        torch.manual_seed(4)
        model = RoPESegmenter(image_size=64, mode="B").eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = model.backbone.forward_features(image)
            actual = model.forward_tokens(image, lam=0.0)[:, 0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_segmentation_loss_reaches_reset_parameter(self):
        torch.manual_seed(5)
        model = RoPESegmenter(image_size=64, mode="B")
        model.configure_trainable(12)
        image = torch.randn(1, 3, 64, 64)
        target = torch.randint(0, 21, (1, 64, 64))
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 21, 64, 64))
        torch.nn.functional.cross_entropy(logits, target).backward()
        grad = model.position.rho.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_fixed_lambda_survives_trainable_configuration_and_optimizer_step(self):
        torch.manual_seed(6)
        model = RoPESegmenter(image_size=64, mode="B", fixed_lambda=1.0)
        for blocks, head_only in [(12, True), (4, False), (12, False)]:
            model.configure_trainable(blocks, head_only=head_only)
            self.assertEqual(list(model.position.parameters()), [])
            self.assertFalse(model.position.rho.requires_grad)
            self.assertEqual(model.position.lambdas.tolist(), [1.0])
        optimizer = torch.optim.AdamW(model.optimizer_groups())
        fixed = model.position.fixed_lambda
        self.assertFalse(any(fixed is parameter for group in optimizer.param_groups for parameter in group["params"]))
        image = torch.randn(1, 3, 64, 64)
        target = torch.randint(0, 21, (1, 64, 64))
        before_head = model.decoder[-1].weight.detach().clone()
        torch.nn.functional.cross_entropy(model(image), target).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before_head, model.decoder[-1].weight))
        self.assertEqual(model.position.lambdas.tolist(), [1.0])
        self.assertIsNone(model.position.rho.grad)
        self.assertIsNone(fixed.grad)


if __name__ == "__main__":
    unittest.main()
