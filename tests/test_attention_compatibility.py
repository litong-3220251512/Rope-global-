"""CPU regressions for the upstream-compatible differentiable attention."""

import unittest
from unittest.mock import patch

import torch

from src.local_rope import LocalAxialRoPE, build_axial_cis
from src.segmentation import differentiable_attention
from models.vit_rope import RoPEAttention


class AttentionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(824)
        self.attention = RoPEAttention(
            dim=128, num_heads=2, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
        ).cpu().eval()

    def test_forward_matches_unmodified_upstream_for_global_and_local_phases(self):
        for height, width in [(2, 3), (3, 2), (4, 4)]:
            inputs = torch.randn(2, height * width + 1, 128, device="cpu")
            for mode in ["G", "B", "C"]:
                with self.subTest(height=height, width=width, mode=mode):
                    phases = build_axial_cis(height, width, mode=mode, lam=0.37)
                    with torch.no_grad():
                        # This is the actual vendored forward, not the patched method.
                        expected = self.attention(inputs, phases)
                        actual = differentiable_attention(self.attention, inputs, phases)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_only_patch_queries_and_keys_are_rotated(self):
        inputs = torch.randn(2, 7, 128, device="cpu")
        phases = build_axial_cis(2, 3, mode="B", lam=0.7)
        with torch.no_grad():
            qkv = self.attention.qkv(inputs).reshape(2, 7, 3, 2, 64)
            q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            expected_logits = torch.zeros(2, 2, 7, 7)
            expected_logits[:, :, 0, 0] = (
                (q[:, :, 0] * self.attention.scale) * k[:, :, 0]
            ).sum(-1)
            expected_attention = expected_logits.softmax(-1)

        def erase_rotated_patches(q_patch, k_patch, supplied_phases):
            self.assertEqual(q_patch.shape[-2], inputs.shape[1] - 1)
            self.assertIs(supplied_phases, phases)
            torch.testing.assert_close(q_patch, q[:, :, 1:], rtol=0, atol=0)
            torch.testing.assert_close(k_patch, k[:, :, 1:], rtol=0, atol=0)
            return torch.zeros_like(q_patch), torch.zeros_like(k_patch)

        observed_attention = []
        handle = self.attention.attn_drop.register_forward_pre_hook(
            lambda module, arguments: observed_attention.append(arguments[0].detach().clone())
        )
        try:
            with patch("src.segmentation.apply_rotary_emb", side_effect=erase_rotated_patches):
                with torch.no_grad():
                    differentiable_attention(self.attention, inputs, phases)
        finally:
            handle.remove()
        self.assertEqual(len(observed_attention), 1)
        # Erasing all rotated Q/K leaves only the unrotated CLS-to-CLS logit.
        torch.testing.assert_close(observed_attention[0], expected_attention,
                                   rtol=1e-6, atol=1e-7)

    def test_learned_phases_receive_gradients_through_frozen_attention(self):
        for parameter in self.attention.parameters():
            parameter.requires_grad_(False)
        inputs = torch.randn(2, 17, 128, device="cpu")
        position = LocalAxialRoPE(mode="C", lambda_init=0.37).cpu()
        phases = position(4, 4)
        output = differentiable_attention(self.attention, inputs, phases)
        loss_weights = torch.randn_like(output)
        loss = (output * loss_weights).mean()
        loss.backward()

        self.assertIsNotNone(position.rho.grad)
        self.assertTrue(torch.isfinite(position.rho.grad).all())
        self.assertTrue((position.rho.grad.abs() > 1e-9).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.attention.parameters()))

        # Check both learned levels against independent central differences.
        analytical = position.rho.grad.detach().clone()
        initial = position.rho.detach().clone()
        epsilon = 0.002
        for level in range(2):
            losses = []
            with torch.no_grad():
                for sign in [1, -1]:
                    position.rho.copy_(initial)
                    position.rho[level].add_(sign * epsilon)
                    output = differentiable_attention(self.attention, inputs, position(4, 4))
                    losses.append((output * loss_weights).mean())
                position.rho.copy_(initial)
            numerical = (losses[0] - losses[1]) / (2 * epsilon)
            torch.testing.assert_close(analytical[level], numerical, rtol=0.025, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
