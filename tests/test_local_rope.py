"""CPU checks against the vendored RoPE-ViT operator and hand-derived phases."""

import ast
import importlib
import math
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def official_functions():
    """Load actual upstream numerical functions without importing timm/models."""
    source = ROOT / "vendor/rope-vit/models/vit_rope.py"
    names = {"init_t_xy", "compute_axial_cis", "reshape_for_broadcast", "apply_rotary_emb"}
    parsed = ast.parse(source.read_text())
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError("The vendored RoPE-ViT numerical API has changed")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    return scope


class LocalRoPETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = official_functions()

    def setUp(self):
        try:
            module = importlib.import_module("src.local_rope")
        except ModuleNotFoundError as exc:
            self.fail(f"The requested local RoPE implementation is missing: {exc}")
        self.build = module.build_axial_cis
        self.module_type = module.LocalAxialRoPE

    def test_global_and_exact_zero_match_upstream_on_square_and_rectangular_grids(self):
        for height, width in [(16, 16), (14, 14), (7, 11)]:
            expected = self.official["compute_axial_cis"](64, width, height, theta=100.0)
            for mode in ["G", "B", "C"]:
                with self.subTest(height=height, width=width, mode=mode):
                    actual = self.build(height, width, head_dim=64, mode=mode, lam=0.0)
                    self.assertTrue(torch.equal(actual, expected))

    def test_low_frequency_allocation_preserves_axis_and_complex_pair_layout(self):
        baseline = self.build(16, 16, mode="G")
        for mode in ["B", "C"]:
            actual = self.build(16, 16, mode=mode, lam=0.7)
            torch.testing.assert_close(actual[:, :8], baseline[:, :8], rtol=0, atol=0)
            torch.testing.assert_close(actual[:, 16:24], baseline[:, 16:24], rtol=0, atol=0)
            # Moving only along x does not change any y phase.
            torch.testing.assert_close(actual[9, 16:], actual[0, 16:], rtol=0, atol=0)
            self.assertGreater((actual[:, 8:16] - baseline[:, 8:16]).abs().max().item(), 0.1)

    def test_same_region_patch_logits_stay_equal_after_nonzero_common_rotation(self):
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(1, 2, 256, 64, generator=generator)
        k = torch.randn(1, 2, 256, 64, generator=generator)
        rotate = self.official["apply_rotary_emb"]
        qg, kg = rotate(q, k, self.build(16, 16, mode="G"))
        for mode in ["B", "C"]:
            qr, kr = rotate(q, k, self.build(16, 16, mode=mode, lam=0.7))
            # (10,10) and (11,11) share both levels, with nonzero origins.
            i, j = 170, 187
            torch.testing.assert_close((qr[:, :, i] * kr[:, :, j]).sum(-1),
                                       (qg[:, :, i] * kg[:, :, j]).sum(-1), atol=8e-6, rtol=2e-6)
            torch.testing.assert_close(qr.square().sum(-1), q.square().sum(-1), atol=2e-5, rtol=2e-6)

    def test_cross_region_score_matches_hand_derived_t1(self):
        q = torch.zeros(1, 1, 256, 64)
        k = torch.zeros_like(q)
        # Plane 8 consists of adjacent real channels 16 and 17.
        q[0, 0, 7, 16:18] = torch.tensor([0.7, -0.3])
        k[0, 0, 8, 16:18] = torch.tensor([0.2, 0.9])
        rotate = self.official["apply_rotary_emb"]
        qg, kg = rotate(q, k, self.build(16, 16, mode="G"))
        qb, kb = rotate(q, k, self.build(16, 16, mode="B", lam=0.7))
        measured = ((qb[0, 0, 7] * kb[0, 0, 8]) - (qg[0, 0, 7] * kg[0, 0, 8])).sum().item()
        a, b = -0.13, -0.69
        epsilon = -0.7 * 0.1 * 8
        expected = -2 * math.hypot(a, b) * math.sin(epsilon / 2) * math.sin(0.1 - math.atan2(b, a) + epsilon / 2)
        self.assertAlmostEqual(measured, expected, places=6)
        self.assertGreater(abs(measured), 0.1)

    def test_nondivisible_rectangular_grid_uses_fractional_origins_without_padding(self):
        actual = self.build(10, 14, mode="C", lam=[0.4, 0.8], dtype=torch.float64)
        # x=4, y=3: level-2 origins are 3.5 and 2.5; plane 9 is level 2.
        x_phase = (4 - 0.8 * 3.5) * 100 ** (-9 / 16)
        y_phase = (3 - 0.8 * 2.5) * 100 ** (-9 / 16)
        self.assertEqual(actual.shape, (140, 32))
        self.assertAlmostEqual(torch.angle(actual[46, 9]).item(), x_phase, places=12)
        self.assertAlmostEqual(torch.angle(actual[46, 25]).item(), y_phase, places=12)
        # Integer-coordinate convention: x=3 remains in the first 3.5-wide region.
        self.assertAlmostEqual(torch.angle(actual[3, 9]).item(), 3 * 100 ** (-9 / 16), places=12)

    def test_both_level_lambdas_have_correct_numerical_gradients(self):
        lam = torch.tensor([0.3, 0.6], dtype=torch.float64, requires_grad=True)
        def phases(values):
            return torch.view_as_real(self.build(5, 7, mode="C", lam=values, dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(phases, (lam,), eps=1e-6, atol=1e-5, rtol=1e-4))
        phases(lam).square().mul(torch.arange(32, dtype=torch.float64)[None, :, None]).sum().backward()
        self.assertTrue(torch.isfinite(lam.grad).all())

    def test_q_and_k_gradients_through_upstream_rotation_match_finite_differences(self):
        phases = self.build(16, 16, mode="B", lam=0.7)
        rotate = self.official["apply_rotary_emb"]
        def score(q_pair, k_pair):
            q = torch.zeros(1, 1, 256, 64)
            k = torch.zeros_like(q)
            q[0, 0, 7, 16:18] = q_pair
            k[0, 0, 8, 16:18] = k_pair
            qr, kr = rotate(q, k, phases)
            return (qr[0, 0, 7] * kr[0, 0, 8]).sum()
        q_pair = torch.tensor([0.7, -0.3], requires_grad=True)
        k_pair = torch.tensor([0.2, 0.9], requires_grad=True)
        gradients = torch.autograd.grad(score(q_pair, k_pair), (q_pair, k_pair))
        for parameter_index in range(2):
            for channel in range(2):
                plus = [q_pair.detach().clone(), k_pair.detach().clone()]
                minus = [value.clone() for value in plus]
                plus[parameter_index][channel] += 0.002
                minus[parameter_index][channel] -= 0.002
                numerical = (score(*plus) - score(*minus)) / 0.004
                torch.testing.assert_close(gradients[parameter_index][channel], numerical,
                                           rtol=5e-4, atol=5e-5)

    def test_shared_rho_learns_and_zero_override_preserves_exact_baseline(self):
        module = self.module_type(head_dim=64, mode="C", lambda_init=0.05)
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 2)
        torch.testing.assert_close(module.lambdas, torch.tensor([0.05, 0.05]))
        phases = module(8, 12)
        loss = phases.real.sum() + 0.37 * phases.imag.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(module.rho.grad).all())
        self.assertTrue((module.rho.grad.abs() > 1e-5).all())
        before = module.rho.detach().clone()
        torch.optim.SGD(module.parameters(), lr=1e-3).step()
        self.assertFalse(torch.equal(before, module.rho))
        expected = self.official["compute_axial_cis"](64, 12, 8)
        self.assertTrue(torch.equal(module(8, 12, lam=0.0), expected))
        baseline = self.module_type(head_dim=64, mode="G")
        self.assertEqual(sum(parameter.numel() for parameter in baseline.parameters()), 0)

    def test_fixed_one_resets_coordinates_exactly_without_sigmoid_parameters(self):
        module = self.module_type(mode="B", fixed_lambda=1.0)
        phases = module(16, 16)
        self.assertEqual(list(module.parameters()), [])
        self.assertEqual(module.lambdas.tolist(), [1.0])
        self.assertFalse(module.lambdas.requires_grad)
        # At x=8 the local coordinate is exactly zero; x=2 and x=10 alias
        # only on the local planes. The global half must still distinguish them.
        self.assertEqual(phases[8, 8].item(), 1 + 0j)
        self.assertTrue(torch.equal(phases[2, 8:16], phases[10, 8:16]))
        self.assertFalse(torch.equal(phases[2, :8], phases[10, :8]))
        self.assertAlmostEqual(torch.angle(phases[8, 8] * phases[7, 8].conj()).item(), -0.7, places=6)
        expected = self.official["compute_axial_cis"](64, 16, 16)
        self.assertTrue(torch.equal(module(16, 16, lam=0.0), expected))
        self.assertEqual(module.lambdas.tolist(), [1.0])

    def test_fixed_values_survive_checkpoint_roundtrip_and_global_mode_stays_global(self):
        module = self.module_type(mode="C", fixed_lambda=1.0)
        restored = self.module_type(mode="C", fixed_lambda=0.0)
        restored.load_state_dict(module.state_dict(), strict=True)
        self.assertEqual(restored.lambdas.tolist(), [1.0, 1.0])
        self.assertTrue(torch.equal(module(10, 14), restored(10, 14)))
        baseline = self.module_type(mode="G", fixed_lambda=1.0)
        self.assertEqual(list(baseline.parameters()), [])
        self.assertEqual(baseline.lambdas.numel(), 0)
        self.assertEqual(baseline.state_dict(), {})
        self.assertTrue(torch.equal(baseline(7, 11), self.official["compute_axial_cis"](64, 11, 7)))

    def test_default_learnable_mode_accepts_legacy_rho_checkpoint_strictly(self):
        module = self.module_type(mode="B")
        legacy = {"rho": torch.tensor([-2.25])}
        module.load_state_dict(legacy, strict=True)
        self.assertEqual(set(module.state_dict()), {"rho"})
        self.assertEqual(len(list(module.parameters())), 1)
        torch.testing.assert_close(module.lambdas, legacy["rho"].sigmoid())

    def test_fixed_lambda_rejects_nonfinite_or_out_of_range_values(self):
        for fixed in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(fixed=fixed), self.assertRaises(ValueError):
                self.module_type(mode="B", fixed_lambda=fixed)

    def test_invalid_layout_and_configuration_are_rejected(self):
        for arguments in [dict(H=0, W=4), dict(H=4, W=4, mode="X"),
                          dict(H=4, W=4, head_dim=62), dict(H=4, W=4, theta=0),
                          dict(H=4, W=4, mode="C", lam=[0.1, 0.2, 0.3])]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.build(**arguments)
        for initial in [0.0, 1.0]:
            with self.assertRaises(ValueError):
                self.module_type(mode="B", lambda_init=initial)


if __name__ == "__main__":
    unittest.main()
