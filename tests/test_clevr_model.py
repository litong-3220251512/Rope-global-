"""Data-free tests for the frozen PUB-R1.1 CLEVR model contract."""

from inspect import signature
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.func import functional_call

from src.clevr_model import (
    CLEVRMixedPosition,
    CLEVRQuestionAnswerer,
    CLEVRQuestionEncoder,
)
from src.local_rope import build_axial_cis
from src.segmentation import differentiable_attention
from models.vit_rope import init_random_2d_freqs


def _valid_answerer_inputs(batch_size=1):
    image = torch.randn(batch_size, 3, 224, 224)
    question_tokens = torch.tensor(
        ((1, 4, 2, 0),) * batch_size, dtype=torch.int64
    )
    question_length = torch.full((batch_size,), 3, dtype=torch.int64)
    return image, question_tokens, question_length


class CLEVRQuestionEncoderTests(unittest.TestCase):
    def test_applies_embedding_exactly_once(self):
        encoder = CLEVRQuestionEncoder(vocab_size=8)
        call_count = 0

        def count_call(_module, _inputs, _output):
            nonlocal call_count
            call_count += 1

        handle = encoder.embedding.register_forward_hook(count_call)
        try:
            encoder(
                torch.tensor(((1, 4, 2),), dtype=torch.int64),
                torch.tensor((3,), dtype=torch.int64),
            )
        finally:
            handle.remove()

        self.assertEqual(call_count, 1)

    def test_applies_gru_exactly_once(self):
        encoder = CLEVRQuestionEncoder(vocab_size=8)
        call_count = 0

        def count_call(_module, _inputs, _output):
            nonlocal call_count
            call_count += 1

        handle = encoder.gru.register_forward_hook(count_call)
        try:
            encoder(
                torch.tensor(((1, 4, 2),), dtype=torch.int64),
                torch.tensor((3,), dtype=torch.int64),
            )
        finally:
            handle.remove()

        self.assertEqual(call_count, 1)

    def test_uses_frozen_embedding_and_gru_dimensions_without_custom_reinitialization(self):
        seed = 20261024
        torch.manual_seed(seed)
        encoder = CLEVRQuestionEncoder(vocab_size=8)

        self.assertEqual(encoder.embedding.num_embeddings, 8)
        self.assertEqual(encoder.embedding.embedding_dim, 64)
        self.assertEqual(encoder.embedding.padding_idx, 0)
        self.assertEqual(encoder.gru.input_size, 64)
        self.assertEqual(encoder.gru.hidden_size, 192)
        self.assertEqual(encoder.gru.num_layers, 1)
        self.assertFalse(encoder.gru.bidirectional)
        self.assertEqual(encoder.gru.dropout, 0.0)

        tokens = torch.tensor(((1, 4, 2, 0), (1, 5, 6, 2)), dtype=torch.int64)
        lengths = torch.tensor((3, 4), dtype=torch.int64)
        self.assertEqual(tuple(encoder(tokens, lengths).shape), (2, 192))

        torch.manual_seed(seed)
        reference_embedding = nn.Embedding(8, 64, padding_idx=0)
        reference_gru = nn.GRU(64, 192, num_layers=1, bidirectional=False)
        for name, state in encoder.embedding.state_dict().items():
            with self.subTest(module="embedding", state=name):
                self.assertTrue(
                    torch.equal(state, reference_embedding.state_dict()[name])
                )
        for name, state in encoder.gru.state_dict().items():
            with self.subTest(module="gru", state=name):
                self.assertTrue(torch.equal(state, reference_gru.state_dict()[name]))

    def test_padded_batch_uses_each_last_valid_token_state(self):
        torch.manual_seed(20261024)
        encoder = CLEVRQuestionEncoder(vocab_size=8).eval()
        with torch.no_grad():
            for parameter in encoder.gru.parameters():
                parameter.zero_()
            encoder.gru.bias_ih_l0[2 * encoder.gru.hidden_size :].fill_(1.0)
        padded_tokens = torch.tensor(
            ((1, 4, 2, 0, 0), (1, 5, 6, 7, 2)), dtype=torch.int64
        )
        lengths = torch.tensor((3, 5), dtype=torch.int64)

        with torch.no_grad():
            batched = encoder(padded_tokens, lengths)
            short_alone = encoder(
                torch.tensor(((1, 4, 2),), dtype=torch.int64),
                torch.tensor((3,), dtype=torch.int64),
            )
            long_alone = encoder(
                torch.tensor(((1, 5, 6, 7, 2),), dtype=torch.int64),
                torch.tensor((5,), dtype=torch.int64),
            )

        self.assertTrue(torch.equal(batched[0], short_alone[0]))
        self.assertTrue(torch.equal(batched[1], long_alone[0]))

    def test_rejects_invalid_vocabulary_and_tensor_contracts(self):
        for vocabulary_size in (True, 4.0, "8"):
            with self.subTest(vocabulary_size=vocabulary_size), self.assertRaises(
                TypeError
            ):
                CLEVRQuestionEncoder(vocabulary_size)
        for vocabulary_size in (-1, 0, 3):
            with self.subTest(vocabulary_size=vocabulary_size), self.assertRaises(
                ValueError
            ):
                CLEVRQuestionEncoder(vocabulary_size)

        encoder = CLEVRQuestionEncoder(vocab_size=8)
        valid_tokens = torch.tensor(((1, 2), (1, 2)), dtype=torch.int64)
        valid_lengths = torch.tensor((2, 2), dtype=torch.int64)
        invalid_calls = (
            ("non-tensor tokens", [[1, 2], [1, 2]], valid_lengths, TypeError),
            ("non-tensor lengths", valid_tokens, [2, 2], TypeError),
            (
                "non-int64 tokens",
                valid_tokens.to(torch.int32),
                valid_lengths,
                TypeError,
            ),
            ("rank-one tokens", valid_tokens[0], valid_lengths, ValueError),
            (
                "rank-three tokens",
                valid_tokens.unsqueeze(0),
                valid_lengths,
                ValueError,
            ),
            (
                "non-int64 lengths",
                valid_tokens,
                valid_lengths.to(torch.int32),
                TypeError,
            ),
            (
                "rank-two lengths",
                valid_tokens,
                valid_lengths.unsqueeze(1),
                ValueError,
            ),
            (
                "batch mismatch",
                valid_tokens,
                valid_lengths[:1],
                ValueError,
            ),
            (
                "sequence width below two",
                valid_tokens[:, :1],
                valid_lengths,
                ValueError,
            ),
        )
        for name, tokens, lengths, exception in invalid_calls:
            with self.subTest(name=name):
                try:
                    encoder(tokens, lengths)
                except Exception as error:
                    self.assertIs(type(error), exception)
                else:
                    self.fail(f"{name} was not rejected")


class CLEVRMixedPositionTests(unittest.TestCase):
    def test_rejects_non_integer_or_negative_seed(self):
        for seed in (True, 1.0, "1"):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                CLEVRMixedPosition(seed=seed)
        with self.assertRaises(ValueError):
            CLEVRMixedPosition(seed=-1)

    def test_initialization_does_not_seed_accelerator_generators(self):
        def unexpected_seed(_seed):
            raise AssertionError("M10 initialization touched an accelerator RNG")

        with (
            patch.object(torch.cuda, "manual_seed_all", side_effect=unexpected_seed),
            patch.object(torch.mps, "manual_seed", side_effect=unexpected_seed),
            patch.object(torch.xpu, "manual_seed_all", side_effect=unexpected_seed),
        ):
            position = CLEVRMixedPosition(seed=20007)

        self.assertEqual(tuple(position.freqs.shape), (2, 6, 96))

    def test_matches_official_flattened_layout_and_base_ten_geometry(self):
        seed = 20007
        position = CLEVRMixedPosition(seed=seed)

        self.assertEqual(tuple(position.freqs.shape), (2, 6, 96))
        self.assertEqual(
            sum(parameter.numel() for parameter in position.parameters()),
            1152,
        )
        self.assertTrue(position.freqs.requires_grad)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expected = torch.stack(
                [
                    init_random_2d_freqs(
                        dim=64, num_heads=3, theta=10.0
                    )
                    for _ in range(6)
                ],
                dim=1,
            ).view(2, 6, 96)
        self.assertTrue(torch.equal(position.freqs, expected))

        semantic = position.freqs.detach().view(2, 6, 3, 32)
        magnitudes = torch.sqrt(semantic[0].square() + semantic[1].square())
        expected_magnitudes = 10.0 ** (
            -torch.arange(16, dtype=magnitudes.dtype) / 16.0
        )
        self.assertTrue(
            torch.allclose(
                magnitudes[..., :16],
                expected_magnitudes.expand(6, 3, -1),
            )
        )
        self.assertTrue(
            torch.allclose(
                magnitudes[..., 16:],
                expected_magnitudes.expand(6, 3, -1),
            )
        )
        dot_products = (
            semantic[0, ..., :16] * semantic[0, ..., 16:]
            + semantic[1, ..., :16] * semantic[1, ..., 16:]
        )
        torch.testing.assert_close(
            dot_products,
            torch.zeros_like(dot_products),
            rtol=1.3e-6,
            atol=1e-5,
        )

    def test_forward_matches_hand_computed_phases_and_relative_invariants(self):
        position = CLEVRMixedPosition(seed=20007)

        phases = position()
        self.assertEqual(tuple(phases.shape), (6, 3, 196, 32))
        self.assertIsNot(phases, position())
        torch.testing.assert_close(
            phases.abs(),
            torch.ones_like(phases.abs()),
            rtol=1.3e-6,
            atol=1e-5,
        )
        self.assertTrue(
            torch.equal(
                position.freqs_t_x[:16],
                torch.tensor(
                    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1),
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                position.freqs_t_y[:16],
                torch.tensor(
                    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
                    dtype=torch.float32,
                ),
            )
        )

        semantic = position.freqs.detach().view(2, 6, 3, 32)
        selected_angle = 2.0 * semantic[0, 0, 0] + 3.0 * semantic[1, 0, 0]
        expected = torch.complex(selected_angle.cos(), selected_angle.sin())
        torch.testing.assert_close(
            phases[0, 0, 44], expected, rtol=1.3e-6, atol=1e-5
        )

        same_patch = phases[0, 0, 44] * phases[0, 0, 44].conj()
        torch.testing.assert_close(
            same_patch,
            torch.ones_like(same_patch),
            rtol=1.3e-6,
            atol=1e-5,
        )
        relative = phases[0, 0, 44] * phases[0, 0, 15].conj()
        translated_relative = phases[0, 0, 118] * phases[0, 0, 89].conj()
        torch.testing.assert_close(
            relative, translated_relative, rtol=1.3e-6, atol=1e-5
        )

    def test_float64_frequency_derivative_passes_gradcheck(self):
        position = CLEVRMixedPosition(seed=20007).double()
        base_frequencies = position.freqs.detach().clone()
        mask = torch.zeros_like(base_frequencies)
        mask[0, 0, 0] = 1.0
        delta = torch.zeros((), dtype=torch.float64, requires_grad=True)

        def selected_phase(real_delta):
            phases = functional_call(
                position,
                {"freqs": base_frequencies + real_delta * mask},
                (),
            )
            return torch.view_as_real(phases[0, 0, (1, 44), 0])

        self.assertTrue(
            torch.autograd.gradcheck(
                selected_phase,
                (delta,),
                eps=1e-6,
                rtol=1e-7,
                atol=1e-7,
            )
        )


class CLEVRQuestionAnswererM10Tests(unittest.TestCase):
    def test_requires_a_dedicated_seed_only_for_m10_and_rejects_lambda(self):
        with self.assertRaises(ValueError):
            CLEVRQuestionAnswerer(vocab_size=8, mode="M10")
        for seed in (True, 1.0, "1"):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                CLEVRQuestionAnswerer(
                    vocab_size=8, mode="M10", m10_seed=seed
                )
        with self.assertRaises(ValueError):
            CLEVRQuestionAnswerer(vocab_size=8, mode="M10", m10_seed=-1)
        for mode in ("G", "B"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                CLEVRQuestionAnswerer(
                    vocab_size=8, mode=mode, m10_seed=20007
                )

        model = CLEVRQuestionAnswerer(
            vocab_size=8, mode="M10", m10_seed=20007
        ).eval()
        self.assertIsInstance(model.position, CLEVRMixedPosition)
        image, question_tokens, question_length = _valid_answerer_inputs()
        with self.assertRaises(ValueError):
            model(
                image,
                question_tokens,
                question_length,
                lam=0.0,
            )

    def test_common_state_and_global_rng_match_g_at_equal_common_seed(self):
        common_seed = 20261025
        torch.manual_seed(common_seed)
        global_model = CLEVRQuestionAnswerer(vocab_size=8, mode="G")
        next_after_global = torch.rand(8)

        torch.manual_seed(common_seed)
        mixed_model = CLEVRQuestionAnswerer(
            vocab_size=8, mode="M10", m10_seed=20007
        )
        next_after_mixed = torch.rand(8)

        global_state = {
            name: value
            for name, value in global_model.state_dict().items()
            if not name.startswith("position.")
        }
        mixed_state = {
            name: value
            for name, value in mixed_model.state_dict().items()
            if not name.startswith("position.")
        }
        self.assertEqual(tuple(global_state), tuple(mixed_state))
        for name, global_value in global_state.items():
            with self.subTest(state=name):
                mixed_value = mixed_state[name]
                self.assertEqual(global_value.shape, mixed_value.shape)
                self.assertEqual(global_value.dtype, mixed_value.dtype)
                self.assertTrue(torch.equal(global_value, mixed_value))
        self.assertTrue(torch.equal(next_after_global, next_after_mixed))

    def test_routes_each_live_mixed_phase_to_its_matching_block(self):
        torch.manual_seed(20261025)
        model = CLEVRQuestionAnswerer(
            vocab_size=8, mode="M10", m10_seed=20007
        ).eval()
        image, question_tokens, question_length = _valid_answerer_inputs()
        captured = []

        def capture_phase(_module, _args, kwargs):
            captured.append(kwargs["freqs_cis"])

        handles = [
            block.register_forward_pre_hook(capture_phase, with_kwargs=True)
            for block in model.backbone.blocks
        ]
        try:
            with torch.no_grad():
                expected = model.position()
                model.forward_tokens(image, question_tokens, question_length)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(captured), 6)
        for block_index, block_phase in enumerate(captured):
            with self.subTest(block=block_index):
                self.assertEqual(tuple(block_phase.shape), (3, 196, 32))
                self.assertTrue(
                    torch.equal(block_phase, expected[block_index])
                )
        self.assertFalse(torch.equal(captured[0], captured[1]))

    def test_answer_loss_reaches_q_k_and_frequencies_across_an_update(self):
        torch.manual_seed(20261025)
        model = CLEVRQuestionAnswerer(
            vocab_size=8, mode="M10", m10_seed=20007
        )
        image, question_tokens, question_length = _valid_answerer_inputs()
        captured = []

        def capture_first_block_phase(_module, _args, kwargs):
            captured.append(kwargs["freqs_cis"])

        handle = model.backbone.blocks[0].register_forward_pre_hook(
            capture_first_block_phase, with_kwargs=True
        )
        try:
            logits = model(image, question_tokens, question_length)
            loss = nn.functional.cross_entropy(
                logits, torch.tensor((1,), dtype=torch.int64)
            )
            loss.backward()

            frequency_gradient = model.position.freqs.grad.detach().clone()
            self.assertTrue(torch.isfinite(frequency_gradient).all())
            self.assertGreater(frequency_gradient.abs().sum().item(), 0.0)
            for block_index, block in enumerate(model.backbone.blocks):
                gradient = block.attn.qkv.weight.grad
                self.assertIsNotNone(gradient)
                for projection, rows in (
                    ("q", slice(0, 192)),
                    ("k", slice(192, 384)),
                ):
                    with self.subTest(
                        block=block_index, projection=projection
                    ):
                        selected = gradient[rows]
                        self.assertTrue(torch.isfinite(selected).all())
                        self.assertGreater(selected.abs().sum().item(), 0.0)

            with torch.no_grad():
                model.position.freqs[0, 0, 0].add_(0.125)
            model.zero_grad(set_to_none=True)

            second_logits = model(image, question_tokens, question_length)
            second_loss = nn.functional.cross_entropy(
                second_logits, torch.tensor((1,), dtype=torch.int64)
            )
            second_loss.backward()
        finally:
            handle.remove()

        self.assertEqual(len(captured), 2)
        self.assertIsNot(captured[0], captured[1])
        self.assertFalse(torch.equal(captured[0], captured[1]))
        second_frequency_gradient = model.position.freqs.grad
        self.assertIsNotNone(second_frequency_gradient)
        self.assertTrue(torch.isfinite(second_frequency_gradient).all())
        self.assertGreater(second_frequency_gradient.abs().sum().item(), 0.0)


class CLEVRQuestionAnswererQuestionOnlyTests(unittest.TestCase):
    def test_same_seed_state_matches_g_and_mode_specific_arguments_are_rejected(self):
        seed = 20261025
        torch.manual_seed(seed)
        global_model = CLEVRQuestionAnswerer(vocab_size=8, mode="G").eval()
        torch.manual_seed(seed)
        question_only = CLEVRQuestionAnswerer(
            vocab_size=8, mode="Q-only"
        ).eval()

        global_state = global_model.state_dict()
        question_only_state = question_only.state_dict()
        self.assertEqual(tuple(global_state), tuple(question_only_state))
        for name, global_value in global_state.items():
            with self.subTest(state=name):
                question_only_value = question_only_state[name]
                self.assertEqual(global_value.shape, question_only_value.shape)
                self.assertEqual(global_value.dtype, question_only_value.dtype)
                self.assertTrue(torch.equal(global_value, question_only_value))

        global_parameters = dict(global_model.named_parameters())
        question_only_parameters = dict(question_only.named_parameters())
        self.assertEqual(tuple(global_parameters), tuple(question_only_parameters))
        self.assertEqual(
            sum(parameter.numel() for parameter in global_parameters.values()),
            sum(
                parameter.numel()
                for parameter in question_only_parameters.values()
            ),
        )

        image, question_tokens, question_length = _valid_answerer_inputs()
        for invalid_image in (image, object()):
            with self.subTest(image=type(invalid_image).__name__), self.assertRaises(
                ValueError
            ):
                question_only(invalid_image, question_tokens, question_length)
        with self.assertRaises(ValueError):
            question_only(None, question_tokens, question_length, lam=0.0)

        torch.manual_seed(seed)
        expected_next_random = torch.rand(4)
        torch.manual_seed(seed)
        with self.assertRaises(ValueError):
            CLEVRQuestionAnswerer(
                vocab_size=8, mode="Q-only", m10_seed=20007
            )
        self.assertTrue(torch.equal(torch.rand(4), expected_next_random))

        rgb_modes = (
            ("G", {}),
            ("B", {}),
            ("M10", {"m10_seed": 20007}),
        )
        for mode, kwargs in rgb_modes:
            with self.subTest(mode=mode):
                model = CLEVRQuestionAnswerer(
                    vocab_size=8, mode=mode, **kwargs
                ).eval()
                with self.assertRaises(TypeError):
                    model(None, question_tokens, question_length)

    def test_zero_patch_tokens_use_g_phases_and_continue_through_all_blocks(self):
        torch.manual_seed(20261025)
        model = CLEVRQuestionAnswerer(vocab_size=8, mode="Q-only").eval()
        _, question_tokens, question_length = _valid_answerer_inputs(2)
        captured_tokens = []
        captured_phases = []

        def reject_projection(_module, _args):
            raise AssertionError("Q-only called the RGB patch projection")

        def capture_block(_module, args, kwargs):
            captured_tokens.append(args[0].detach().clone())
            captured_phases.append(kwargs["freqs_cis"])

        projection_handle = model.backbone.patch_embed.register_forward_pre_hook(
            reject_projection
        )
        block_handles = [
            block.register_forward_pre_hook(capture_block, with_kwargs=True)
            for block in model.backbone.blocks
        ]
        try:
            with torch.no_grad():
                expected_cls = model.backbone.cls_token.expand(2, -1, -1) + (
                    model.question_encoder(
                        question_tokens, question_length
                    ).unsqueeze(1)
                )
                logits = model(None, question_tokens, question_length)
        finally:
            projection_handle.remove()
            for handle in block_handles:
                handle.remove()

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(len(captured_tokens), 6)
        self.assertEqual(tuple(captured_tokens[0].shape), (2, 197, 192))
        self.assertTrue(torch.equal(captured_tokens[0][:, :1], expected_cls))
        self.assertTrue(
            torch.equal(
                captured_tokens[0][:, 1:],
                torch.zeros((2, 196, 192), dtype=captured_tokens[0].dtype),
            )
        )
        self.assertGreater(captured_tokens[1][:, 1:].abs().sum().item(), 0.0)

        expected_phase = build_axial_cis(
            14, 14, mode="G", theta=100.0
        )
        self.assertEqual(len(captured_phases), 6)
        for block_index, phase in enumerate(captured_phases):
            with self.subTest(block=block_index):
                self.assertIs(phase, captured_phases[0])
                self.assertTrue(torch.equal(phase, expected_phase))

    def test_answer_loss_skips_projection_and_reaches_nonvisual_components(self):
        torch.manual_seed(20261025)
        model = CLEVRQuestionAnswerer(vocab_size=8, mode="Q-only")
        _, question_tokens, question_length = _valid_answerer_inputs(2)

        logits = model(None, question_tokens, question_length)
        nn.functional.cross_entropy(logits, torch.tensor((0, 1))).backward()

        projection = model.backbone.patch_embed.proj
        self.assertIsNone(projection.weight.grad)
        self.assertIsNone(projection.bias.grad)
        parameters = {
            "embedding": model.question_encoder.embedding.weight,
            "gru input weight": model.question_encoder.gru.weight_ih_l0,
            "gru hidden weight": model.question_encoder.gru.weight_hh_l0,
            "cls": model.backbone.cls_token,
            "final norm": model.backbone.norm.weight,
            "answer head": model.backbone.head.weight,
        }
        for block_index, block in enumerate(model.backbone.blocks):
            parameters[f"block {block_index} qkv"] = block.attn.qkv.weight
            parameters[f"block {block_index} mlp"] = block.mlp.fc1.weight
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_rejects_empty_question_batch_without_an_image(self):
        model = CLEVRQuestionAnswerer(vocab_size=8, mode="Q-only").eval()

        with self.assertRaises(ValueError):
            model(
                None,
                torch.empty((0, 2), dtype=torch.int64),
                torch.empty((0,), dtype=torch.int64),
            )


class CLEVRQuestionAnswererArchitectureTests(unittest.TestCase):
    def test_constructs_the_exact_pub_r1_1_backbone(self):
        model = CLEVRQuestionAnswerer(vocab_size=8)

        self.assertEqual(model.backbone.patch_embed.img_size, (224, 224))
        self.assertEqual(model.backbone.patch_embed.patch_size, (16, 16))
        self.assertEqual(model.backbone.patch_embed.grid_size, (14, 14))
        self.assertEqual(model.backbone.patch_embed.num_patches, 196)
        patch_projection = model.backbone.patch_embed.proj
        self.assertIsInstance(patch_projection, nn.Conv2d)
        self.assertEqual(patch_projection.in_channels, 3)
        self.assertEqual(patch_projection.out_channels, 192)
        self.assertEqual(patch_projection.kernel_size, (16, 16))
        self.assertEqual(patch_projection.stride, (16, 16))
        self.assertIsNotNone(patch_projection.bias)

        self.assertEqual(model.backbone.embed_dim, 192)
        self.assertEqual(model.backbone.num_heads, 3)
        self.assertEqual(len(model.backbone.blocks), 6)
        self.assertFalse(model.backbone.use_ape)
        self.assertIsNone(model.backbone.pos_embed)
        self.assertFalse(model.backbone.rope_mixed)
        self.assertEqual(model.backbone.dropout_rate, 0.0)
        for attribute in (
            "relative_position_bias_table",
            "relative_position_index",
            "relative_position_bias",
            "rel_pos_bias",
            "attn_mask",
            "local_attn_mask",
            "local_mask",
            "window_size",
        ):
            with self.subTest(module="backbone", forbidden_attribute=attribute):
                self.assertFalse(hasattr(model.backbone, attribute))
        for block in model.backbone.blocks:
            with self.subTest(block=id(block)):
                self.assertEqual(block.norm1.normalized_shape, (192,))
                self.assertEqual(block.norm1.eps, 1e-6)
                self.assertEqual(block.norm2.normalized_shape, (192,))
                self.assertEqual(block.norm2.eps, 1e-6)
                self.assertEqual(block.attn.num_heads, 3)
                self.assertEqual(192 // block.attn.num_heads, 64)
                self.assertEqual(block.attn.qkv.in_features, 192)
                self.assertEqual(block.attn.qkv.out_features, 576)
                self.assertIsNotNone(block.attn.qkv.bias)
                self.assertIs(
                    block.attn.forward.__func__, differentiable_attention
                )
                self.assertEqual(
                    tuple(signature(block.attn.forward).parameters),
                    ("x", "freqs_cis"),
                )
                self.assertEqual(
                    tuple(name for name, _ in block.attn.named_children()),
                    ("qkv", "attn_drop", "proj", "proj_drop"),
                )
                self.assertEqual(
                    tuple(block.attn.named_parameters(recurse=False)), ()
                )
                self.assertEqual(tuple(block.attn.named_buffers(recurse=False)), ())
                for attribute in (
                    "q_norm",
                    "k_norm",
                    "relative_position_bias_table",
                    "relative_position_index",
                    "relative_position_bias",
                    "rel_pos_bias",
                    "attn_mask",
                    "local_attn_mask",
                    "local_mask",
                    "window_size",
                ):
                    self.assertFalse(hasattr(block.attn, attribute))
                self.assertEqual(block.attn.attn_drop.p, 0.0)
                self.assertEqual(block.attn.proj_drop.p, 0.0)
                self.assertEqual(block.mlp.fc1.in_features, 192)
                self.assertEqual(block.mlp.fc1.out_features, 768)
                self.assertEqual(block.mlp.fc2.in_features, 768)
                self.assertEqual(block.mlp.fc2.out_features, 192)
                self.assertIsInstance(block.mlp.act, nn.GELU)
                self.assertEqual(block.mlp.drop1.p, 0.0)
                self.assertEqual(block.mlp.drop2.p, 0.0)
                self.assertIsInstance(block.drop_path, nn.Identity)
                self.assertTrue(
                    torch.equal(block.gamma_1, torch.full_like(block.gamma_1, 1e-4))
                )
                self.assertTrue(
                    torch.equal(block.gamma_2, torch.full_like(block.gamma_2, 1e-4))
                )

        self.assertEqual(model.backbone.norm.normalized_shape, (192,))
        self.assertEqual(model.backbone.norm.eps, 1e-6)
        self.assertIsInstance(model.backbone.head, nn.Linear)
        self.assertEqual(model.backbone.head.in_features, 192)
        self.assertEqual(model.backbone.head.out_features, 2)

    def test_first_block_receives_question_conditioned_cls_and_unchanged_patches(self):
        torch.manual_seed(20261024)
        model = CLEVRQuestionAnswerer(vocab_size=8).eval()
        image, question_tokens, question_length = _valid_answerer_inputs(2)
        captured = []

        def capture_input(_module, args, _kwargs):
            captured.append(args[0].detach().clone())

        handle = model.backbone.blocks[0].register_forward_pre_hook(
            capture_input, with_kwargs=True
        )
        try:
            with torch.no_grad():
                expected_patches = model.backbone.patch_embed(image)
                expected_cls = model.backbone.cls_token.expand(2, -1, -1) + (
                    model.question_encoder(question_tokens, question_length).unsqueeze(1)
                )
                model(image, question_tokens, question_length)
        finally:
            handle.remove()

        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0][:, :1], expected_cls))
        self.assertTrue(torch.equal(captured[0][:, 1:], expected_patches))

    def test_forward_returns_two_logits_and_rejects_non_contract_images_or_batches(self):
        model = CLEVRQuestionAnswerer(vocab_size=8).eval()
        image, question_tokens, question_length = _valid_answerer_inputs()
        with torch.no_grad():
            logits = model(image, question_tokens, question_length)
        self.assertEqual(tuple(logits.shape), (1, 2))

        invalid_calls = (
            ("rank", image[0], question_tokens, question_length),
            ("channels", image[:, :1], question_tokens, question_length),
            ("height", image[:, :, :-1], question_tokens, question_length),
            ("width", image[:, :, :, :-1], question_tokens, question_length),
            ("empty batch", image[:0], question_tokens[:0], question_length[:0]),
            (
                "token batch disagreement",
                image,
                question_tokens.expand(2, -1),
                question_length,
            ),
            (
                "length batch disagreement",
                image,
                question_tokens,
                question_length.expand(2),
            ),
        )
        for name, bad_image, bad_tokens, bad_length in invalid_calls:
            with self.subTest(name=name), self.assertRaises(ValueError):
                model(bad_image, bad_tokens, bad_length)


class CLEVRQuestionAnswererPositionTests(unittest.TestCase):
    def test_same_seed_g_and_b_have_equal_non_position_state_and_parameter_count(self):
        seed = 20261024
        torch.manual_seed(seed)
        global_model = CLEVRQuestionAnswerer(vocab_size=8, mode="G")
        torch.manual_seed(seed)
        local_model = CLEVRQuestionAnswerer(vocab_size=8, mode="B")

        global_state = {
            name: value
            for name, value in global_model.state_dict().items()
            if not name.startswith("position.")
        }
        local_state = {
            name: value
            for name, value in local_model.state_dict().items()
            if not name.startswith("position.")
        }
        self.assertEqual(tuple(global_state), tuple(local_state))
        for name, global_value in global_state.items():
            with self.subTest(state=name):
                local_value = local_state[name]
                self.assertEqual(global_value.shape, local_value.shape)
                self.assertEqual(global_value.dtype, local_value.dtype)
                self.assertTrue(torch.equal(global_value, local_value))

        global_parameters = dict(global_model.named_parameters())
        local_parameters = dict(local_model.named_parameters())
        self.assertEqual(tuple(global_parameters), tuple(local_parameters))
        for name, global_parameter in global_parameters.items():
            with self.subTest(parameter=name):
                local_parameter = local_parameters[name]
                self.assertEqual(global_parameter.shape, local_parameter.shape)
                self.assertEqual(global_parameter.dtype, local_parameter.dtype)
        self.assertEqual(
            sum(p.numel() for p in global_model.parameters() if p.requires_grad),
            sum(p.numel() for p in local_model.parameters() if p.requires_grad),
        )
        self.assertEqual(list(global_model.position.parameters()), [])
        self.assertEqual(list(local_model.position.parameters()), [])
        self.assertTrue(
            torch.equal(
                local_model.position.fixed_lambda,
                torch.tensor([1.0], dtype=torch.float32),
            )
        )

    def test_every_b_block_receives_the_exact_fixed_local_phase(self):
        model = CLEVRQuestionAnswerer(vocab_size=8, mode="B").eval()
        image, question_tokens, question_length = _valid_answerer_inputs()
        captured = []

        def capture_phase(_module, _args, kwargs):
            captured.append(kwargs["freqs_cis"])

        handles = [
            block.register_forward_pre_hook(capture_phase, with_kwargs=True)
            for block in model.backbone.blocks
        ]
        try:
            with torch.no_grad():
                model.forward_tokens(image, question_tokens, question_length)
        finally:
            for handle in handles:
                handle.remove()

        expected = build_axial_cis(14, 14, mode="B", lam=1.0, theta=100.0)
        global_phase = build_axial_cis(14, 14, mode="G", theta=100.0)
        self.assertEqual(len(captured), 6)
        for block_index, phase in enumerate(captured):
            with self.subTest(block=block_index):
                self.assertIs(phase, captured[0])
                self.assertTrue(torch.equal(phase, expected))
        self.assertTrue(torch.equal(captured[0][:, :8], global_phase[:, :8]))
        self.assertTrue(torch.equal(captured[0][:, 16:24], global_phase[:, 16:24]))
        self.assertFalse(torch.equal(captured[0][:, 8:16], global_phase[:, 8:16]))
        self.assertFalse(torch.equal(captured[0][:, 24:32], global_phase[:, 24:32]))

    def test_b_lambda_zero_matches_g_end_to_end_with_paired_state(self):
        seed = 20261024
        torch.manual_seed(seed)
        global_model = CLEVRQuestionAnswerer(vocab_size=8, mode="G").eval()
        torch.manual_seed(seed)
        local_model = CLEVRQuestionAnswerer(vocab_size=8, mode="B").eval()
        image, question_tokens, question_length = _valid_answerer_inputs(2)

        with torch.no_grad():
            global_tokens = global_model.forward_tokens(
                image, question_tokens, question_length
            )
            local_tokens = local_model.forward_tokens(
                image, question_tokens, question_length, lam=0.0
            )
            global_logits = global_model(image, question_tokens, question_length)
            local_logits = local_model(
                image, question_tokens, question_length, lam=0.0
            )

        self.assertTrue(torch.equal(local_tokens, global_tokens))
        self.assertTrue(torch.equal(local_logits, global_logits))

    def test_answer_loss_reaches_every_non_position_component(self):
        torch.manual_seed(20261024)
        model = CLEVRQuestionAnswerer(vocab_size=8, mode="B")
        image, question_tokens, question_length = _valid_answerer_inputs(2)
        logits = model(image, question_tokens, question_length)
        torch.nn.functional.cross_entropy(logits, torch.tensor((0, 1))).backward()

        parameters = {
            "embedding": model.question_encoder.embedding.weight,
            "gru input weight": model.question_encoder.gru.weight_ih_l0,
            "gru hidden weight": model.question_encoder.gru.weight_hh_l0,
            "cls": model.backbone.cls_token,
            "patch projection": model.backbone.patch_embed.proj.weight,
            "final norm": model.backbone.norm.weight,
            "answer head": model.backbone.head.weight,
        }
        parameters.update(
            {
                f"block {index} qkv": block.attn.qkv.weight
                for index, block in enumerate(model.backbone.blocks)
            }
        )
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertEqual(list(model.position.parameters()), [])

    def test_rejects_c_and_unknown_modes(self):
        seed = 20261024
        torch.manual_seed(seed)
        expected_next_random = torch.rand(4)
        for mode in ("C", "unknown"):
            with self.subTest(mode=mode):
                torch.manual_seed(seed)
                error = None
                try:
                    CLEVRQuestionAnswerer(vocab_size=8, mode=mode)
                except Exception as caught:
                    error = caught
                next_random = torch.rand(4)
                with self.subTest(contract="rejected before allocation"):
                    self.assertTrue(torch.equal(next_random, expected_next_random))
                with self.subTest(contract="raises ValueError"):
                    self.assertIs(type(error), ValueError)


if __name__ == "__main__":
    unittest.main()
