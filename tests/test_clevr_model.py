"""Data-free tests for the frozen PUB-R1.1 CLEVR model contract."""

import unittest

import torch
from torch import nn

from src.clevr_model import CLEVRQuestionEncoder


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


if __name__ == "__main__":
    unittest.main()
