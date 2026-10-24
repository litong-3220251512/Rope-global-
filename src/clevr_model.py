"""PUB-R1.1 model components for CLEVR question answering."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CLEVRQuestionEncoder(nn.Module):
    """Encode padded CLEVR questions at each example's final valid token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if type(vocab_size) is not int:
            raise TypeError("vocab_size must be an integer")
        if vocab_size < 4:
            raise ValueError("vocab_size must include the four fixed special tokens")

        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.gru = nn.GRU(
            64, 192, num_layers=1, batch_first=True, bidirectional=False
        )

    def forward(
        self, question_tokens: Tensor, question_length: Tensor
    ) -> Tensor:
        """Return ``[N, 192]`` states; lengths must satisfy ``2 <= length <= L``."""
        if not isinstance(question_tokens, Tensor):
            raise TypeError("question_tokens must be a tensor")
        if question_tokens.dtype != torch.int64:
            raise TypeError("question_tokens must have dtype int64")
        if question_tokens.ndim != 2:
            raise ValueError("question_tokens must have shape [N, L]")
        if not isinstance(question_length, Tensor):
            raise TypeError("question_length must be a tensor")
        if question_length.dtype != torch.int64:
            raise TypeError("question_length must have dtype int64")
        if question_length.ndim != 1:
            raise ValueError("question_length must have shape [N]")
        if question_tokens.shape[0] != question_length.shape[0]:
            raise ValueError("question token and length batches must match")
        if question_tokens.shape[1] < 2:
            raise ValueError("question token sequence width must be at least two")

        embedded = self.embedding(question_tokens)
        gru_output, _ = self.gru(embedded)
        batch_indices = torch.arange(
            question_tokens.shape[0], device=question_tokens.device
        )
        return gru_output[batch_indices, question_length - 1]
