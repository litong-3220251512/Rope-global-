"""PUB-R1.1 model components for CLEVR question answering."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.classification import RoPEClassifier
from src.local_rope import Lambda
from models.vit_rope import (
    compute_mixed_cis,
    init_random_2d_freqs,
    init_t_xy,
)


class CLEVRMixedPosition(nn.Module):
    """Official RoPE-Mixed frequencies for the fixed PUB-R1.1 backbone."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            frequencies = torch.stack(
                [
                    init_random_2d_freqs(
                        dim=64, num_heads=3, theta=10.0
                    )
                    for _ in range(6)
                ],
                dim=1,
            ).view(2, 6, 96)
        self.freqs = nn.Parameter(frequencies)
        t_x, t_y = init_t_xy(end_x=14, end_y=14)
        self.register_buffer("freqs_t_x", t_x)
        self.register_buffer("freqs_t_y", t_y)

    def forward(self) -> Tensor:
        """Return live per-layer/head phases with no cross-forward cache."""
        t_x = self.freqs_t_x.to(dtype=self.freqs.dtype)
        t_y = self.freqs_t_y.to(dtype=self.freqs.dtype)
        return compute_mixed_cis(self.freqs, t_x, t_y, num_heads=3)


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


class CLEVRQuestionAnswerer(nn.Module):
    """Frozen PUB-R1.1 answerer with G, B, M10, or Q-only input."""

    def __init__(
        self,
        vocab_size: int,
        *,
        mode: str = "G",
        m10_seed: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"G", "B", "M10", "Q-only"}:
            raise ValueError("mode must be G, B, M10, or Q-only")
        if mode == "M10":
            if m10_seed is None:
                raise ValueError("M10 requires m10_seed")
            if type(m10_seed) is not int:
                raise TypeError("m10_seed must be an integer")
            if m10_seed < 0:
                raise ValueError("m10_seed must be non-negative")
        elif m10_seed is not None:
            raise ValueError("m10_seed is only valid for M10")

        classifier = RoPEClassifier(
            mode="G" if mode in {"M10", "Q-only"} else mode,
            fixed_lambda=1.0,
            image_size=224,
            patch_size=16,
            embed_dim=192,
            depth=6,
            heads=3,
            num_classes=2,
            attention_backend="manual",
            init_scale=1e-4,
            drop_path_rate=0.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            theta=100.0,
        )
        self.backbone = classifier.backbone
        self.position = classifier.position
        self.question_encoder = CLEVRQuestionEncoder(vocab_size)
        if mode == "M10":
            self.position = CLEVRMixedPosition(seed=m10_seed)
        self.mode = mode

    @staticmethod
    def _validate_batch(
        image: Tensor, question_tokens: Tensor, question_length: Tensor
    ) -> None:
        if not isinstance(image, Tensor):
            raise TypeError("image must be a tensor")
        if image.ndim != 4:
            raise ValueError("image must have shape [N, 3, 224, 224]")
        if image.shape[0] == 0:
            raise ValueError("image batch must not be empty")
        if tuple(image.shape[1:]) != (3, 224, 224):
            raise ValueError("image must have shape [N, 3, 224, 224]")
        if image.shape[0] != question_tokens.shape[0]:
            raise ValueError("image and question token batches must match")
        if image.shape[0] != question_length.shape[0]:
            raise ValueError("image and question length batches must match")

    def forward_tokens(
        self,
        image: Tensor | None,
        question_tokens: Tensor,
        question_length: Tensor,
        *,
        lam: Lambda | None = None,
    ) -> Tensor:
        """Return all normalized CLS and patch tokens."""
        if self.mode == "Q-only":
            if image is not None:
                raise ValueError("image must be None for Q-only")
            if lam is not None:
                raise ValueError("lam is not valid for Q-only")
            question = self.question_encoder(question_tokens, question_length)
            if question.shape[0] == 0:
                raise ValueError("question batch must not be empty")
            batch_size = question.shape[0]
        else:
            self._validate_batch(image, question_tokens, question_length)
            if self.mode == "M10" and lam is not None:
                raise ValueError("lam is not valid for M10")
            patches = self.backbone.patch_embed(image)
            question = self.question_encoder(question_tokens, question_length)
            batch_size = image.shape[0]
        cls = self.backbone.cls_token.expand(batch_size, -1, -1)
        cls = cls + question.unsqueeze(1)
        if self.mode == "Q-only":
            patches = cls.new_zeros(
                (
                    batch_size,
                    self.backbone.patch_embed.num_patches,
                    self.backbone.embed_dim,
                )
            )
        tokens = torch.cat((cls, patches), dim=1)
        if self.mode == "M10":
            phases = self.position()
            for block_index, block in enumerate(self.backbone.blocks):
                tokens = block(tokens, freqs_cis=phases[block_index])
        else:
            phases = self.position(14, 14, lam=lam, device=tokens.device)
            for block in self.backbone.blocks:
                tokens = block(tokens, freqs_cis=phases)
        return self.backbone.norm(tokens)

    def forward_features(
        self,
        image: Tensor | None,
        question_tokens: Tensor,
        question_length: Tensor,
        *,
        lam: Lambda | None = None,
    ) -> Tensor:
        return self.forward_tokens(
            image, question_tokens, question_length, lam=lam
        )[:, 0]

    def forward(
        self,
        image: Tensor | None,
        question_tokens: Tensor,
        question_length: Tensor,
        *,
        lam: Lambda | None = None,
    ) -> Tensor:
        features = self.forward_features(
            image, question_tokens, question_length, lam=lam
        )
        return self.backbone.head(features)
