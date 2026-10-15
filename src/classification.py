"""From-scratch RoPE-ViT classification with shared G/B/C initialization.

The model is constructed directly from the vendored implementation; no
checkpoint lookup or pretrained-weight download is performed. Tiny uses
embed_dim=192/heads=3 and Small uses embed_dim=384/heads=6. Both have head
dimension 64 and use an 8x8 patch grid for image_size=32/patch_size=4.
"""

from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.local_rope import Lambda, LocalAxialRoPE
from src.segmentation import apply_rotary_emb, differentiable_attention
from models.vit_rope import (
    RoPEAttention,
    RoPE_Layer_scale_init_Block,
    rope_vit_models,
)


def sdpa_attention(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
    """The same patch-only Q/K rotation with PyTorch's fused attention API."""
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, channels // self.num_heads
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q_patch, k_patch = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis)
    q = torch.cat((q[:, :, :1], q_patch), dim=2)
    k = torch.cat((k[:, :, :1], k_patch), dim=2)
    output = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.attn_drop.p if self.training else 0.0,
        scale=self.scale,
    )
    output = output.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(output))


class RoPEClassifier(nn.Module):
    """Trainable official RoPE-ViT blocks with a CLS classification head.

    ``fixed_lambda=1.0`` is the default: B/C reset the prescribed local
    coordinates exactly and introduce no trainable positional parameters.
    Use ``shared_state_dict`` once on a random G model, then load it into
    each branch with ``load_shared_state_dict``. The strict backbone load
    includes the patch embedding, CLS, all blocks, norm and classifier.

    ``manual`` and ``sdpa`` compute the same attention formula. SDPA may
    choose different kernels, so floating-point results need not be bitwise
    equal. Both leave CLS queries and keys unrotated and preserve gradients.
    """

    def __init__(
        self,
        *,
        mode: str = "G",
        fixed_lambda: float | None = 1.0,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        num_classes: int = 100,
        attention_backend: str = "manual",
        init_scale: float = 1e-4,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        theta: float = 100.0,
        lambda_init: float = 0.05,
    ) -> None:
        super().__init__()
        for name, value in {
            "image_size": image_size, "patch_size": patch_size,
            "embed_dim": embed_dim, "depth": depth, "heads": heads,
            "num_classes": num_classes,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        if attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be manual or sdpa")

        # Validate the head dimension and mode before allocating the backbone.
        self.position = LocalAxialRoPE(
            head_dim=embed_dim // heads, mode=mode, theta=theta,
            lambda_init=lambda_init, fixed_lambda=fixed_lambda,
        )
        self.backbone = rope_vit_models(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=heads, num_classes=num_classes,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_layers=RoPE_Layer_scale_init_Block,
            Attention_block=RoPEAttention,
            rope_theta=theta, rope_mixed=False, use_ape=False,
            init_scale=init_scale, drop_path_rate=drop_path_rate,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        )
        attention = differentiable_attention if attention_backend == "manual" else sdpa_attention
        for block in self.backbone.blocks:
            block.attn.forward = MethodType(attention, block.attn)
        self.image_size = image_size
        self.patch_size = patch_size
        self.attention_backend = attention_backend

    def shared_state_dict(self) -> OrderedDict[str, Tensor]:
        """Return an independent snapshot of all non-position model state."""
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in self.backbone.state_dict().items()
        )

    def load_shared_state_dict(self, state: Mapping[str, Tensor]):
        """Strictly load shared initialization without touching local phases."""
        return self.backbone.load_state_dict(state, strict=True)

    def forward_tokens(self, image: Tensor, lam: Lambda | None = None) -> Tensor:
        height = image.shape[-2] // self.patch_size
        width = image.shape[-1] // self.patch_size
        tokens = self.backbone.patch_embed(image)
        cls = self.backbone.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        phases = self.position(height, width, lam=lam, device=image.device)
        for block in self.backbone.blocks:
            tokens = block(tokens, freqs_cis=phases)
        return self.backbone.norm(tokens)

    def forward_features(self, image: Tensor, lam: Lambda | None = None) -> Tensor:
        return self.forward_tokens(image, lam=lam)[:, 0]

    def forward(self, image: Tensor, lam: Lambda | None = None) -> Tensor:
        features = self.forward_features(image, lam=lam)
        if self.backbone.dropout_rate:
            features = F.dropout(
                features, p=float(self.backbone.dropout_rate), training=self.training
            )
        return self.backbone.head(features)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {f"backbone.{name}" for name in self.backbone.no_weight_decay()} | {
            f"position.{name}" for name, _ in self.position.named_parameters()
        }
