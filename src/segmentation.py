"""Dense-prediction adapter retaining the official RoPE-ViT attention."""

from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from src.local_rope import LocalAxialRoPE, build_axial_cis

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "rope-vit"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from models.vit_rope import apply_rotary_emb, rope_axial_deit_small_patch16_LS


def differentiable_attention(self, x, freqs_cis):
    """Upstream attention without its in-place assignment into Q/K views.

    Fixed phases did not require a phase gradient in the upstream model.
    Learnable phases do, so their saved Q/K inputs must remain unmodified.
    The same correction is used for G/B/C; all arithmetic stays unchanged.
    """
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, channels // self.num_heads
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q_patch, k_patch = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis)
    q = torch.cat((q[:, :, :1], q_patch), dim=2)
    k = torch.cat((k[:, :, :1], k_patch), dim=2)
    attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    attention = self.attn_drop(attention)
    output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(output))


class RoPESegmenter(nn.Module):
    def __init__(self, checkpoint=None, image_size=256, mode="G", lambda_init=0.05, fixed_lambda=None):
        super().__init__()
        self.backbone = rope_axial_deit_small_patch16_LS(
            pretrained=False, img_size=image_size
        )
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = payload["model"] if "model" in payload else payload
            # Reject silently ignored/missing weights, including nested checkpoints.
            self.backbone.load_state_dict(state, strict=True)
        for block in self.backbone.blocks:
            block.attn.forward = MethodType(differentiable_attention, block.attn)
        self.backbone.patch_embed.strict_img_size = False
        self.position = LocalAxialRoPE(
            head_dim=64, mode=mode, lambda_init=lambda_init, fixed_lambda=fixed_lambda
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 21, 1),
        )
        self.local_start_block = 0
        self.frozen_prefix_blocks = 0

    def configure_trainable(self, blocks=12, head_only=False):
        if blocks not in range(1, 13):
            raise ValueError("blocks must be between 1 and 12; use head_only=True for head warmup")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.position.parameters():
            parameter.requires_grad_(not head_only)
        self.local_start_block = 0 if head_only else 12 - blocks
        self.frozen_prefix_blocks = 12 if head_only else 12 - blocks
        if not head_only and blocks:
            for block in self.backbone.blocks[-blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)
            if blocks == 12:
                self.backbone.cls_token.requires_grad_(True)
                for parameter in self.backbone.patch_embed.parameters():
                    parameter.requires_grad_(True)

    def forward_tokens(self, image, lam=None):
        height, width = image.shape[-2] // 16, image.shape[-1] // 16
        frozen = self.frozen_prefix_blocks
        with torch.set_grad_enabled(torch.is_grad_enabled() and frozen == 0):
            tokens = self.backbone.patch_embed(image)
            cls = self.backbone.cls_token.expand(image.shape[0], -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
        global_phases = build_axial_cis(height, width, device=image.device)
        local_phases = self.position(height, width, lam=lam, device=image.device)
        for index, block in enumerate(self.backbone.blocks):
            phases = global_phases if index < self.local_start_block else local_phases
            with torch.set_grad_enabled(torch.is_grad_enabled() and index >= frozen):
                tokens = block(tokens, freqs_cis=phases)
        return self.backbone.norm(tokens)

    def forward(self, image, lam=None):
        height, width = image.shape[-2] // 16, image.shape[-1] // 16
        tokens = self.forward_tokens(image, lam=lam)[:, 1:]
        features = tokens.transpose(1, 2).reshape(image.shape[0], 384, height, width)
        return F.interpolate(
            self.decoder(features), size=image.shape[-2:], mode="bilinear", align_corners=False
        )

    def optimizer_groups(self, backbone_lr=1e-5, head_lr=1e-3, rho_lr=1e-2):
        groups = []
        for prefix, module, lr in [
            ("backbone", self.backbone, backbone_lr),
            ("decoder", self.decoder, head_lr),
            ("position", self.position, rho_lr),
        ]:
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    no_decay = parameter.ndim < 2 or prefix == "position"
                    groups.append({
                        "params": [parameter], "lr": lr,
                        "weight_decay": 0.0 if no_decay else 0.05,
                    })
        return groups
