"""Global/local axial phases compatible with naver-ai/rope-vit.

Patch positions are integer indices, not DINOv3's normalized patch centers.
Only phases are constructed here: the caller retains upstream Q/K rotation,
prefix-token handling, attention, and pretrained frequency order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


Lambda = float | Sequence[float] | Tensor


def _level_count(mode: str, head_dim: int, theta: float) -> int:
    if mode not in {"G", "B", "C"}:
        raise ValueError("mode must be G, B, or C")
    divisor = {"G": 4, "B": 8, "C": 16}[mode]
    if not isinstance(head_dim, int) or head_dim <= 0 or head_dim % divisor:
        raise ValueError(f"head_dim must be positive and divisible by {divisor} for mode {mode}")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    return {"G": 0, "B": 1, "C": 2}[mode]


def build_axial_cis(
    H: int,
    W: int,
    head_dim: int = 64,
    mode: str = "G",
    lam: Lambda = 0.0,
    theta: float = 100.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return complex phases [H*W, head_dim/2], x planes followed by y.

    For head_dim=64, B resets axis frequencies 8..15 at level 1. C resets
    8,10,12,14 at level 1 and 9,11,13,15 at level 2. All higher frequencies
    remain global. ``lam`` is a shared scalar or one value per local level;
    tensor inputs retain their autograd connection. Use 0.0 for exact global
    phases, including when a module has learned a nonzero lambda.

    Regions use r=floor(2**level * x/W), origin=r*W/2**level, independently
    on each axis. Nondivisible grids retain fractional origins; no padding
    or coordinate rescaling occurs. Caller-supplied lambda is normally in
    [0,1]; the functional API also permits extrapolation for diagnostics.
    Float64 is available for numerical derivative checks; training uses FP32.
    """
    levels = _level_count(mode, head_dim, theta)
    if not isinstance(H, int) or not isinstance(W, int) or H <= 0 or W <= 0:
        raise ValueError("H and W must be positive integer patch counts")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("phase dtype must be torch.float32 or torch.float64")
    if device is None and isinstance(lam, Tensor):
        device = lam.device

    # The arithmetic/order matches upstream compute_axial_cis at lambda=0.
    frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 4, device=device, dtype=dtype) / head_dim))
    index = torch.arange(H * W, device=device)
    x_index = index.remainder(W)
    y_index = torch.div(index, W, rounding_mode="floor")
    x = x_index.to(dtype)[:, None]
    y = y_index.to(dtype)[:, None]

    if levels:
        strength = torch.as_tensor(lam, device=device, dtype=dtype).reshape(-1)
        if strength.numel() == 1:
            strength = strength.expand(levels)
        elif strength.numel() != levels:
            raise ValueError(f"mode {mode} needs one scalar or {levels} lambda values")
        planes = torch.arange(head_dim // 4, device=device)
        local_start = head_dim // 8
        for level in range(1, levels + 1):
            mask = planes >= local_start
            if mode == "C":
                mask = mask & ((planes - local_start).remainder(2) == level - 1)
            regions = 2**level
            # Integer division makes region membership exact on odd grids.
            ox = torch.div(x_index * regions, W, rounding_mode="floor").to(dtype) * (W / regions)
            oy = torch.div(y_index * regions, H, rounding_mode="floor").to(dtype) * (H / regions)
            weight = strength[level - 1] * mask.to(dtype)[None, :]
            x = x - ox[:, None] * weight
            y = y - oy[:, None] * weight

    angles_x = x * frequency[None, :]
    angles_y = y * frequency[None, :]
    cis_x = torch.polar(torch.ones_like(angles_x), angles_x)
    cis_y = torch.polar(torch.ones_like(angles_y), angles_y)
    return torch.cat((cis_x, cis_y), dim=-1)


class LocalAxialRoPE(nn.Module):
    """Shared learned or fixed lambda values per local level.

    Call ``module(H, W, device=x.device)`` once per forward and pass its
    phases to the upstream blocks. Do not cache phases across optimizer
    updates or detach them in learned mode: doing so prevents rho from
    learning. ``fixed_lambda=1.0`` stores exact values as a checkpointed
    buffer, creates no position parameters, and does not use a sigmoid.
    The default learned mode retains the original ``rho`` checkpoint key.
    """

    def __init__(
        self,
        head_dim: int = 64,
        mode: str = "G",
        lambda_init: float = 0.05,
        theta: float = 100.0,
        fixed_lambda: float | None = None,
    ) -> None:
        super().__init__()
        levels = _level_count(mode, head_dim, theta)
        if fixed_lambda is not None and (not math.isfinite(fixed_lambda) or not 0.0 <= fixed_lambda <= 1.0):
            raise ValueError("fixed_lambda must be finite and lie between 0 and 1 inclusive")
        if fixed_lambda is None and not 0.0 < lambda_init < 1.0:
            raise ValueError("lambda_init must lie strictly between 0 and 1; use forward(lam=0.0) for exact zero")
        self.head_dim = head_dim
        self.mode = mode
        self.theta = theta
        self.register_buffer(
            "fixed_lambda",
            torch.full((levels,), fixed_lambda, dtype=torch.float32)
            if levels and fixed_lambda is not None else None,
        )
        if levels and fixed_lambda is None:
            initial_rho = math.log(lambda_init / (1.0 - lambda_init))
            self.rho = nn.Parameter(torch.full((levels,), initial_rho, dtype=torch.float32))
        else:
            self.register_buffer("rho", torch.empty(0), persistent=False)

    @property
    def lambdas(self) -> Tensor:
        return self.fixed_lambda if self.fixed_lambda is not None else self.rho.sigmoid()

    def forward(
        self,
        H: int,
        W: int,
        lam: Lambda | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return build_axial_cis(
            H, W, self.head_dim, self.mode,
            self.lambdas if lam is None else lam,
            self.theta,
            device=self.rho.device if device is None else device,
            dtype=dtype,
        )
