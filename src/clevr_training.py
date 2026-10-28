"""PUB-R1.1 paired model initialization and training configuration."""

from __future__ import annotations

from collections import OrderedDict
import math

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from timm.optim import param_groups_weight_decay

from src.clevr_model import CLEVRQuestionAnswerer


_M10_SEED_OFFSET = 20_000
_MAX_TORCH_SEED = (1 << 64) - 1
_MODES = {"G", "B", "M10", "Q-only"}
_BASE_LEARNING_RATE = 3e-4
_MINIMUM_LEARNING_RATE = 1e-6
_WARMUP_UPDATES = 400
_TOTAL_UPDATES = 8_000
_WEIGHT_DECAY = 0.05


class CLEVRTrainingConfigurationError(ValueError):
    """Raised when a PUB-R1.1 training configuration is invalid."""


class CLEVRPairedInitialization:
    """Own one common non-position state and copy it into one arm at a time."""

    def __init__(self, *, vocab_size: int, run_seed: int) -> None:
        if type(vocab_size) is not int:
            raise CLEVRTrainingConfigurationError(
                "vocab_size must be an ordinary integer"
            )
        if vocab_size < 4:
            raise CLEVRTrainingConfigurationError(
                "vocab_size must include the four fixed special tokens"
            )
        if type(run_seed) is not int:
            raise CLEVRTrainingConfigurationError(
                "run_seed must be an ordinary integer"
            )
        if run_seed < 0 or run_seed > _MAX_TORCH_SEED - _M10_SEED_OFFSET:
            raise CLEVRTrainingConfigurationError(
                "run_seed is outside the range supported by the M10 seed mapping"
            )

        self._vocab_size = vocab_size
        self._run_seed = run_seed
        self._m10_seed = _M10_SEED_OFFSET + run_seed
        source = self._construct("G")
        self._common_state = self._extract_common_state(source)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def run_seed(self) -> int:
        return self._run_seed

    @property
    def m10_seed(self) -> int:
        return self._m10_seed

    def common_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in self._common_state.items()
        )

    def build(self, mode: str) -> CLEVRQuestionAnswerer:
        if type(mode) is not str or mode not in _MODES:
            raise CLEVRTrainingConfigurationError(
                "mode must be G, B, M10, or Q-only"
            )

        model = self._construct(mode)
        complete_state = model.state_dict()
        target_common = self._extract_common_state(model)
        if tuple(target_common) != tuple(self._common_state):
            raise CLEVRTrainingConfigurationError(
                "answerer common state keys disagree with the paired initialization"
            )
        for name, source in self._common_state.items():
            target = target_common[name]
            if target.shape != source.shape:
                raise CLEVRTrainingConfigurationError(
                    f"answerer common state shape disagrees for {name}"
                )
            if target.dtype != source.dtype:
                raise CLEVRTrainingConfigurationError(
                    f"answerer common state dtype disagrees for {name}"
                )
            complete_state[name] = source.detach().clone()

        try:
            model.load_state_dict(complete_state, strict=True)
        except RuntimeError as exc:
            raise CLEVRTrainingConfigurationError(
                "answerer rejected the paired initialization state"
            ) from exc

        loaded_common = self._extract_common_state(model)
        for name, source in self._common_state.items():
            if not torch.equal(loaded_common[name], source):
                raise CLEVRTrainingConfigurationError(
                    f"answerer common state copy disagrees for {name}"
                )
        return model

    def _construct(self, mode: str) -> CLEVRQuestionAnswerer:
        arguments: dict[str, object] = {
            "vocab_size": self._vocab_size,
            "mode": mode,
        }
        if mode == "M10":
            arguments["m10_seed"] = self._m10_seed
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(self._run_seed)
            return CLEVRQuestionAnswerer(**arguments)

    @staticmethod
    def _extract_common_state(
        model: CLEVRQuestionAnswerer,
    ) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in model.state_dict().items()
            if not name.startswith("position.")
        )


def build_clevr_adamw(model: CLEVRQuestionAnswerer) -> AdamW:
    """Build the fixed AdamW groups with directly auditable parameter names."""
    if not isinstance(model, CLEVRQuestionAnswerer):
        raise CLEVRTrainingConfigurationError(
            "model must be a CLEVRQuestionAnswerer"
        )

    named_parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    parameter_by_id = {
        id(parameter): (name, parameter)
        for name, parameter in named_parameters.items()
    }
    skip = {"backbone.cls_token"} | {
        name for name in named_parameters if name.startswith("position.")
    }
    raw_groups = param_groups_weight_decay(
        model,
        weight_decay=_WEIGHT_DECAY,
        no_weight_decay_list=skip,
    )

    decay_by_id: dict[int, float] = {}
    for group in raw_groups:
        weight_decay = float(group["weight_decay"])
        if weight_decay not in {0.0, _WEIGHT_DECAY}:
            raise CLEVRTrainingConfigurationError(
                "timm returned an unexpected weight-decay group"
            )
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in parameter_by_id:
                raise CLEVRTrainingConfigurationError(
                    "timm returned an unknown or non-trainable parameter"
                )
            if parameter_id in decay_by_id:
                raise CLEVRTrainingConfigurationError(
                    "timm returned a parameter more than once"
                )
            decay_by_id[parameter_id] = weight_decay

    if set(decay_by_id) != set(parameter_by_id):
        raise CLEVRTrainingConfigurationError(
            "timm parameter groups do not cover every trainable parameter"
        )

    optimizer_groups: list[dict[str, object]] = []
    for group_name, weight_decay in (
        ("decay", _WEIGHT_DECAY),
        ("no_decay", 0.0),
    ):
        names = [
            name
            for name, parameter in named_parameters.items()
            if decay_by_id[id(parameter)] == weight_decay
        ]
        parameters = [named_parameters[name] for name in names]
        if not parameters:
            raise CLEVRTrainingConfigurationError(
                f"AdamW {group_name} parameter group must not be empty"
            )
        optimizer_groups.append(
            {
                "params": parameters,
                "param_names": names,
                "group_name": group_name,
                "weight_decay": weight_decay,
            }
        )

    return AdamW(
        optimizer_groups,
        lr=_BASE_LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def clevr_learning_rate(update: int) -> float:
    """Return the frozen learning rate for one-indexed optimizer update."""
    if type(update) is not int:
        raise CLEVRTrainingConfigurationError(
            "update must be an ordinary integer"
        )
    if update < 1 or update > _TOTAL_UPDATES:
        raise CLEVRTrainingConfigurationError(
            "update must be between 1 and 8000"
        )
    if update <= _WARMUP_UPDATES:
        return _BASE_LEARNING_RATE * update / _WARMUP_UPDATES
    return _MINIMUM_LEARNING_RATE + (
        (_BASE_LEARNING_RATE - _MINIMUM_LEARNING_RATE)
        / 2
        * (
            1
            + math.cos(
                math.pi
                * (update - _WARMUP_UPDATES)
                / (_TOTAL_UPDATES - _WARMUP_UPDATES)
            )
        )
    )


def build_clevr_lr_scheduler(optimizer: AdamW) -> LambdaLR:
    """Set each group to the LR for the next protocol optimizer update."""
    if not isinstance(optimizer, AdamW):
        raise CLEVRTrainingConfigurationError("optimizer must be AdamW")
    if not optimizer.param_groups:
        raise CLEVRTrainingConfigurationError(
            "optimizer must contain at least one parameter group"
        )
    for group in optimizer.param_groups:
        learning_rate = group.get("lr")
        if type(learning_rate) not in {int, float}:
            raise CLEVRTrainingConfigurationError(
                "optimizer group learning rate must be an ordinary number"
            )
        if learning_rate != _BASE_LEARNING_RATE:
            raise CLEVRTrainingConfigurationError(
                "optimizer group learning rate must start at 3e-4"
            )
        if "initial_lr" in group:
            initial_learning_rate = group["initial_lr"]
            if type(initial_learning_rate) not in {int, float}:
                raise CLEVRTrainingConfigurationError(
                    "optimizer group initial_lr must be an ordinary number"
                )
            if initial_learning_rate != _BASE_LEARNING_RATE:
                raise CLEVRTrainingConfigurationError(
                    "optimizer group initial_lr must be 3e-4"
                )

    def multiplier(scheduler_epoch: int) -> float:
        next_update = min(scheduler_epoch + 1, _TOTAL_UPDATES)
        return clevr_learning_rate(next_update) / _BASE_LEARNING_RATE

    return LambdaLR(optimizer, lr_lambda=multiplier)
