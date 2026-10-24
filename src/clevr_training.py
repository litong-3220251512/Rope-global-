"""PUB-R1.1 paired model initialization and training configuration."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor

from src.clevr_model import CLEVRQuestionAnswerer


_M10_SEED_OFFSET = 20_000
_MAX_TORCH_SEED = (1 << 64) - 1
_MODES = {"G", "B", "M10", "Q-only"}


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
