"""Committed PUB-R1.1 PCG64 training-index stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from numpy.random import Generator, PCG64


_SAMPLER_SEED_OFFSET = 10_000
_PROTOCOL_VERSION = "PUB-R1.1"
_PCG64_STATE_FIELDS = {"bit_generator", "state", "has_uint32", "uinteger"}
_PCG64_CORE_STATE_FIELDS = {"state", "inc"}
_UINT32_LIMIT = 1 << 32
_UINT128_LIMIT = 1 << 128
_STATE_FIELDS = {
    "protocol_version",
    "dataset_size",
    "run_seed",
    "sampler_seed",
    "epoch_index",
    "permutation",
    "cursor",
    "bit_generator_state",
}


class CLEVRSamplingError(ValueError):
    """Raised when the paired index-stream contract is violated."""


class CLEVRIndexStream:
    """Preview sample indices without advancing the committed position."""

    def __init__(self, *, dataset_size: int, run_seed: int) -> None:
        if type(dataset_size) is not int:
            raise CLEVRSamplingError("dataset_size must be an ordinary integer")
        if dataset_size <= 0:
            raise CLEVRSamplingError("dataset_size must be positive")
        if type(run_seed) is not int:
            raise CLEVRSamplingError("run_seed must be an ordinary integer")
        if run_seed < 0:
            raise CLEVRSamplingError("run_seed must be non-negative")

        self._dataset_size = dataset_size
        self._run_seed = run_seed
        self._sampler_seed = _SAMPLER_SEED_OFFSET + run_seed
        self._generator = Generator(PCG64(self._sampler_seed))
        self._permutation = self._next_permutation(self._generator)
        self._epoch_index = 0
        self._cursor = 0

    @property
    def dataset_size(self) -> int:
        return self._dataset_size

    @property
    def run_seed(self) -> int:
        return self._run_seed

    @property
    def sampler_seed(self) -> int:
        return self._sampler_seed

    def preview(self, count: int, *, offset: int = 0) -> tuple[int, ...]:
        if type(count) is not int:
            raise CLEVRSamplingError("preview count must be an ordinary integer")
        if count <= 0:
            raise CLEVRSamplingError("preview count must be positive")
        if type(offset) is not int:
            raise CLEVRSamplingError("preview offset must be an ordinary integer")
        if offset < 0:
            raise CLEVRSamplingError("preview offset must be non-negative")

        generator = self._clone_generator()
        permutation = self._permutation
        cursor = self._cursor
        cursor, permutation = self._advance_local(
            generator,
            permutation,
            cursor,
            offset,
        )

        output: list[int] = []
        remaining = count
        while remaining:
            if cursor == self._dataset_size:
                permutation = self._next_permutation(generator)
                cursor = 0
            take = min(remaining, self._dataset_size - cursor)
            output.extend(permutation[cursor : cursor + take])
            cursor += take
            remaining -= take
        return tuple(output)

    def commit(self, indices: Sequence[int]) -> None:
        if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
            raise CLEVRSamplingError("commit indices must be a concrete sequence")
        supplied = tuple(indices)
        if not supplied:
            raise CLEVRSamplingError("commit indices must not be empty")
        if any(type(index) is not int for index in supplied):
            raise CLEVRSamplingError(
                "commit indices must contain ordinary Python integers"
            )
        if any(index < 0 or index >= self._dataset_size for index in supplied):
            raise CLEVRSamplingError("commit index is outside the dataset range")

        expected = self.preview(len(supplied))
        if supplied != expected:
            raise CLEVRSamplingError(
                "commit indices disagree with the expected stream prefix"
            )
        self._advance_committed(len(supplied))

    def state_dict(self) -> dict[str, object]:
        return {
            "protocol_version": _PROTOCOL_VERSION,
            "dataset_size": self._dataset_size,
            "run_seed": self._run_seed,
            "sampler_seed": self._sampler_seed,
            "epoch_index": self._epoch_index,
            "permutation": tuple(self._permutation),
            "cursor": self._cursor,
            "bit_generator_state": deepcopy(self._generator.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise CLEVRSamplingError("sampler state must be a mapping")
        if set(state) != _STATE_FIELDS:
            missing = sorted(_STATE_FIELDS.difference(state))
            extra = sorted(set(state).difference(_STATE_FIELDS))
            raise CLEVRSamplingError(
                f"sampler state fields disagree; missing={missing}, extra={extra}"
            )
        if state["protocol_version"] != _PROTOCOL_VERSION:
            raise CLEVRSamplingError(
                f"sampler protocol_version must be {_PROTOCOL_VERSION!r}"
            )

        dataset_size = self._state_integer(state, "dataset_size")
        run_seed = self._state_integer(state, "run_seed")
        sampler_seed = self._state_integer(state, "sampler_seed")
        epoch_index = self._state_integer(state, "epoch_index")
        cursor = self._state_integer(state, "cursor")
        if dataset_size != self._dataset_size:
            raise CLEVRSamplingError(
                "sampler state dataset_size disagrees with the constructor"
            )
        if run_seed != self._run_seed:
            raise CLEVRSamplingError(
                "sampler state run_seed disagrees with the constructor"
            )
        if sampler_seed != self._sampler_seed:
            raise CLEVRSamplingError(
                "sampler state sampler_seed disagrees with the frozen seed mapping"
            )
        if epoch_index < 0:
            raise CLEVRSamplingError("sampler state epoch_index must be non-negative")
        if cursor < 0 or cursor > self._dataset_size:
            raise CLEVRSamplingError("sampler state cursor is outside the valid range")

        permutation = state["permutation"]
        if not isinstance(permutation, tuple):
            raise CLEVRSamplingError("sampler state permutation must be a tuple")
        if len(permutation) != self._dataset_size:
            raise CLEVRSamplingError(
                "sampler state permutation length disagrees with dataset_size"
            )
        if any(type(index) is not int for index in permutation):
            raise CLEVRSamplingError(
                "sampler state permutation must contain ordinary Python integers"
            )
        if sorted(permutation) != list(range(self._dataset_size)):
            raise CLEVRSamplingError(
                "sampler state permutation must be a dataset-index bijection"
            )

        raw_bit_generator_state = state["bit_generator_state"]
        if not isinstance(raw_bit_generator_state, Mapping):
            raise CLEVRSamplingError(
                "sampler PCG64 bit_generator_state must be a mapping"
            )
        bit_generator_state = self._validated_pcg64_state(raw_bit_generator_state)
        try:
            structural_probe = PCG64(0)
            structural_probe.state = deepcopy(bit_generator_state)
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise CLEVRSamplingError(
                "sampler state has an invalid PCG64 bit_generator_state"
            ) from exc

        replay_generator = Generator(PCG64(self._sampler_seed))
        replay_permutation: tuple[int, ...] | None = None
        for _ in range(epoch_index + 1):
            replay_permutation = self._next_permutation(replay_generator)
        if replay_permutation != permutation:
            raise CLEVRSamplingError(
                "sampler state permutation is inconsistent with seed/epoch replay"
            )
        if replay_generator.bit_generator.state != bit_generator_state:
            raise CLEVRSamplingError(
                "sampler PCG64 state is inconsistent with seed/epoch replay"
            )

        self._generator = replay_generator
        self._permutation = replay_permutation
        self._epoch_index = epoch_index
        self._cursor = cursor

    def _clone_generator(self) -> Generator:
        bit_generator = PCG64(0)
        bit_generator.state = deepcopy(self._generator.bit_generator.state)
        return Generator(bit_generator)

    @staticmethod
    def _state_integer(state: Mapping[str, object], field: str) -> int:
        value = state[field]
        if type(value) is not int:
            raise CLEVRSamplingError(
                f"sampler state {field} must be an ordinary integer"
            )
        return value

    @staticmethod
    def _validated_pcg64_state(state: Mapping[str, object]) -> dict[str, object]:
        if set(state) != _PCG64_STATE_FIELDS:
            raise CLEVRSamplingError(
                "sampler PCG64 bit_generator_state fields disagree"
            )
        if state["bit_generator"] != "PCG64":
            raise CLEVRSamplingError(
                "sampler PCG64 bit_generator_state names the wrong generator"
            )

        core_state = state["state"]
        if not isinstance(core_state, Mapping):
            raise CLEVRSamplingError("sampler PCG64 core state must be a mapping")
        if set(core_state) != _PCG64_CORE_STATE_FIELDS:
            raise CLEVRSamplingError("sampler PCG64 core state fields disagree")

        scalar_limits = (
            ("state", core_state["state"], _UINT128_LIMIT),
            ("inc", core_state["inc"], _UINT128_LIMIT),
            ("has_uint32", state["has_uint32"], 2),
            ("uinteger", state["uinteger"], _UINT32_LIMIT),
        )
        for field, value, limit in scalar_limits:
            if type(value) is not int:
                raise CLEVRSamplingError(
                    f"sampler PCG64 {field} must be an ordinary integer"
                )
            if value < 0 or value >= limit:
                raise CLEVRSamplingError(
                    f"sampler PCG64 {field} is outside the valid range"
                )

        return deepcopy(dict(state))

    def _next_permutation(self, generator: Generator) -> tuple[int, ...]:
        return tuple(int(index) for index in generator.permutation(self._dataset_size))

    def _advance_local(
        self,
        generator: Generator,
        permutation: tuple[int, ...],
        cursor: int,
        count: int,
    ) -> tuple[int, tuple[int, ...]]:
        remaining = count
        while remaining:
            if cursor == self._dataset_size:
                permutation = self._next_permutation(generator)
                cursor = 0
            step = min(remaining, self._dataset_size - cursor)
            cursor += step
            remaining -= step
        return cursor, permutation

    def _advance_committed(self, count: int) -> None:
        remaining = count
        while remaining:
            if self._cursor == self._dataset_size:
                self._permutation = self._next_permutation(self._generator)
                self._epoch_index += 1
                self._cursor = 0
            step = min(remaining, self._dataset_size - self._cursor)
            self._cursor += step
            remaining -= step
