"""Committed PUB-R1.1 PCG64 training-index stream."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from numpy.random import Generator, PCG64


_SAMPLER_SEED_OFFSET = 10_000


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
        if not indices:
            raise CLEVRSamplingError("commit indices must not be empty")
        if any(type(index) is not int for index in indices):
            raise CLEVRSamplingError(
                "commit indices must contain ordinary Python integers"
            )
        if any(index < 0 or index >= self._dataset_size for index in indices):
            raise CLEVRSamplingError("commit index is outside the dataset range")

        supplied = tuple(indices)
        expected = self.preview(len(supplied))
        if supplied != expected:
            raise CLEVRSamplingError(
                "commit indices disagree with the expected stream prefix"
            )
        self._advance_committed(len(supplied))

    def _clone_generator(self) -> Generator:
        bit_generator = PCG64(0)
        bit_generator.state = deepcopy(self._generator.bit_generator.state)
        return Generator(bit_generator)

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
