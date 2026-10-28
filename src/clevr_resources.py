"""Operation-scoped clocks and CUDA allocator/event telemetry; no monitoring."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from time import perf_counter

import torch


def _rate(samples: int, seconds: float) -> float | None:
    return samples / seconds if seconds > 0 else None


@dataclass(frozen=True, slots=True)
class CLEVRResourceUsage:
    samples: int
    e2e_seconds: float
    cuda_compute_seconds: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def __post_init__(self):
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("resource sample count must be a positive integer")
        for seconds in (self.e2e_seconds, self.cuda_compute_seconds):
            if seconds is not None and (type(seconds) not in (int, float)
                                       or not math.isfinite(seconds) or seconds < 0):
                raise ValueError("resource duration must be finite and nonnegative")
        if self.e2e_seconds is None:
            raise ValueError("wall duration is required")
        for size in (self.peak_allocated_bytes, self.peak_reserved_bytes):
            if size is not None and (type(size) is not int or size < 0):
                raise ValueError("allocator peak must be nonnegative integer bytes")
        fields = (self.cuda_compute_seconds, self.peak_allocated_bytes, self.peak_reserved_bytes)
        if any(value is None for value in fields) and not all(value is None for value in fields):
            raise ValueError("CUDA resource fields must all be available or all unavailable")

    def as_dict(self) -> dict[str, object]:
        return {
            "resource_schema_version": 1, "samples": self.samples,
            "e2e_seconds": self.e2e_seconds,
            "samples_per_second": _rate(self.samples, self.e2e_seconds),
            "cuda_compute_seconds": self.cuda_compute_seconds,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "cuda_timing": "unavailable" if self.cuda_compute_seconds is None else "event_spans",
        }


class CLEVRResourceMeter:
    """One update/evaluation; no global defaults, RNG changes or GPU polling."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.events = []
        if self.cuda:
            if device.index is None:
                raise ValueError("resource measurement requires an explicit CUDA device index")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.started = perf_counter()

    @contextmanager
    def compute(self):
        if not self.cuda:
            yield
            return
        stream = torch.cuda.current_stream(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        yield
        end.record(stream)
        self.events.append((start, end))

    def finish(self, samples: int) -> CLEVRResourceUsage:
        if self.cuda:
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - self.started
        compute = (sum(start.elapsed_time(end) for start, end in self.events) / 1000
                   if self.cuda else None)
        return CLEVRResourceUsage(
            samples, elapsed, compute,
            torch.cuda.max_memory_allocated(self.device) if self.cuda else None,
            torch.cuda.max_memory_reserved(self.device) if self.cuda else None,
        )


class CLEVRResourceSummary:
    """Costs from this process attempt only; prior/reused results are not charged."""

    def __init__(self, *, shared_setup_seconds: float = 0.0):
        self.started = perf_counter()
        self.shared_setup_seconds = shared_setup_seconds
        self.records = {"training": [], "evaluation": []}
        self.unmeasured_operations = 0
        self.reused_evaluations = 0

    def add(self, operation: str, usage: CLEVRResourceUsage | None) -> None:
        if operation not in self.records:
            raise ValueError("unknown resource operation")
        if usage is None:
            self.unmeasured_operations += 1
        else:
            self.records[operation].append(usage)

    def snapshot(self, *, succeeded: bool) -> dict[str, object]:
        result = {
            "resource_schema_version": 1, "attempt_succeeded": succeeded,
            "completed_operations_only": True,
            "unmeasured_operations": self.unmeasured_operations,
            "reused_evaluations": self.reused_evaluations,
            "arm_wall_seconds": perf_counter() - self.started,
            "shared_setup_seconds": self.shared_setup_seconds,
        }
        all_records = []
        for name, records in self.records.items():
            count = sum(item.samples for item in records)
            wall = sum(item.e2e_seconds for item in records)
            gpu = (sum(item.cuda_compute_seconds for item in records)
                   if records and all(item.cuda_compute_seconds is not None for item in records) else None)
            result[name] = {"operations": len(records), "samples": count,
                            "e2e_seconds": wall, "cuda_compute_seconds": gpu,
                            "samples_per_second": _rate(count, wall)}
            all_records.extend(records)
        gpu_available = (bool(all_records) and not self.unmeasured_operations
                         and all(item.cuda_compute_seconds is not None for item in all_records))
        result["cuda_compute_hours"] = (
            sum(item.cuda_compute_seconds for item in all_records) / 3600 if gpu_available else None)
        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
            result[field] = max(getattr(item, field) for item in all_records) if gpu_available else None
        return result
