# CLEVR resource measurement contract

## Definition before implementation

PUB-R1.1 section 5.3 requires measured GPU time, wall time including data I/O,
peak memory and throughput. The baseline runner records none of these.
This is instrumentation only: models, sample order, losses, gradients,
optimizer/scheduler calls, update budgets and scientific metrics stay fixed.
No additional warm-up updates or benchmark runs are introduced.

For one completed update/evaluation j with N_j sample presentations:

- W_j = perf_counter(end after CUDA synchronization) - perf_counter(start).
  This includes reading samples, preprocessing, collation, transfers, compute,
  metric extraction and measurement overhead, but not checkpoint/log publication.
- C_j = sum_r elapsed_time(start_event_r,end_event_r)/1000 seconds. Each region
  starts after data transfer and ends after forward/loss/backward, or after
  gradient clipping/optimizer/scheduler/gradient clearing. Evaluation regions
  contain forward/loss/metric work; CPU prediction serialization is outside.
- Throughput = N_j/W_j; zero-duration clocks produce unavailable throughput,
  not division by a fabricated epsilon. Training N_j=128; evaluation uses its
  actual partition size (including a short final batch).
- P_j is max_memory_allocated and R_j is max_memory_reserved, in bytes, after
  resetting allocator peaks at operation entry. Existing live allocations count;
  reserved cache is reported separately. These are this process's PyTorch
  allocator peaks, not device-wide NVML usage or setup/checkpoint memory peaks.

CUDA events measure stream elapsed computation spans, including launch/host
gaps within the marked regions; they are not a sum of kernel-active time.
Host data preparation between micro-batches is excluded from C_j. Synchronize
before the operation and at its end, not after every region. Do not infer CPU
or I/O time as W_j-C_j. CPU fixtures make no CUDA calls and report GPU fields
as null/unavailable, never as measured zero.

An attempt summary sums completed training/evaluation operations separately:
W=sum W_j, C=sum C_j, throughput=sum N_j/sum W_j, GPU compute hours=C/3600,
peak=max_j P_j and max_j R_j. Example: two updates of 128 samples taking 2 and
6 seconds have throughput 256/8=32, not mean(64,21.333). GPU spans of 250 and
750 ms sum to 1 second. Resumed attempts start new counters; replayed prediction
files do not count as new GPU work. Failed/incomplete operations are not included
in completed-operation totals and failure summaries explicitly state this.

Also record arm wall time from per-arm setup through final/failure logging
preparation, including checkpoints/evaluations, plus shared preflight/setup wall
time charged once to the first arm of an invocation. Close/final-event write is
outside that snapshot. Base-test evaluation remains separately authorized and
has its own attempt summary. These scopes prevent reporting update-only time as
the total runtime. No historical costs are silently reconstructed after resume.

## Chosen implementation and library review (2026-10-28)

Reuse installed PyTorch 2.6.0+cu124 (BSD-3-Clause): CUDA Event, current_stream,
synchronize, reset_peak_memory_stats, max_memory_allocated/max_memory_reserved.
Read installed torch/cuda/streams.py Event record/elapsed_time and memory.py.
Also checked current official [Event documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)
and [allocator peaks](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html).
Use Python 3.11 stdlib time.perf_counter (PSF license), per its
[official definition](https://docs.python.org/3.11/library/time.html#time.perf_counter).
Both are maintained upstream libraries; installed versions are not asserted to
be latest. No new dependency. A profiler trace is unnecessary overhead for
aggregate telemetry; NVML polling does not isolate these computation spans and
would introduce a separate monitor. Use a thin event/clock wrapper and existing
attempt journals instead.

## Falsification and validation

Fail the implementation if input-loading gaps enter GPU spans, sample count or
duration units are wrong, peak counters are not reset for a new operation,
CPU code touches CUDA, resumes double-count prior work, or instrumentation
changes tensors, gradients, sample order or update count. Test hand-computable
fake CUDA events/clocks, CPU model update/evaluation seams, attempt/failure
summaries and reused-boundary behavior. No real GPU execution or user data.

Actual CUDA timings, overhead and GPU allocator behavior still need the user's
separately authorized engineering smoke and pilot. No performance or memory
numbers from synthetic tests are presented as real hardware results.

## Recorded output

Records live in the existing
`<output>/<phase>/seed_<seed>/<method>/attempts/attempt_<number>.jsonl` journal.
`update` and freshly computed `evaluation` events contain `details.resources`.
`complete` and `failure` events contain the attempt summary at the same key,
with separate `training` and `evaluation` totals. A reused evaluation has null
operation resources and increments `reused_evaluations`. Sum attempt costs to
account for retries; do not replace earlier failed-attempt records with the
last successful attempt. All successful operation presentations count, including
repeated updates after checkpoint recovery. Failed-operation partial compute
is unavailable; `completed_operations_only` and `attempt_succeeded` preserve
that distinction. GPU fields are null for CPU verification.

No telemetry is restored from the scientific checkpoint or used to choose
samples, hyperparameters, checkpoints or termination. Instrumentation performs
no autonomous monitoring or additional experiment launches.
