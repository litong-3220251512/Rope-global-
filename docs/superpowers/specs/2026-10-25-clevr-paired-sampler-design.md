# CLEVR Paired Committed Index Stream Design

Date: 2026-10-25
Protocol: PUB-R1.1
Status: implementation basis; the user has instructed the agent to continue the accepted plan without another design-question round

## 1. Intent

PUB-R1.1 requires G, B, M10, and Q-only runs with the same training seed to
consume exactly the same sample sequence. The sequence is an infinite
concatenation of independent, without-replacement epoch permutations produced
by NumPy PCG64 with seed `10000 + run_seed`. A batch that reaches an epoch tail
continues into the next permutation; no selected sample is dropped.

Checkpointing adds a separate constraint: saved sampler position must describe
samples the training loop actually consumed, not indices a multi-worker
`DataLoader` prefetched. This unit therefore provides an explicit committed
index stream. Callers may preview future indices without changing checkpointed
state and advance that state only by committing the exact consumed prefix.

This is the first independent subunit of protocol implementation unit 4. It
does not build a DataLoader, model, optimizer, scheduler, training loop, or
checkpoint file, and it does not run an experiment.

## 2. Goals, non-goals, and anti-goals

### Goals

- Generate the frozen PCG64 sample stream from only `dataset_size` and
  `run_seed`, with internal sampler seed exactly `10000 + run_seed`.
- Traverse each epoch permutation without replacement and concatenate the next
  permutation whenever a requested range crosses an epoch boundary.
- Make preview, including look-ahead offsets for worker prefetch, observationally
  pure.
- Advance the committed cursor only when the caller presents the exact expected
  consumed prefix.
- Serialize enough direct state to resume the next sample exactly: current
  permutation, committed cursor, current epoch index, and PCG64 state.
- Strictly validate restored state before any mutation, including direct replay
  from the recorded seed instead of hashes.
- Leave Python's global `random`, NumPy's legacy global RNG, PyTorch RNG, and
  CUDA RNG untouched.

### Non-goals

- No model initialization, state copying, optimizer groups, learning-rate
  scheduler, gradient accumulation, clipping, forward/backward, or checkpoint
  file I/O.
- No Dataset or DataLoader construction and no worker process management. A
  later runner owns how previewed indices are dispatched and when consumption
  is acknowledged.
- No real CLEVR access, engineering smoke, GPU use, pilot, training, evaluation,
  or performance claim.
- No restriction to formal seeds `101`--`105` or pilot seed `9001` in this low
  level utility. The later run-config validator owns the run matrix; this class
  accepts any non-negative integer seed so bounded tests and future protocol
  revisions need no hidden bypass.

### Anti-goals

- Do not subclass a mutating `torch.utils.data.Sampler` whose iterator cursor
  can advance when DataLoader workers prefetch rather than when training
  consumes data.
- Do not materialize all `8000 * 128` formal-run indices in memory.
- Do not restart an epoch after resume, drop an incomplete tail, pad with a
  repeated final item, or use replacement within an epoch.
- Do not derive state equivalence with SHA, MD5, or a custom digest.
- Do not touch a process-global RNG or couple sampling to model/M10 seeds.

## 3. Approaches considered

### A. Committed stream with pure preview and exact-prefix commit — selected

Keep one current permutation, its committed cursor, the zero-based epoch index,
and an owned NumPy `Generator(PCG64(...))`. `preview(count, offset=...)` clones
that state in memory and reads from the clone. `commit(indices)` first compares
the supplied indices with `preview(len(indices))`; only an exact match advances
the owned state.

This directly separates dispatched/prefetched indices from consumed indices.
Unconsumed previews disappear after a crash and are regenerated from the saved
committed state. Memory is `O(dataset_size)` rather than `O(total_run_samples)`.

### B. Stateful PyTorch Sampler or BatchSampler — rejected

The installed PyTorch `Sampler` base has no `state_dict`/`load_state_dict`
contract. More importantly, DataLoader can request indices ahead of the
training loop; with workers, installed PyTorch defaults to two prefetched
batches per worker. Saving the iterator cursor would therefore violate the
protocol's consumed-position rule unless another committed-position layer were
added. Approach A is that missing layer and can later feed any loader strategy.

### C. Precompute the complete run sequence — rejected

Precomputing `1,024,000` indices per formal run would make restore simple, but
duplicates the generator output in every checkpoint and ties the utility to one
budget. It also obscures the required current-permutation/cursor state and is
unnecessary for exact replay.

### D. Recreate future permutations from seed and epoch only — rejected

Replay from seed is useful for validation, but omitting the PCG64 state from the
checkpoint would not satisfy the explicit sampler-RNG state requirement.
Storing both permits direct inspection and independent consistency validation.

## 4. Public API

Add `src/clevr_sampling.py` with:

```python
class CLEVRSamplingError(ValueError): ...


class CLEVRIndexStream:
    def __init__(self, *, dataset_size: int, run_seed: int) -> None: ...

    @property
    def dataset_size(self) -> int: ...

    @property
    def run_seed(self) -> int: ...

    @property
    def sampler_seed(self) -> int: ...

    def preview(self, count: int, *, offset: int = 0) -> tuple[int, ...]: ...

    def commit(self, indices: Sequence[int]) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
```

`dataset_size` must be a positive Python integer and `run_seed` a non-negative
Python integer; booleans are rejected. `count` must be positive and `offset`
non-negative. `commit` requires a non-empty finite sequence of Python integers;
it does not accept an arbitrary iterable because replaying a generator during
validation would obscure what was committed.

`preview(count, offset=o)` means: starting at the current committed position,
skip the next `o` stream elements and return the following `count` elements.
It never changes `state_dict()`. This supports bounded look-ahead: a runner may
dispatch offsets `0`, `16`, ..., but it must commit returned prefixes in order
after actual consumption.

`commit(indices)` compares the complete tuple to `preview(len(indices))` with
ordinary element equality. A mismatch is a hard error and leaves state exactly
unchanged. An exact match advances by that many elements. The later trainer is
responsible for committing only samples whose forward/backward contribution it
accepts; checkpoint files are still written only at complete update boundaries.

## 5. Mathematical contract

Let the dataset's already-frozen base order be the index set

\[
I=\{0,1,\ldots,N-1\},\qquad N>0.
\]

For run seed \(s\ge 0\), instantiate one independent generator

\[
R_s=\operatorname{Generator}(\operatorname{PCG64}(10000+s)).
\]

Successive calls to `R_s.permutation(N)` produce epoch permutations
\(\pi_0,\pi_1,\ldots\), each a bijection from positions
\(0,\ldots,N-1\) to \(I\). Define the infinite sample stream by

\[
S_s=\pi_0\Vert\pi_1\Vert\pi_2\Vert\cdots.
\]

If \(c\ge0\) is the number of committed samples, then

\[
\operatorname{preview}(q,o)=
(S_s[c+o],\ldots,S_s[c+o+q-1]),
\]

for integer \(q>0\), \(o\ge0\). Preview leaves \(c\) and all owned RNG state
unchanged. For supplied tuple \(x\), commit is

\[
c' = \begin{cases}
c+|x|,&x=\operatorname{preview}(|x|,0),\\
c,&\text{otherwise, with an exception}.
\end{cases}
\]

The stored epoch/cursor representation satisfies

\[
c=eN+r,qquad e\in\mathbb Z_{\ge0},\quad 0\le r\le N,
\]

with one canonical boundary convention: after committing exactly the last item
of \(\pi_e\), state may retain `epoch_index=e`, `cursor=N`; generation of
\(\pi_{e+1}\) occurs only when a later preview or commit needs another item.
This makes the checkpoint say exactly which current permutation has been fully
consumed rather than eagerly moving into an unconsumed epoch.

For any arms \(a,b\in\{G,B,M10,Q\text{-only}\}\) instantiated with equal
\((N,s)\), their streams and every committed state reached by equal commits are
identical. The method does not infer or compare model mode; the later runner
must pass the same run seed and committed prefixes to each paired arm.

## 6. Checkpoint state and validation

`state_dict()` returns fresh containers with exactly:

```text
protocol_version: "PUB-R1.1"
dataset_size: positive int
run_seed: non-negative int
sampler_seed: 10000 + run_seed
epoch_index: non-negative int
permutation: tuple[int, ...] of length dataset_size
cursor: int in [0, dataset_size]
bit_generator_state: deep-copied PCG64 state mapping
```

No returned mutable object aliases the stream. Before `load_state_dict` changes
anything, it validates:

1. input is a mapping with exactly the listed keys;
2. protocol and constructor-bound `dataset_size`, `run_seed`, and derived
   sampler seed match;
3. integer fields are Python integers, not booleans, and within range;
4. `permutation` is a tuple of ordinary integers and sorts exactly to
   `range(dataset_size)`;
5. a temporary `PCG64` accepts the deep-copied bit-generator state;
6. independent replay from `sampler_seed` through `epoch_index` produces a
   permutation and PCG64 state directly equal to the serialized values.

Only after every check passes are owned fields replaced. Thus malformed or
inconsistent state cannot partially move the committed position. Replay is at
most about 108 permutations for the frozen 9,506-question, 8,000-update formal
run, so validation cost is bounded and incurred only on restore.

This sampler state is only one part of the later checkpoint. The later runner
must additionally save model, optimizer, scheduler, update number, and the
required Python/NumPy/PyTorch/CUDA RNG states. This class's owned PCG64 generator
is intentionally separate from NumPy's legacy global state.

## 7. Hand-worked and independent numerical example

For `dataset_size=5`, `run_seed=101`, sampler seed is `10101`. A direct local
NumPy 2.2.6 check produced:

```text
pi_0 = [0, 2, 1, 4, 3]
pi_1 = [4, 3, 1, 2, 0]
pi_2 = [3, 0, 4, 2, 1]
```

Therefore, before any commit:

```text
preview(7)           = (0, 2, 1, 4, 3, 4, 3)
preview(4, offset=1) = (2, 1, 4, 3)
state                = epoch 0, cursor 0
```

After `commit((0,2,1))`, state is epoch 0, cursor 3 and `preview(4)` is
`(4,3,4,3)`. Committing that tuple consumes the two-item tail of `pi_0` and the
first two items of `pi_1`, leaving epoch 1, cursor 2. No tail is discarded and
the repeated values across the epoch boundary are legitimate because
without-replacement applies within each epoch, not across epochs.

If a caller instead tries to commit `(4,3,3,4)`, the tuple differs from the
expected prefix; the call fails and state stays at epoch 0, cursor 3.

## 8. Invariants and falsification

The implementation must maintain:

1. **Per-epoch bijection:** every generated permutation contains each dataset
   index exactly once.
2. **Frozen seed mapping:** the only sampling seed is `10000 + run_seed`.
3. **Pairing:** equal constructor inputs and equal commits yield exact equal
   outputs/state across arms.
4. **Preview purity:** arbitrary previews and offsets leave all state directly
   equal, including PCG64 state.
5. **Commit atomicity:** wrong indices or invalid inputs raise without changing
   state.
6. **Boundary continuity:** ranges can cross one or many epoch boundaries with
   no dropped or inserted element.
7. **Restore continuity:** a restored stream's future previews and states equal
   an uninterrupted stream after the same commits.
8. **RNG isolation:** constructing, previewing, committing, saving, and loading
   do not change global Python, NumPy, PyTorch, or CUDA RNG state.

The design is falsified by any sequence mismatch for paired arms, any state
change caused only by preview/prefetch, accepting an inconsistent checkpoint,
a failed commit that changes later output, or a restored next index differing
from uninterrupted execution.

## 9. Mature-library and license review

Local inspection on 2026-10-25 found:

| Need | Candidate | Evidence | Decision |
| --- | --- | --- | --- |
| Frozen permutations | `numpy.random.Generator(PCG64)` | NumPy `2.2.6`; installed metadata carries the NumPy BSD license text; protocol already selects this API | Reuse directly |
| DataLoader sampling | `torch.utils.data.Sampler` | PyTorch `2.6.0+cu124`, BSD-3-Clause; base class exposes no state hooks | Do not subclass for committed-state ownership |
| Worker look-ahead | PyTorch `DataLoader` | Installed source documents/defaults `prefetch_factor=2` for `num_workers>0` and primes `prefetch_factor * num_workers` tasks | Keep preview separate from commit |
| Stateful loader | `torchdata.stateful_dataloader` | `torchdata` is not installed in the locked environment | Do not add a dependency for this small protocol-specific layer |
| State copies | Python `copy.deepcopy` and direct mapping/list/tuple comparisons | Standard library | Reuse; no hash/digest |

No external source is copied. The custom code is limited to the protocol's
seed formula, concatenated epoch semantics, committed cursor, and strict state
validation that general libraries do not encode together.

## 10. Tests and implementation boundary

Add `tests/test_clevr_sampling.py` before `src/clevr_sampling.py`. Tests use
small index domains as mathematical checks, not a synthetic training dataset:

- the fixed `N=5, s=101` literals above, including offsets and multi-epoch
  crossing;
- per-epoch bijection and identical paired streams;
- preview purity and commit mismatch atomicity by direct state equality;
- exact checkpoint round trip and uninterrupted/restored future equality at an
  epoch boundary and inside an epoch;
- strict rejection of constructor, preview, commit, schema, permutation,
  cursor, seed, epoch, and bit-generator inconsistencies, with no partial
  mutation;
- independence from global Python/NumPy/PyTorch CPU RNG states, checked by
  direct state/tensor equality rather than digests.

The PR may modify only this design, its implementation plan,
`src/clevr_sampling.py`, and `tests/test_clevr_sampling.py`. It does not import
the unmerged CLEVR dataset/model branches and does not add a command that starts
training or touches user data.
