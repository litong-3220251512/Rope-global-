# CLEVR Paired Committed Index Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable PUB-R1.1 PCG64 index stream whose checkpoint position advances only after the training loop explicitly commits the exact consumed sample prefix.

**Architecture:** Add one NumPy-backed `CLEVRIndexStream` independent of Dataset, DataLoader, models, and training. It owns one epoch permutation plus committed cursor and RNG, provides pure offset previews for prefetch, validates exact-prefix commits, and restores only checkpoint states that directly replay from the frozen seed.

**Tech Stack:** Python 3.11 standard library, NumPy 2.2.6 `Generator(PCG64)`, `unittest`; PyTorch 2.6.0 is used only in RNG-isolation regression tests.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-paired-sampler-design.md`

## Global Constraints

- Work only on `feat/clevr-paired-sampler` in `/tmp/rope-clevr-paired-sampler`, based on master SHA `7dc1291d9da2298c9eb3614936603d2a83990d12`; never modify `master`.
- Beyond the committed design and this plan, create only `src/clevr_sampling.py` and `tests/test_clevr_sampling.py`.
- Use exactly `numpy.random.Generator(numpy.random.PCG64(10000 + run_seed))`; never use or seed a process-global RNG.
- `dataset_size` is a positive ordinary Python `int`; `run_seed` is a non-negative ordinary Python `int`; booleans are invalid.
- Preview is pure; only exact, non-empty, in-order commit advances state. A failed commit or failed restore is atomic.
- State contains exact protocol, constructor identity, epoch, permutation, cursor, and deep-copied PCG64 state; validation uses direct values and replay, never hashes.
- Do not add a dependency, Dataset/DataLoader wrapper, worker, model, optimizer, scheduler, checkpoint file, runner, CLI, download, data access, GPU operation, smoke, training, or evaluation command.
- Follow TDD and run only CPU tests with the explicit flightgpt interpreter.
- Keep `tests.test_cifar_training` and `tests.test_cifar_data` out of aggregate commands because they execute legacy hash logic prohibited by repository policy.

## Review Focus

- Python booleans and NumPy scalar integers must not silently pass constructor/state schema checks; Task 1 and Task 2 add strict-type cases.
- A preview crossing several epochs or using a nonzero offset must not mutate current permutation, cursor, epoch, or RNG; Task 1 adds direct repeated-output and later state checks.
- A wrong commit after a correct preview must leave the next output unchanged; Task 1 adds an atomic mismatch case.
- A checkpoint whose permutation and PCG64 state are each structurally valid but belong to different epochs/seeds must fail before mutation; Task 2 adds replay-consistency cases.
- Mutating nested containers returned by `state_dict()` must not mutate the stream; Task 2 adds alias-isolation checks for permutation and bit-generator state.

---

### Task 1: Pure preview and committed progression

**Files:**
- Create: `tests/test_clevr_sampling.py`
- Create: `src/clevr_sampling.py`

**Interfaces:**
- Consumes: `numpy.random.Generator`, `numpy.random.PCG64`, and `collections.abc.Sequence`.
- Produces: `CLEVRSamplingError(ValueError)` and `CLEVRIndexStream` constructor/properties plus `preview(count, offset=0)` and `commit(indices)`.
- Defers to Task 2: `state_dict()` and `load_state_dict()`.

- [ ] **Step 1: Write failing fixed-sequence and boundary tests**

Create `tests/test_clevr_sampling.py` and import the wished-for public API. Add:

```python
class CLEVRIndexStreamTests(unittest.TestCase):
    def test_matches_fixed_pcg64_sequence_and_crosses_epoch_tail(self): ...
    def test_equal_run_inputs_produce_identical_paired_streams(self): ...
    def test_preview_offset_and_multi_epoch_lookahead_are_pure(self): ...
    def test_wrong_commit_is_atomic_and_exact_commit_advances(self): ...
    def test_rejects_invalid_constructor_preview_and_commit_inputs(self): ...
```

Use only hand-recorded literals from the design for `dataset_size=5`,
`run_seed=101`: permutations `[0,2,1,4,3]`, `[4,3,1,2,0]`, and
`[3,0,4,2,1]`. Assert initial `preview(7)` is
`(0,2,1,4,3,4,3)`, `preview(4, offset=1)` is `(2,1,4,3)`, committing
`(0,2,1)` makes the next four `(4,3,4,3)`, and committing those leaves the
next three `(1,2,0)`.

Prove preview purity without Task 2 by repeating earlier and later previews
around deep look-ahead and comparing literal/previous tuples. Prove commit
atomicity by recording the next tuple, rejecting a reordered tuple, and
observing the same next tuple afterward. Strict invalid cases cover zero,
negative, float, bool constructor fields; non-positive/bool count; negative or
bool offset; empty/string/generator commits; bool, float, negative, and
out-of-range elements.

- [ ] **Step 2: Run Task 1 tests and observe RED**

```bash
PYTHONPATH=/tmp/rope-clevr-paired-sampler CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests/test_clevr_sampling.py -v
```

Expected: import failure because `src.clevr_sampling` does not exist.

- [ ] **Step 3: Implement the minimum committed stream**

In `src/clevr_sampling.py`, define:

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
```

Generate `pi_0` in the constructor. Keep `_epoch_index=0`, `_cursor=0`, and an
owned generator. Implement preview with a cloned generator state, copied
permutation, and local cursor/epoch; skip `offset`, then return exactly `count`
Python integers. Keep the canonical `cursor == dataset_size` boundary until a
later read needs another index. `commit` validates a concrete non-empty
sequence of ordinary Python integers, compares its tuple to `preview(len(...))`,
then advances the owned state with the same epoch-transition semantics.

- [ ] **Step 4: Run Task 1 tests and observe GREEN**

Run the Step 2 command. Expected: all Task 1 tests pass with no warnings.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/clevr_sampling.py tests/test_clevr_sampling.py
git commit -m "feat: add committed CLEVR index stream"
```

---

### Task 2: Strict checkpoint state and exact restore

**Files:**
- Modify: `tests/test_clevr_sampling.py`
- Modify: `src/clevr_sampling.py`

**Interfaces:**
- Consumes: Task 1 `CLEVRIndexStream` committed state.
- Produces: fresh-container `state_dict() -> dict[str, object]` and atomic `load_state_dict(state: Mapping[str, object]) -> None`.
- State keys: `protocol_version`, `dataset_size`, `run_seed`, `sampler_seed`, `epoch_index`, `permutation`, `cursor`, `bit_generator_state`.

- [ ] **Step 1: Write failing checkpoint and isolation tests**

Extend the test class with:

```python
def test_state_dict_is_exact_and_does_not_alias_stream(self): ...
def test_restore_matches_uninterrupted_stream_inside_and_at_epoch_boundary(self): ...
def test_rejects_invalid_or_inconsistent_state_without_partial_mutation(self): ...
def test_operations_leave_global_python_numpy_and_torch_rng_unchanged(self): ...
```

Assert exact state keys and values after committing three items and after
committing an exact five-item epoch. Mutate the returned permutation container
and nested PCG64 fields, then prove a fresh state and future preview are
unchanged. Restore into a new equal-constructor stream and compare at least 17
future indices and subsequent direct states with uninterrupted execution.

Table-drive invalid state: non-mapping, missing/extra key, protocol mismatch,
constructor/sampler seed mismatch, bool/negative epoch or cursor, cursor above
size, non-tuple/wrong-length/duplicate/out-of-range/bool permutation, malformed
PCG64 mapping, and structurally valid permutation/RNG state copied from a
different epoch. Before each load, save a trusted state and next preview; after
failure, compare both again to prove atomicity.

For RNG isolation, seed and snapshot Python `random`, NumPy legacy global state,
and PyTorch CPU state; perform construct/preview/commit/state/load; compare every
snapshot directly. Do not initialize or query CUDA in this CPU test.

- [ ] **Step 2: Run the new tests and observe RED**

Run the Task 1 test-file command. Expected: errors because `state_dict` and
`load_state_dict` do not exist.

- [ ] **Step 3: Implement deep-copied state and replay validation**

Add:

```python
def state_dict(self) -> dict[str, object]: ...
def load_state_dict(self, state: Mapping[str, object]) -> None: ...
```

Use `copy.deepcopy` for PCG64 state and a tuple of ordinary Python integers for
the permutation. Validate the exact field set and all scalar/container types
before mutation. Reconstruct a temporary `Generator(PCG64(sampler_seed))`, call
`permutation(dataset_size)` through `epoch_index`, and directly compare the
resulting current permutation and bit-generator state with the serialized
values. Assign only canonical replayed values after all checks pass.

- [ ] **Step 4: Run focused and allowed CPU regressions**

Run the test file; expected all sampler tests pass. Then run the established
aggregate CPU suite with the new module, excluding only the two policy-forbidden
CIFAR hash modules. Expected: all allowed tests pass, one existing CUDA-only
test skips, and only existing dependency warnings remain.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/clevr_sampling.py tests/test_clevr_sampling.py
git commit -m "feat: restore CLEVR sample stream exactly"
```

---

### Task 3: Scope audit, independent review, and real PR

**Files:**
- Verify: `docs/superpowers/specs/2026-10-25-clevr-paired-sampler-design.md`
- Verify: `docs/superpowers/plans/2026-10-25-clevr-paired-sampler.md`
- Verify: `src/clevr_sampling.py`
- Verify: `tests/test_clevr_sampling.py`

**Interfaces:**
- Consumes: Tasks 1-2 committed implementation.
- Produces: reproducible CPU evidence and a real non-draft PR against `master`; no new feature.

- [ ] **Step 1: Run the complete committed-head allowed CPU suite**

Use `CUDA_VISIBLE_DEVICES=''` and the explicit flightgpt interpreter. Exclude
only `tests.test_cifar_training` and `tests.test_cifar_data` for the recorded
hash-policy reason. Expected: all allowed tests pass; the existing CUDA-only
test skips; only understood existing warnings remain.

- [ ] **Step 2: Audit branch scope and direct evidence**

```bash
git diff --check master...HEAD
git diff --stat master...HEAD
git diff --name-status master...HEAD
git status --short --branch
```

Inspect the diff and confirm it contains exactly the four planned files, no
hash, dependency, Dataset/DataLoader/model/trainer/checkpoint-file/CLI code, and
no command that reads user data or launches GPU/training work.

- [ ] **Step 3: Obtain one fresh whole-branch review**

Give the reviewer exact base/head SHAs, design, plan, diff package, test
evidence, Review Focus, and ledger rulings. Require severity-ranked findings,
independent fixed-sequence/state checks, and an explicit Declined-to-judge list.
Fix every re-graded Critical/Important item once with RED-to-GREEN tests; record
Minor items without expanding scope.

- [ ] **Step 4: Push and create the real PR**

Push `feat/clevr-paired-sampler` and create a non-draft PR against `master`.
Include problem, equations/state contract, mature-library and license review,
changes, RED/GREEN/full-suite evidence, independent review, and explicit limits.
Verify PR state/base/head and remote/local SHA directly. Do not merge or tag and
do not claim CI unless an actual check exists.
