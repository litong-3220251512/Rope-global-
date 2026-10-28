# CLEVR Paired Training Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the PUB-R1.1 one-source paired initialization factory, auditable AdamW parameter groups, and exact update-indexed learning-rate scheduler without adding a training runner.

**Architecture:** One CPU-only `CLEVRPairedInitialization` owns cloned non-position state from one G construction and builds one arm at a time with strict direct copy checks. The same focused module wraps timm's mature decay splitter with a parameter-name manifest, PyTorch AdamW, a pure finite-budget LR formula, and PyTorch LambdaLR.

**Tech Stack:** Python 3.11 standard library, PyTorch 2.6.0 `Module.state_dict`, `AdamW`, and `LambdaLR`; timm 1.0.24 `param_groups_weight_decay`; `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-training-configuration-design.md`

## Global Constraints

- Work only on `feat/clevr-training-config` in `/tmp/rope-clevr-training-config`, based on `feat/clevr-qonly` SHA `98596e5b7fe7f51cdaaf13909e552579fe2fac40`; never modify the base branch.
- Beyond the committed design and this plan, create only `src/clevr_training.py` and `tests/test_clevr_training.py`.
- The common PyTorch seed is `run_seed`; the M10 seed is exactly `20000 + run_seed`; valid `run_seed` is an ordinary Python integer in `[0, 2**64 - 1 - 20000]` with no modulo mapping.
- Common state is every ordered answerer state entry not prefixed `position.`; copy and validate by names, shapes, dtypes, and `torch.equal`, never hashes or approximate summaries.
- Model construction and snapshots must not alter global Python, NumPy, PyTorch CPU, or accelerator RNG streams.
- Reuse timm `param_groups_weight_decay`, PyTorch AdamW, and PyTorch LambdaLR; add no dependency and copy no external source.
- AdamW is fixed to weight decay 0.05/0 groups, `lr=3e-4`, `betas=(0.9,0.999)`, and `eps=1e-8`; every trainable parameter appears exactly once with its exact name recorded.
- The LR function accepts only updates 1 through 8000. LambdaLR represents the next update's LR and is stepped only after a successful optimizer update; after update 8000 it retains `eta_8000`.
- Do not add Dataset/DataLoader/sampler integration, loss/accumulation/clipping loops, checkpoint files, runner/CLI, downloads, real-data access, GPU work, smoke, training, or evaluation.
- Follow TDD and use only CPU tests with the explicit flightgpt interpreter. Exclude `tests.test_cifar_training` and `tests.test_cifar_data` from aggregate commands because repository policy forbids their legacy hash logic.

## Review Focus

- Boundary seeds `2**64 - 1 - 20000` and one above it must be accepted/rejected before RNG mutation, including bool/NumPy scalar cases; Task 1 adds direct boundary and RNG-state tests.
- A returned common snapshot or a mutated built model must not alias the factory or a later arm; Task 1 compares storage pointers and fresh direct state values after mutation.
- Invalid modes must fail before model allocation or any random draw, and M10 construction must not call CUDA/MPS/XPU seeding; Task 1 patches accelerator seed entry points and compares the next CPU random values.
- timm's dimensionality rule must not silently omit, duplicate, or misclassify the fixed model's CLS, GRU biases, LayerScale, or M10 frequency; Task 2 checks exhaustive identity coverage plus literal representative names and exact group counts.
- LambdaLR's initialization/step convention and restore order must not shift warmup or cosine by one update or scale from a non-protocol base LR; Task 3 checks exact boundary states and rejects wrong optimizer/base-LR inputs.

---

### Task 1: One-source paired initialization

**Files:**
- Create: `tests/test_clevr_training.py`
- Create: `src/clevr_training.py`

**Interfaces:**
- Consumes: `CLEVRQuestionAnswerer(vocab_size, mode=..., m10_seed=...)` and PyTorch `state_dict` / `load_state_dict`.
- Produces: `CLEVRTrainingConfigurationError(ValueError)` and `CLEVRPairedInitialization` constructor, `vocab_size`, `run_seed`, `m10_seed`, `common_state_dict()`, and `build(mode)`.
- Defers to Task 2: optimizer construction.
- Defers to Task 3: LR function and scheduler.

- [ ] **Step 1: Write failing initialization and isolation tests**

Create `tests/test_clevr_training.py` and add:

```python
class CLEVRPairedInitializationTests(unittest.TestCase):
    def test_builds_all_arms_from_one_exact_common_state(self): ...
    def test_m10_uses_exact_independent_seed_outside_common_state(self): ...
    def test_snapshots_and_built_models_do_not_alias(self): ...
    def test_rejects_invalid_inputs_and_seed_overflow_before_rng_change(self): ...
    def test_factory_and_builds_leave_global_rng_streams_unchanged(self): ...
```

Use `vocab_size=8`, `run_seed=101`, and literal `m10_seed=20101`. Require the
common snapshot to be a non-empty `OrderedDict` with no `position.` key. For
each of `G`, `B`, `M10`, and `Q-only`, compare the complete ordered non-position
state key sequence, shape, dtype, and every value with `torch.equal`. Compare
M10 `position.freqs` directly with a separately constructed
`CLEVRMixedPosition(seed=20101)` and require that name to be absent from common
state.

Mutate one returned snapshot tensor and one built model parameter under
`torch.no_grad()`, then prove a fresh snapshot and later arm retain the original
literal/direct values and distinct storage pointers. Invalid values cover bool,
float, NumPy integer, negative, `2**64 - 1 - 20000 + 1`, small vocabularies,
and non-string/unknown modes; compare CPU RNG before/after invalid calls.

For full RNG isolation, snapshot Python `random`, NumPy legacy state, and
PyTorch CPU state, patch CUDA/MPS/XPU seed functions to raise if called, then
construct the factory, take snapshots, and build all modes. Compare every RNG
state directly afterward; do not initialize or query a CUDA device.

- [ ] **Step 2: Run Task 1 tests and observe RED**

```bash
PYTHONPATH=/tmp/rope-clevr-training-config CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests/test_clevr_training.py -v
```

Expected: import failure because `src.clevr_training` does not exist.

- [ ] **Step 3: Implement the minimum paired initialization factory**

In `src/clevr_training.py`, define:

```python
class CLEVRTrainingConfigurationError(ValueError): ...


class CLEVRPairedInitialization:
    def __init__(self, *, vocab_size: int, run_seed: int) -> None: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def run_seed(self) -> int: ...
    @property
    def m10_seed(self) -> int: ...
    def common_state_dict(self) -> OrderedDict[str, Tensor]: ...
    def build(self, mode: str) -> CLEVRQuestionAnswerer: ...
```

Validate all scalar inputs before entering an RNG context. Construct the one G
source inside `torch.random.fork_rng(devices=[])` after calling only
`torch.random.default_generator.manual_seed(run_seed)`. Clone detached
non-position state into private storage. Build each arm in a fresh identical
CPU-only context, passing `m10_seed` only for M10. Compare target common keys,
shape, and dtype before replacing those entries in a complete target state;
load strictly and verify every copied value with `torch.equal` before return.

- [ ] **Step 4: Run Task 1 tests and observe GREEN**

Run the Step 2 command. Expected: all five Task 1 tests pass; only the existing
timm import warnings may appear.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/clevr_training.py tests/test_clevr_training.py
git commit -m "feat: add paired CLEVR model initialization"
```

---

### Task 2: Auditable AdamW parameter groups

**Files:**
- Modify: `tests/test_clevr_training.py`
- Modify: `src/clevr_training.py`

**Interfaces:**
- Consumes: Task 1 `CLEVRPairedInitialization.build(mode)` and timm `param_groups_weight_decay`.
- Produces: `build_clevr_adamw(model: CLEVRQuestionAnswerer) -> AdamW`; each optimizer group contains `group_name` and ordered `param_names` in addition to standard fields.
- Defers to Task 3: scheduler construction from the fixed-base-LR optimizer.

- [ ] **Step 1: Write failing optimizer partition tests**

Add:

```python
class CLEVRAdamWTests(unittest.TestCase):
    def test_groups_cover_every_trainable_parameter_once_with_names(self): ...
    def test_protocol_parameter_categories_have_exact_decay_and_hyperparameters(self): ...
    def test_qonly_none_gradient_leaves_patch_projection_unchanged_and_untracked(self): ...
    def test_rejects_non_answerer_input(self): ...
```

For G, M10, and Q-only, map optimizer parameters back to exact model parameter
identities. Require no duplicates, no missing trainable parameter, no frozen
parameter, and group counts `(67 no_decay, 29 decay)` for G/Q-only and
`(68 no_decay, 29 decay)` for M10. Require `param_names` to equal the parameter
identities' names in model order and to survive `optimizer.state_dict()`.

Check literal decay names including patch Conv, block-0 QKV/output/MLP, answer
head, Embedding, and both GRU weights. Check literal no-decay names including
CLS, representative/all biases, block-0 LayerNorm, both LayerScale values, and
M10 `position.freqs`. Assert AdamW, base LR `3e-4`, betas `(0.9,0.999)`, epsilon
`1e-8`, and group decay values `{0.0,0.05}`.

For Q-only, clone `backbone.patch_embed.proj` weight/bias, run one real
question-only cross-entropy backward and AdamW step, require both gradients to
remain `None`, both tensors to remain `torch.equal`, and both Parameters to be
absent from `optimizer.state` despite belonging to the decay/no-decay groups.

- [ ] **Step 2: Run Task 2 tests and observe RED**

Run the test-file command from Task 1. Expected: import or attribute failures
because `build_clevr_adamw` does not exist.

- [ ] **Step 3: Implement named timm groups and AdamW**

Add:

```python
def build_clevr_adamw(model: CLEVRQuestionAnswerer) -> AdamW: ...
```

Require an actual `CLEVRQuestionAnswerer`. Build the explicit skip set from
`backbone.cls_token` plus all trainable names beginning `position.`. Call
`timm.optim.param_groups_weight_decay(model, weight_decay=0.05,
no_weight_decay_list=skip)`, map returned parameter identities back to the
ordered trainable names, and reject unknown/duplicate/missing identities.
Normalize to one `decay` and one `no_decay` group with matching `param_names`,
then construct PyTorch AdamW with the frozen hyperparameters.

- [ ] **Step 4: Run Task 2 focused and model regressions**

```bash
PYTHONPATH=/tmp/rope-clevr-training-config CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests/test_clevr_training.py tests/test_clevr_model.py -v
```

Expected: all training-configuration and existing CLEVR model tests pass; only
the existing timm/vendored deprecation warnings may appear.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/clevr_training.py tests/test_clevr_training.py
git commit -m "feat: configure audited CLEVR AdamW groups"
```

---

### Task 3: Exact finite-budget LR and resumable scheduler state

**Files:**
- Modify: `tests/test_clevr_training.py`
- Modify: `src/clevr_training.py`

**Interfaces:**
- Consumes: Task 2 AdamW base LR `3e-4` and PyTorch `LambdaLR`.
- Produces: `clevr_learning_rate(update: int) -> float` and `build_clevr_lr_scheduler(optimizer: AdamW) -> LambdaLR`.
- Later runner contract: use current LR, perform successful `optimizer.step()`, then call `scheduler.step()` once; construct optimizer and scheduler before loading both saved states.

- [ ] **Step 1: Write failing formula, step-boundary, and restore tests**

Add:

```python
class CLEVRLearningRateTests(unittest.TestCase):
    def test_formula_matches_hand_derived_boundaries_and_rejects_invalid_updates(self): ...
    def test_lambda_scheduler_current_lr_tracks_the_next_update_without_off_by_one(self): ...
    def test_scheduler_state_restore_matches_uninterrupted_next_rates(self): ...
    def test_scheduler_rejects_non_adamw_or_wrong_base_lr(self): ...
```

Require exact/`math.isclose` values `eta_1=7.5e-7`, `eta_400=3e-4`,
`eta_4200=1.505e-4`, `eta_8000=1e-6`, and a direct standard-library cosine
calculation for update 401. Reject 0, 8001, negative, bool, float, and NumPy
integers.

Use a one-parameter CPU AdamW at the fixed base LR to make the scheduler test
independent of model compute. At construction require `last_epoch==0` and
current LR `eta_1`; after 1, 399, 400, 4199, 7999, and 8000 successful
optimizer/scheduler steps require the design's current next-update LR and
counter exactly. Checkpoint at 1000 steps, reconstruct optimizer and scheduler,
load both states, and compare at least the next ten rates/counters with an
uninterrupted pair. Reject SGD and AdamW whose current base LR is not `3e-4`
before constructing a scheduler.

- [ ] **Step 2: Run Task 3 tests and observe RED**

Run the Task 1 test-file command. Expected: import or attribute failures because
the LR functions do not exist.

- [ ] **Step 3: Implement the pure formula and LambdaLR adapter**

Add:

```python
def clevr_learning_rate(update: int) -> float: ...
def build_clevr_lr_scheduler(optimizer: AdamW) -> LambdaLR: ...
```

Implement the two exact branches from the design with `math.cos` and `math.pi`.
Validate ordinary integer type and `[1,8000]`. Require `type(optimizer) is
AdamW` or `isinstance(optimizer, AdamW)` and every current group LR equal to
`3e-4`. Construct LambdaLR with multiplier
`clevr_learning_rate(min(scheduler_epoch + 1, 8000)) / 3e-4`.

- [ ] **Step 4: Run focused and allowed CPU regressions**

Run the training test file; expected all tests pass. Then run the established
aggregate CPU suite with the new module, excluding only the two policy-forbidden
CIFAR hash modules. Expected: all allowed tests pass, one existing CUDA-only
test skips, and only existing dependency warnings remain.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/clevr_training.py tests/test_clevr_training.py
git commit -m "feat: add frozen CLEVR learning-rate schedule"
```

---

### Task 4: Scope audit, independent review, and real stacked PR

**Files:**
- Verify: `docs/superpowers/specs/2026-10-25-clevr-training-configuration-design.md`
- Verify: `docs/superpowers/plans/2026-10-25-clevr-training-configuration.md`
- Verify: `src/clevr_training.py`
- Verify: `tests/test_clevr_training.py`

**Interfaces:**
- Consumes: Tasks 1-3 committed implementation.
- Produces: reproducible CPU evidence and a real non-draft PR against `feat/clevr-qonly`; no training runner or checkpoint file.

- [ ] **Step 1: Run the complete committed-head allowed CPU suite**

Use `CUDA_VISIBLE_DEVICES=''` and the explicit flightgpt interpreter. Exclude
only `tests.test_cifar_training` and `tests.test_cifar_data` for the recorded
hash-policy reason. Expected: all allowed tests pass; the existing CUDA-only
test skips; only understood existing warnings remain.

- [ ] **Step 2: Audit branch scope and direct evidence**

```bash
git diff --check feat/clevr-qonly...HEAD
git diff --stat feat/clevr-qonly...HEAD
git diff --name-status feat/clevr-qonly...HEAD
git status --short --branch
```

Inspect the diff and confirm exactly the four planned files, no data/sampler
merge, hash, dependency, Dataset/DataLoader, training loop, checkpoint file,
runner/CLI, download, real-data access, or GPU launch.

- [ ] **Step 3: Obtain one fresh whole-branch review**

Give the reviewer exact base/head SHAs, design, plan, diff package, test
evidence, Review Focus, and ledger rulings. Require severity-ranked findings,
an independent parameter-membership audit, LR boundary/restore checks, and an
explicit Declined-to-judge list. Fix every re-graded Critical/Important item
once with RED-to-GREEN tests; record Minor items without expanding scope.

- [ ] **Step 4: Push and create the real stacked PR**

Push `feat/clevr-training-config` and create a non-draft PR against
`feat/clevr-qonly`. Include trigger, seed/state equations, mature-library and
license review, changes, RED/GREEN/full-suite evidence, independent review, and
explicit limitations. Verify PR state/base/head and remote/local SHA directly.
Do not merge or tag and do not claim CI unless an actual check exists.
