# CLEVR PUB-R1.1 Checkpoint Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Atomically save and transactionally restore one complete PUB-R1.1 optimizer-update boundary, including model, AdamW, LambdaLR, committed sample stream, and Python/NumPy/PyTorch CPU/explicit CUDA RNG state.

**Architecture:** `src/clevr_checkpoint.py` is a protocol-specific checkpoint boundary stacked on the real `CLEVRIndexStream`. It writes a pure primitive/tensor payload through `torch.save`, publishes a new file with same-parent staging plus no-overwrite hard-link commit, loads only with `weights_only=True`, validates the complete payload before mutation, then applies or rolls back all live state as one process-local transaction.

**Tech Stack:** Python 3.11.14 standard library (`copy`, `dataclasses`, `math`, `os`, `pathlib`, `platform`, `random`, `tempfile`), PyTorch 2.6.0+cu124, NumPy 2.2.6, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-checkpoint-design.md`

## Global Constraints

- PUB-R1.1 is the only protocol. No data/weight download, RGB/question/scene access, DataLoader, forward/backward/optimizer update, training, evaluation, GPU work, process monitoring, or scientific claim.
- This branch is based on `feat/clevr-paired-sampler`; do not merge or copy model/data/training-config/reporting branches. The later runner owns those integrations.
- Public API is exactly `CLEVRCheckpointError`, `CLEVRCheckpointPublicationError(committed)`, frozen `CLEVRCheckpointMetadata`, `save_clevr_checkpoint`, and `load_clevr_checkpoint` from the spec.
- Payload contains only tensors and `weights_only=True`-compatible primitive/container values. Load always passes `map_location="cpu", weights_only=True`; never allowlist custom globals and never fall back to unrestricted pickle.
- `method ∈ {G,B,M10,Q-only}`, nonnegative ordinary-int seed, ordinary-int `0 <= completed_updates <= 8000`, and exact protocol/schema/environment fields. Bools and NumPy integer aliases are not ordinary ints.
- Complete-boundary invariant is `epoch_index * dataset_size + cursor == 128 * completed_updates`; sampler seed/size must match; every model gradient is `None`.
- Scheduler must reference the supplied optimizer; checkpoint `last_epoch == completed_updates`, `_step_count == completed_updates + 1`, and `_last_lr` equals optimizer group `lr`. The later runner separately verifies the frozen learning-rate formula.
- `cuda_devices` is an explicit, strictly increasing tuple of unique nonnegative ordinary ints. Empty tuple performs no CUDA call. Current verification uses `CUDA_VISIBLE_DEVICES=''` and never claims CUDA coverage.
- Save targets a new file. Same-parent staging is mode 0600; hard-link creation is the commit point; existing file/directory/symlink is preserved; post-commit cleanup never removes final output. No hash/digest is computed.
- Load validates before mutation and catches `BaseException` during application. Success restores all state; ordinary failure restores every live object and RNG to its entry state; rollback failure is reported explicitly rather than hidden.
- Exact recovery is conditional on Python 3.11.14, PyTorch 2.6.0+cu124, NumPy 2.2.6, matching CUDA runtime string, and compatible hardware. No cross-version/host bitwise claim, `fsync` durability, SIGKILL cleanup, stale-file recovery, hostile-input resource bound, or network-filesystem guarantee.

## Review Focus

- **Prefetch or fabricated update count:** a state with valid sampler internals but `eD+c != 128k` must fail before publication/load and leave all live state unchanged (Tasks 1–2 tests).
- **Cross-spliced optimizer/scheduler:** same-shaped model weights with different optimizer `param_names`/order or scheduler counter/LR state must not load by positional accident (Task 2 tests).
- **Commit-return ambiguity:** a wrapper that performs the real hard link and then raises must report `committed=True`, retain a complete target, and clean only this invocation's staging (Task 1 tests).
- **Mid-apply BaseException:** model/scheduler/optimizer/sampler/RNG failure after earlier mutations must restore direct entry-state equality; rollback failure must be surfaced as such (Task 2 tests).
- **Hidden CUDA initialization:** `cuda_devices=()` must make zero CUDA RNG calls; malformed/nonmatching explicit CUDA state must fail without silently using another device (Tasks 1–2 tests).

---

### Task 1: Strict Snapshot and Atomic New-File Publication

**Files:**
- Create: `src/clevr_checkpoint.py`
- Create: `tests/test_clevr_checkpoint.py`

**Interfaces:**
- Consumes: `src.clevr_sampling.CLEVRIndexStream`; PyTorch `nn.Module`, `AdamW`, `LambdaLR`; global Python/NumPy/PyTorch RNG.
- Produces:
  - `CLEVRCheckpointError(ValueError)`;
  - `CLEVRCheckpointPublicationError(CLEVRCheckpointError)` with `committed: bool`;
  - frozen `CLEVRCheckpointMetadata(path: str, size_bytes: int, method: str, run_seed: int, completed_updates: int, dataset_size: int, cuda_devices: tuple[int, ...])`;
  - `save_clevr_checkpoint(path: Path, *, method: str, run_seed: int, completed_updates: int, model: nn.Module, optimizer: AdamW, scheduler: LambdaLR, sampler: CLEVRIndexStream, cuda_devices: tuple[int, ...]) -> CLEVRCheckpointMetadata`.

- [ ] **Step 1: Write happy-path snapshot tests before production code**

Add a deterministic tiny fixture with an `nn.Sequential` model, two named AdamW groups (`decay`, `no_decay`) carrying literal `param_names`, one LambdaLR, and real `CLEVRIndexStream(dataset_size=5, run_seed=101)`. Complete one synthetic optimizer/scheduler step, clear gradients with `set_to_none=True`, and commit exactly 128 previewed sampler indices.

Add `CheckpointSaveTests.test_saves_complete_weights_only_payload_at_update_boundary`. Save method `G`, seed101, update1, `cuda_devices=()`. Load the artifact directly with `torch.load(..., map_location="cpu", weights_only=True)` and assert the exact top-level/RNG/environment key sets and literal protocol/schema/metadata. Assert model tensors, optimizer/scheduler fields, sampler state, Python/NumPy/PyTorch CPU states, mode0600 staging discipline through the final inode, metadata resolved path/size, no staging files, and zero CUDA RNG calls. Mutate every live tensor/state after save and prove the loaded payload does not alias it.

Add `test_boundary_hand_example_preserves_next_committed_indices`: use the spec's `D=5`, seed101 state, assert consumed `128`, and assert the payload sampler restored into a fresh stream previews the same literal next prefix as the uninterrupted stream. The production mutation named in the comment is saving preview/dispatched position instead of committed position.

- [ ] **Step 2: Run the Task 1 happy path and verify RED**

Run:

```bash
PYTHONPATH=/tmp/rope-clevr-checkpoint CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests.test_clevr_checkpoint.CheckpointSaveTests.test_saves_complete_weights_only_payload_at_update_boundary \
  tests.test_clevr_checkpoint.CheckpointSaveTests.test_boundary_hand_example_preserves_next_committed_indices -v
```

Expected: import error because `src.clevr_checkpoint` does not exist.

- [ ] **Step 3: Implement strict snapshot codecs and successful publication**

Implement the Task 1 public records/API and private helpers in `src/clevr_checkpoint.py`. Keep `PROTOCOL_VERSION="PUB-R1.1"`, `CHECKPOINT_SCHEMA_VERSION=1`, effective batch128, and total updates8000 as named constants. Snapshot tensors with `detach().cpu().clone()` through a recursive copier that accepts only exact primitive/container/tensor types required by PyTorch state dicts.

Encode NumPy legacy MT19937 state as exact fields `bit_generator,keys,pos,has_gauss,cached_gaussian`, with keys as 624-element CPU int64 tensor in `[0,2**32)`. Validate Python state on a private `random.Random`, NumPy state on a private `numpy.random.RandomState`, and CPU torch state on a private CPU `torch.Generator`, never by mutating globals. Empty `cuda_devices` must bypass all CUDA RNG APIs.

Validate model gradients, run/sampler identity, `eD+c=128k`, optimizer named-group coverage/order, scheduler ownership/count/current LR, exact environment strings, and weights-only payload schema before file I/O. Parent must already be a directory and target must not exist or be a symlink.

Write to `tempfile.mkstemp` in the target parent with mode0600, close a flushed binary descriptor after `torch.save`, capture staging inode, hard-link to target, classify the link as committed, and unlink staging. Return metadata from the final target. Do not add fsync, hashes, auto-discovery, overwrite, or parent creation.

- [ ] **Step 4: Run happy-path tests and verify GREEN**

Run the Step 2 command. Expected: both tests PASS and no CUDA access.

- [ ] **Step 5: Write strict-input and publication-failure tests; verify RED**

Add focused tests that name the production mutation they catch:

- public records are frozen and reject direct inconsistent construction;
- wrong object types, method, bool/negative/overflow seed/update, unordered/duplicate/bool CUDA devices, nonempty CUDA devices before CUDA initialization, scheduler not owning optimizer, non-None gradient, sampler seed/size mismatch, consumed count one before/after `128k`, scheduler counter or current-LR mismatch, missing/non-directory parent all fail before creating staging/final;
- existing file, directory, and symlink remain byte/state unchanged;
- injected `torch.save`, descriptor flush/close, and ordinary `os.link` failures leave no final/staging;
- a noncooperating target appearing before link is preserved;
- wrapper calls real `os.link` then raises `KeyboardInterrupt`: exception is `CLEVRCheckpointPublicationError(committed=True)`, target is complete, no staging remains;
- staging unlink failure before commit plus a writer error reports `committed=False`; unlink failure after commit reports `committed=True`, retains complete final and own staging; neither deletes an existing target;
- an interrupt after staging descriptor acquisition closes it; short writes are delegated to `torch.save`/buffered file rather than reimplementing serialization.

Run:

```bash
PYTHONPATH=/tmp/rope-clevr-checkpoint CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests.test_clevr_checkpoint.CheckpointSaveValidationTests \
  tests.test_clevr_checkpoint.CheckpointPublicationTests -v
```

Expected: failures/errors at unimplemented validation and commit/cleanup paths, while the Step 4 happy path remains green.

- [ ] **Step 6: Implement validation and complete publication transaction**

Add only the behavior required by Step 5. Catch `BaseException` after staging acquisition. On an ambiguous link exception, classify committed only if target and staging are regular non-symlink paths with the captured identical `(st_dev, st_ino)`. Before commit remove only own staging; after commit never unlink final. `CLEVRCheckpointPublicationError` must preserve `committed`, primary cause, and cleanup notes without replacing an existing path.

- [ ] **Step 7: Run Task 1 and sampler regressions**

Run all `tests.test_clevr_checkpoint` save/publication classes and `tests.test_clevr_sampling`; expected all PASS with `CUDA_VISIBLE_DEVICES=''` and no unexpected warnings.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/clevr_checkpoint.py tests/test_clevr_checkpoint.py
git commit -m "feat: publish complete CLEVR checkpoints"
```

---

### Task 2: Strict Transactional Restore

**Files:**
- Modify: `src/clevr_checkpoint.py`
- Modify: `tests/test_clevr_checkpoint.py`

**Interfaces:**
- Consumes: Task 1 payload/metadata/private validators and public save function.
- Produces: `load_clevr_checkpoint(path: Path, *, expected_method: str, expected_run_seed: int, model: nn.Module, optimizer: AdamW, scheduler: LambdaLR, sampler: CLEVRIndexStream, cuda_devices: tuple[int, ...]) -> CLEVRCheckpointMetadata`.

- [ ] **Step 1: Write exact successful restore test before implementation**

Add `CheckpointLoadTests.test_restores_all_live_state_and_next_observations_exactly`. Save a valid update-boundary fixture, record direct payload state, uninterrupted next 128 sampler indices, and next Python/NumPy/PyTorch CPU random values. Perturb model/optimizer/scheduler/sampler and all global RNG, then load into same-structure live objects. Assert returned metadata, every model tensor, recursive optimizer/scheduler state (tensor leaves by `torch.equal`), sampler state/next indices, and next RNG values exactly match. Assert explicit `weights_only=True`, `map_location="cpu"`, scheduler-then-optimizer semantics, and no CUDA call for `cuda_devices=()` through observable behavior rather than only mock counts.

- [ ] **Step 2: Run successful restore test and verify RED**

Run:

```bash
PYTHONPATH=/tmp/rope-clevr-checkpoint CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests.test_clevr_checkpoint.CheckpointLoadTests.test_restores_all_live_state_and_next_observations_exactly -v
```

Expected: import/attribute failure because `load_clevr_checkpoint` is missing.

- [ ] **Step 3: Implement read, preflight validation, and successful restore**

Require an explicit `Path` resolving to a regular checkpoint file. Call `torch.load` exactly with CPU map and restricted weights-only mode. Validate exact top-level/nested fields, types, protocol/schema, environment, expected method/seed, CUDA device tuple, model key/shape/dtype, optimizer named groups/order, scheduler ownership/count/LR, sampler through a fresh `CLEVRIndexStream`, every RNG state through private probes, and `eD+c=128k` before live mutation.

Capture deep live snapshots. Apply model → scheduler → optimizer → sampler → Python → NumPy → torch CPU → each explicit CUDA RNG state. Directly verify all resulting state against the canonical validated payload and return metadata for the resolved input file.

- [ ] **Step 4: Run successful restore test and verify GREEN**

Run Step 2 again. Expected: PASS.

- [ ] **Step 5: Write malformed/cross-spliced payload tests and verify RED**

Create independent files with `torch.save` from a valid payload, mutate one field at a time, and assert rejection before live state/RNG changes for:

- absent/not-file input, unrestricted/custom-class payload, top-level and every nested missing/extra key;
- target model carrying any non-`None` gradient, proving a half accumulation window is rejected before any live mutation;
- wrong protocol/schema/environment; caller method/seed mismatch; bool/NumPy scalar numeric fields;
- wrong model key/shape/dtype/non-tensor;
- optimizer group count/name/`param_names`/order mismatch, including a same-shaped cross-splice;
- scheduler `last_epoch`, `_step_count`, `_last_lr`, or optimizer link mismatch;
- sampler internal invalidity and otherwise valid state with consumed count unequal to `128k`;
- malformed Python state, NumPy MT19937 name/key count/range/dtype/position/cache fields, CPU RNG dtype/rank, and CUDA device/state count/order/dtype mismatch.

The custom-class test must prove its reducer is not executed; it is evidence for `weights_only=True`, not a request to relax the loader.

Run all strict-load tests; expected focused failures until validators are complete.

- [ ] **Step 6: Complete strict preflight validation**

Implement only missing checks from Step 5. Do not coerce bools, NumPy scalars, lists to tuples, float seeds, state shapes, or environment versions. Do not add unsafe globals, fallback pickle, digest checks, resource thresholds, auto-discovery, or migration of old schema.

- [ ] **Step 7: Write application/rollback fault tests and verify RED**

Add real-state tests that inject one first-call failure after mutation at each boundary: model, scheduler, optimizer, sampler, Python RNG, NumPy RNG, torch CPU RNG, and (using a local fake adapter only around PyTorch CUDA RNG calls) one explicit CUDA device. Each primary `Exception` and `KeyboardInterrupt` must leave model/optimizer/scheduler/sampler and all RNG exactly at function-entry state. The test records direct state and next outputs before calling; it does not assert only call order.

Add one rollback-failure test: primary apply failure followed by injected rollback failure must raise `CLEVRCheckpointError` with the primary as cause and a rollback note; it must not claim atomic restoration. Run these tests and observe RED at the missing rollback behavior.

- [ ] **Step 8: Implement rollback and post-restore verification**

Catch `BaseException` only after entry snapshots exist. Restore all components even if one rollback action fails, collect rollback failures, and directly verify rollback equality when cleanup succeeds. Re-raise the untouched primary error only when rollback is proven complete; otherwise raise `CLEVRCheckpointError` chained from primary with every rollback failure noted. Successful application also performs the same recursive equality checks.

- [ ] **Step 9: Run Task 2 and allowed CPU regressions**

Run:

```bash
PYTHONPATH=/tmp/rope-clevr-checkpoint CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests.test_clevr_checkpoint tests.test_clevr_sampling -v
```

Then run the established allowed suite:

```bash
PYTHONPATH=/tmp/rope-clevr-checkpoint CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  tests.test_attention_compatibility tests.test_classification \
  tests.test_clevr_audit tests.test_clevr_partitions \
  tests.test_clevr_programs tests.test_clevr_sampling \
  tests.test_clevr_checkpoint tests.test_clevr_schema \
  tests.test_clevr_text tests.test_local_rope tests.test_segmentation -v
```

Expected: every runnable CPU test PASS, exactly one existing CUDA-only test SKIP, and only the two understood timm deprecation warnings.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/clevr_checkpoint.py tests/test_clevr_checkpoint.py
git commit -m "feat: restore CLEVR checkpoints transactionally"
```

---

### Task 3: Fresh Review, Final Verification, and Stacked PR Handoff

**Files:**
- Modify only if a reviewer-confirmed Critical/Important defect requires the single permitted final TDD fix wave.

**Interfaces:**
- Consumes: committed Tasks 1–2.
- Produces: a reviewable branch and real non-draft stacked PR targeting `feat/clevr-paired-sampler` when GitHub PR creation is available.

- [ ] **Step 1: Dispatch fresh whole-branch review**

Give a fresh reviewer the spec, plan, base-to-head package, and explicit focus on committed-vs-preview sampler position, exact environment/run identity, optimizer/scheduler cross-splice, weights-only enforcement, CUDA no-touch empty path, hard-link commit reconciliation, and BaseException rollback. Require spec-compliance and code-quality verdicts plus independent focused CPU tests; no GPU or training.

- [ ] **Step 2: Resolve findings once**

For Critical/Important findings, use one fix implementer for the complete list, strict defect RED before production change, one scoped re-review, and no second fix wave. Record Minor/out-of-scope observations and every ruling without silently broadening into runner/data/model behavior.

- [ ] **Step 3: Run final evidence from committed HEAD**

Run the Task 2 focused and allowed suites, then:

```bash
PYTHONPATH=/tmp/rope-clevr-checkpoint CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m compileall -q \
  src tests
git diff --check 54d1f00959fa7ac1f3dcab1650b7a3fd93d06343...HEAD
git status --short --branch
git log --oneline 54d1f00959fa7ac1f3dcab1650b7a3fd93d06343..HEAD
git diff --name-only 54d1f00959fa7ac1f3dcab1650b7a3fd93d06343...HEAD
```

Expected diff contains only the checkpoint design, plan, module, and test file. Do not run real data, model training, evaluation, GPU, or CIFAR hash-policy tests.

- [ ] **Step 4: Push and create the real stacked PR when possible**

Push `feat/clevr-checkpoint`; create a non-draft PR with base `feat/clevr-paired-sampler`. PR body includes: trigger; state-vector and complete-boundary equation; official/local library evidence and licenses; exact changes; TDD/review/test evidence; CPU-only/CUDA-unverified status; hard-link/no-fsync/SIGKILL/DoS/cross-version limits; and explicit later-runner integration gates. Verify remote/local head equality and PR state/base/head. Do not merge, tag, or claim CI. If the PR interface remains unavailable, retain the pushed branch and exact stacked compare URL and continue under the user's skip-blocker instruction.
