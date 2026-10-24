# CLEVR PUB-R1.1 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the accepted PUB-R1.1 data, four model arms, paired training configuration, committed sampler, checkpoint recovery, and prediction format into an explicit user-launched runner without executing an experiment.

**Architecture:** Add a model-safe provenance accessor, a separate immutable/append-only artifact layer, a dependency-injected update/evaluation engine, and a thin production assembler plus CLI. Training indexes datasets directly from the committed sampler, while formal base-test evaluation remains a separate command.

**Tech Stack:** Python 3.11 standard library; PyTorch 2.6.0; torchvision 0.21.0; timm 1.0.24; Pillow 12.0.0; NumPy 2.2.6; `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-runner-design.md`

## Global Constraints

- Work only on `feat/clevr-runner` in `/tmp/rope-clevr-runner`; do not modify `master` or prerequisite feature branches.
- Before Task 1, the controller must merge exact prerequisite heads `feat/clevr-qonly-data@340fb33`, `feat/clevr-training-config@b48526b`, `feat/clevr-checkpoint@764ff45`, and `feat/clevr-reporting@d4a1b74` into this integration branch and record the merge commits in the ledger.
- Model batches remain exact: RGB has `image,question_tokens,question_length,answer`; Q-only has `question_tokens,question_length,answer`. IDs and audit fields never enter a sample, collated batch, model call, loss, or gradient.
- Fixed phases are smoke `(seed=9001, updates=1)`, pilot `(seed=9001, updates=800)`, and formal `(seed in 101..105, updates=8000)`; there is no arbitrary evidence budget.
- Each optimizer update is exactly 128 samples split into eight contiguous micro-batches of 16; each mean CE is divided by 8; clip the global gradient norm at 1.0; call optimizer, scheduler, clear gradients, then commit all 128 indices.
- Use direct map-style indexing and the existing collators; do not introduce DataLoader prefetch or another sample-order owner.
- Fresh checkpoints are update 0, each 1000 updates, and the final update; checkpoint paths are new-file-only. Resume is only from a caller-explicit checkpoint.
- Training evaluates dev at every 1000 updates and at pilot update 800; formal update 8000 also evaluates the complete train partition. Training never evaluates base_test or confirmation.
- Formal base-test publication is a separate explicit operation and only accepts a formal seed plus update-8000 checkpoint. Confirmation evaluation is absent.
- Production train/evaluate require one explicit CUDA index, FP32, AMP absent, CUDA/cuDNN TF32 disabled. Unit tests use injected CPU doubles and mocks and do not weaken this production gate.
- No downloads, data/weight discovery, user-data access, GPU work, engineering smoke, pilot, formal training, or evaluation may be executed while implementing this plan.
- Do not add dependencies, hash checks, model parameters, thresholds, label changes, alternate losses, or statistics changes.
- Follow strict TDD: every production behavior gets a focused failing test observed before implementation, then focused green and the allowed CPU regression suite. Exclude `tests.test_cifar_data` and `tests.test_cifar_training` because project policy forbids their legacy hash logic.
- Use `/workspace/smbu_lt/my_envs/flightgpt/bin/python3.11` and `PYTHONDONTWRITEBYTECODE=1` for all tests. Do not create bytecode artifacts.

## Review Focus

- Python negative, bool, non-index, and out-of-range dataset indices must have the same lookup/exception behavior for `__getitem__` and `sample_identity`; Task 1 adds direct paired tests without redefining the existing dataset indexing contract.
- Concurrent or interrupted attempt allocation must never make two writers share a journal or expose partial immutable JSON/prediction files; Task 2 adds race/fault-injection tests around exclusive files, link publication, flush, and cleanup.
- A failure in any of eight micro-batches, clipping, optimizer, scheduler, or commit must not falsely report an update or advance the sampler before the commit point; Task 3 adds one failure test at each boundary.
- Evaluation must restore the model's prior train/eval flag even when dataset access or forward raises, and aggregate by question rather than batch; Task 3 tests uneven final batches, ties, and exceptional restoration.
- Resume must reject a checkpoint/config/dataset/method/seed/target mismatch before an update, and a training command must never construct base_test/confirmation; Task 4 tests every mismatch and records the called partitions.

---

### Task 1: Add a model-safe dataset provenance surface

**Files:**
- Modify: `src/clevr_dataset.py`
- Modify: `tests/test_clevr_dataset.py`

**Interfaces:**
- Consumes: existing validated `_RuntimeRecord`, `CLEVRQuestionDataset`, and `CLEVRQuestionOnlyDataset` ordering.
- Produces: immutable `CLEVRSampleIdentity`; `vocab_size: int` and `sample_identity(index: int) -> CLEVRSampleIdentity` on both public datasets.
- Defers: no artifact writing, model call, runner state, or CLI.

- [ ] **Step 1: Write failing identity/value tests**

Add tests proving both datasets expose equal `vocab_size`, return equal identities in deterministic sorted order, and the identity has exactly `partition,official_split,image_index,image_filename,question_index,source_position`. Prove it is frozen and that mutating a returned value is impossible.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python3.11 -m unittest -v tests.test_clevr_dataset
```

Expected: FAIL because `CLEVRSampleIdentity`, `vocab_size`, and `sample_identity` do not exist.

- [ ] **Step 3: Implement the immutable identity projection**

Add:

```python
@dataclass(frozen=True, slots=True)
class CLEVRSampleIdentity:
    partition: str
    official_split: str
    image_index: int
    image_filename: str
    question_index: int
    source_position: int
```

Retain those values in `_RuntimeRecord`, retain `len(vocabulary.id_to_token)` as an ordinary integer, and return a newly constructed identity without resolving/opening an image.

- [ ] **Step 4: Add failing boundary/leakage tests**

Test `-1`, invalid positive/negative bounds, bool/non-index inputs according to the existing tuple indexing behavior, no `Image.open` from Q-only identity access, and unchanged exact sample/collate keys after the accessor exists.

- [ ] **Step 5: Implement consistent indexing and keep batch keys unchanged**

Use the same private tuple lookup for `__getitem__` and `sample_identity`; do not catch or coerce its `IndexError`/`TypeError`. Keep all audit fields out of `__getitem__` and both collators.

- [ ] **Step 6: Verify focused and regression GREEN**

Run the focused test above, then the allowed CPU suite including all merged CLEVR/model/sampler/checkpoint/reporting tests. Expected: all pass with only the known timm deprecation warnings and expected CUDA skip.

- [ ] **Step 7: Commit**

```bash
git add src/clevr_dataset.py tests/test_clevr_dataset.py
git commit -m "feat: expose CLEVR sample provenance safely"
```

### Task 2: Add immutable run artifacts and per-attempt journals

**Files:**
- Create: `src/clevr_run_artifacts.py`
- Create: `tests/test_clevr_run_artifacts.py`

**Interfaces:**
- Consumes: `Path`, strict JSON-compatible mappings, Task 1 identities only indirectly through prediction records.
- Produces: `CLEVRRunArtifactError`, `CLEVRRunSpec`, `CLEVRPredictionRecord`, `CLEVRRunArtifacts.open(...)`, `ensure_common_state(...)`, `append_event(...)`, `checkpoint_path(...)`, `prediction_path(...)`, `publish_predictions(...)`, and `close()`.
- Defers: no torch/model/dataset construction, checkpoint serialization, update loop, or device selection.

- [ ] **Step 1: Write failing run-spec tests**

Test exact accepted triples `("smoke",9001,1)`, `("pilot",9001,800)`, and each `("formal",seed,8000)` for seeds 101--105; exact methods; derived evidence role; rejection of booleans, NumPy scalar aliases, unknown strings, conflicting explicit target updates, and confirmation semantics.

- [ ] **Step 2: Verify run-spec RED, then implement `CLEVRRunSpec`**

Use a frozen slots dataclass with fields `phase: str`, `method: str`, `seed: int` and read-only `target_updates` and `evidence_role` properties. Values come from one literal phase table; no caller-configurable budget.

- [ ] **Step 3: Write failing fresh/resume artifact tests**

Define this constructor:

```python
@classmethod
def open(
    cls,
    *,
    output_root: Path,
    spec: CLEVRRunSpec,
    expanded_config: Mapping[str, object],
    resume: bool,
) -> "CLEVRRunArtifacts": ...
```

Test canonical run directory `<root>/<phase>/seed_<seed>/<method>`, immutable `run_config.json`, exact-config validation on resume, fresh-run refusal of an existing target, resume refusal of a missing target/config, strict JSON (`allow_nan=False`), and no symlink/file substitution for the run directory.

- [ ] **Step 4: Verify artifact RED, then implement new-file config publication**

Serialize complete bytes before publication. Use a private same-parent mode-0600 staging file and `os.link` commit so readers see no partial immutable file and existing targets are not overwritten. Clean only staging inodes owned by this call; report whether publication committed when reconciliation is ambiguous.

- [ ] **Step 5: Write failing per-attempt journal tests**

Test exclusive monotonic allocation of `attempts/attempt_0001.jsonl`, then 0002; exact common fields `protocol_version,phase,method,seed,attempt,completed_updates,event,details`; one compact strict-JSON line per append; explicit flush after each line; rejection after close; and two concurrently opened attempts receiving different files.

- [ ] **Step 6: Implement journals and path helpers**

`checkpoint_path(update)` returns `<run>/checkpoints/update_<four digits>.pt` only for 0..target and creates only its parent. `prediction_path(partition, update)` accepts only train/dev at valid run updates and returns a new target under `predictions/<partition>/`; base-test publication uses the dedicated formal method below.

- [ ] **Step 7: Write failing common-state artifact tests**

Test `ensure_common_state(common_state: Mapping[str, Tensor]) -> Path` publishes `<root>/<phase>/seed_<seed>/common_state.pt` once with protocol, seed, exact locked environment, ordered cloned CPU tensors, restricted `torch.load(...,weights_only=True)` compatibility, and no hash. A second method/attempt must validate ordered keys, shape, dtype, tensor equality, and environment; mismatch or malformed/untrusted payload fails before any run update. Exercise non-overwrite and writer/link cleanup faults.

- [ ] **Step 8: Implement common-state publication and exact verification**

Use the same atomic new-file publisher around `torch.save`; never load modules or use `weights_only=False`. Snapshot input tensors before writing and compare with `torch.equal` on read. Concurrent equal publishers may accept the winner only after loading and verifying it; a conflicting winner is a hard error.

- [ ] **Step 9: Write failing prediction-publication tests**

Add frozen `CLEVRPredictionRecord(question_index:int,label:int,logits:tuple[float,float])`. Test ordinary nonnegative IDs, binary label, two finite floats, unique IDs, sort by question ID, exact six-key rows, canonical formal base-test filename `<method>_seed_<seed>.jsonl`, formal-only/update-8000 gate, and non-overwrite/cleanup under serialization, flush, link, and unlink failures.

- [ ] **Step 10: Implement `publish_predictions`**

Signature:

```python
def publish_predictions(
    self,
    *,
    partition: str,
    completed_updates: int,
    rows: Sequence[CLEVRPredictionRecord],
) -> Path: ...
```

Use the same atomic new-file publisher. For `base_test`, require formal/update 8000 and use the reporter-compatible canonical file. For train/dev, include update in the filename and accept only protocol-required evaluation boundaries.

- [ ] **Step 11: Verify focused and regression GREEN, then commit**

Run `tests.test_clevr_run_artifacts`, then the allowed CPU suite. Commit:

```bash
git add src/clevr_run_artifacts.py tests/test_clevr_run_artifacts.py
git commit -m "feat: record CLEVR run artifacts immutably"
```

### Task 3: Implement the committed update and evaluation engine

**Files:**
- Create: `src/clevr_runner.py`
- Create: `tests/test_clevr_runner.py`

**Interfaces:**
- Consumes: Task 1 dataset identity/sample APIs, existing collators, `CLEVRIndexStream`, PyTorch model/AdamW/LambdaLR APIs, and Task 2 prediction records.
- Produces: `CLEVRRunnerError`, `CLEVRUpdateResult`, `CLEVREvaluationResult`, `run_clevr_update(...)`, and `evaluate_clevr_model(...)`.
- Defers: production CUDA/config/model/checkpoint/artifact assembly to Task 4.

- [ ] **Step 1: Write failing update-order and loss tests**

Use a tiny CPU dataset of at least 129 indexable samples, a matching identity method, a small answerer double with the production forward signature, real AdamW/LambdaLR, and real `CLEVRIndexStream`. Prove one call obtains exactly `preview(128)`, loads eight contiguous groups of 16, computes the arithmetic mean of the 128 per-example CE values through eight `mean/8` terms, clips once at 1.0, uses the pre-step LR, executes optimizer then scheduler then `zero_grad(set_to_none=True)`, commits exactly the preview, and returns matching dataset/question IDs.

- [ ] **Step 2: Verify update RED, then implement `run_clevr_update`**

Signature:

```python
def run_clevr_update(
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    dataset: object,
    collate_fn: Callable[[Sequence[Mapping[str, Tensor]]], Mapping[str, Tensor]],
    method: str,
    device: torch.device,
) -> CLEVRUpdateResult: ...
```

Require the two optimizer groups to expose one common finite LR. RGB calls `model(image, tokens, lengths)`; Q-only requires no image key and calls `model(None, tokens, lengths)`. Move only model fields to the explicit device.

- [ ] **Step 3: Add failing update-boundary tests**

Inject failure at micro-batches 1 and 8, clipping, optimizer, scheduler, and commit. Prove no success result is returned, no commit occurs before the final commit call, and commit mismatch cannot be reported as complete. Test wrong batch keys/shapes, non-finite loss/logits/gradient norm, wrong method/image contract, inconsistent group LRs, and Q-only patch parameters retaining `grad is None` when unused.

- [ ] **Step 4: Implement strict update failures without rollback claims**

Always start with `optimizer.zero_grad(set_to_none=True)`. Validate logits `[16,2]`, labels, finite loss, and result metadata. Do not attempt to roll back model/optimizer after a failing optimizer/scheduler call; leave recovery to the last explicit checkpoint and never append a completed-update record from this function.

- [ ] **Step 5: Write failing evaluation tests**

Define:

```python
def evaluate_clevr_model(
    *,
    model: nn.Module,
    dataset: object,
    collate_fn: Callable[..., Mapping[str, Tensor]],
    method: str,
    partition: str,
    device: torch.device,
) -> CLEVREvaluationResult: ...
```

Test ascending indices, fixed maximum batch 16 with an uneven final batch, `inference_mode`, exact sum-reduction CE/total-question mean, micro accuracy, tie-to-zero, finite CPU Python logits, unique question IDs, labels matching batch values, Q-only no-image behavior, and exact rows sorted by question ID.

- [ ] **Step 6: Implement evaluation and exceptional mode restoration**

Record `was_training`, call `eval()`, and restore `train(was_training)` in `finally`, including when item access, collation, or forward raises. Reject empty datasets and duplicate identities. Do not publish files here.

- [ ] **Step 7: Verify focused and regression GREEN, then commit**

Run `tests.test_clevr_runner`, then the allowed CPU suite. Commit:

```bash
git add src/clevr_runner.py tests/test_clevr_runner.py
git commit -m "feat: execute committed CLEVR updates"
```

### Task 4: Assemble production training, recovery, evaluation, and CLI

**Files:**
- Modify: `src/clevr_runner.py`
- Modify: `tests/test_clevr_runner.py`
- Create: `run_clevr_experiment.py`
- Create: `tests/test_run_clevr_experiment.py`

**Interfaces:**
- Consumes: all prior tasks; `CLEVRPairedInitialization`, `build_clevr_adamw`, `build_clevr_lr_scheduler`, `save_clevr_checkpoint`, and `load_clevr_checkpoint`.
- Produces: `preflight_clevr_run(...)`, `train_clevr_seed(...)`, `evaluate_clevr_checkpoint(...)`, and CLI subcommands `inspect`, `train`, `evaluate`.

- [ ] **Step 1: Write failing production configuration/device tests**

Test explicit existing absolute `audit_dir`, explicit `data_root` iff any RGB arm is requested, non-empty duplicate-free methods in canonical `G,B,M10,Q-only` order, explicit nonnegative CUDA index, and no CUDA query/initialization before all path/spec checks pass. Test expanded config contains exact protocol/model/training constants, locked package versions, device/backend/determinism flags, and no hash.

- [ ] **Step 2: Implement preflight and explicit runtime setup**

`preflight_clevr_run` constructs only the required train/dev datasets and verifies equal lengths, vocabularies, and every sample identity projection between RGB and Q-only when both are requested; it accesses no item tensor. Runtime setup then requires the requested CUDA device, sets it explicitly, disables CUDA/cuDNN TF32, and returns `torch.device(f"cuda:{index}")`. Do not auto-select or fall back to CPU.

- [ ] **Step 3: Write failing fresh/resume orchestration tests**

With production builders patched by strict fakes, prove `train_clevr_seed` creates one paired factory per seed, atomically publishes/verifies its exact common state before any arm update, and builds one arm at a time. It reseeds global Python/NumPy/PyTorch/selected CUDA streams before each fresh arm, builds the method-correct train/dev datasets and collator, saves update 0, runs to the fixed target, writes update events only after successful committed updates, checkpoints each 1000 and final, evaluates dev at required boundaries, evaluates train only at formal 8000, publishes predictions/summary events, and emits a final completion event.

For resume, require a per-method explicit checkpoint mapping, construct all objects first, load transactionally, require metadata method/seed/dataset/update/target consistency, start at `completed_updates+1`, preserve earlier attempt files, and never recreate update 0. Test interrupted attempts append failure information when possible and re-raise `KeyboardInterrupt`/exceptions.

- [ ] **Step 4: Implement `train_clevr_seed`**

Signature:

```python
def train_clevr_seed(
    *,
    spec_phase: str,
    seed: int,
    methods: tuple[str, ...],
    audit_dir: Path,
    data_root: Path | None,
    output_root: Path,
    device_index: int,
    resume_checkpoints: Mapping[str, Path],
) -> None: ...
```

Use Task 3 for computation and Task 2 for all records. Call the reviewed checkpoint API only after gradients are None and sampler commit is complete. Never instantiate base_test or confirmation here.

- [ ] **Step 5: Write failing formal evaluation tests**

Prove `evaluate_clevr_checkpoint` accepts only formal seeds/methods, verifies the seed common-state artifact and immutable run config, constructs a train dataset solely to recover sampler size, constructs base_test only after all preflight checks, loads the explicit update-8000 checkpoint, evaluates exactly once, publishes the canonical file, and refuses pilot/smoke, update<8000, existing output, confirmation, unknown CUDA, and mismatched identity/config.

- [ ] **Step 6: Implement `evaluate_clevr_checkpoint`**

Use the same model/optimizer/scheduler/sampler construction as training, restore with the reviewed loader, check returned metadata before base-test item access, and call Task 3 plus Task 2 publication. Do not call the statistics/reporting command automatically.

- [ ] **Step 7: Write failing CLI parser/dispatch tests**

Test required explicit paths, repeatable `--method`, repeatable `--resume METHOD=PATH`, phase/seed gates, no download/URL/auto-device options, `inspect` avoiding model/CUDA/item access, `train` and `evaluate` forwarding exact normalized values, JSON error output to stderr, and nonzero exit on validation/runtime failure. Importing the entry point must have no side effect.

- [ ] **Step 8: Implement `run_clevr_experiment.py`**

Use `argparse`, a `main(argv: Sequence[str] | None = None) -> int`, and the standard `if __name__ == "__main__"` guard. Help text must state that invoking train/evaluate starts user-authorized GPU work and that base-test is separate. Never spawn, daemonize, monitor, or retry a run.

- [ ] **Step 9: Verify focused integration and complete CPU regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python3.11 -m unittest -v tests.test_clevr_dataset tests.test_clevr_run_artifacts tests.test_clevr_runner tests.test_run_clevr_experiment
```

Then run the entire allowed CPU suite, `python -m compileall -q` over the changed Python files with bytecode redirected to a temporary directory or disabled, and `git diff --check <runner-base>...HEAD`. Expected: all tests pass; only documented timm warnings and the expected CUDA skip remain; no user data/GPU/training/evaluation occurs.

- [ ] **Step 10: Commit**

```bash
git add src/clevr_runner.py tests/test_clevr_runner.py run_clevr_experiment.py tests/test_run_clevr_experiment.py
git commit -m "feat: add explicit CLEVR experiment runner"
```

## Final handoff

After all four task reviews pass, run one independent whole-branch review over the runner design base through HEAD, allow one bounded final fix wave, and repeat the exact final verification. Prepare a PR body with problem, mathematical basis, mature-library/licensing decision, merged prerequisite heads, changes, test evidence, and limits. Push/create the real PR only if authorization succeeds; otherwise preserve the branch/worktree, record the exact blocker, and continue no further experiment execution. Do not create a tag before prerequisite and runner PRs merge and their verification is reproducible.
