# CLEVR Q-only Runtime Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict PUB-R1.1 question-only dataset and collator that preserve audited question order and labels while requiring and reading no RGB or scene data.

**Architecture:** Keep the public RGB dataset unchanged and add a dedicated `CLEVRQuestionOnlyDataset` with no `data_root` argument. Parameterize the existing private manifest parser so canonical image metadata is always validated but filesystem resolution occurs only for the RGB class; share tensor validation between separate exact-field collators.

**Tech Stack:** Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, Pillow 12.0.0, standard-library JSON/path handling, `unittest` temporary fixtures.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-question-only-data-design.md`

## Global Constraints

- Work only on `feat/clevr-qonly-data` in `/tmp/rope-clevr-qonly-data`, based on `feat/clevr-runtime-dataset` SHA `2daed5d7fc81d28f806120b9d5fd67aeab8f4389`; never modify `master`.
- Modify only `src/clevr_dataset.py` and `tests/test_clevr_dataset.py` beyond the committed design and this plan.
- The Q-only constructor accepts only keyword-only `audit_dir: Path` and `partition: str`; it never accepts, searches for, or infers `data_root`.
- Validate every manifest row and all four partition counts, including canonical image identity/path metadata, but never resolve, stat, open, or decode image paths on the Q-only path.
- Q-only samples and batches expose exactly `question_tokens`, `question_length`, and `answer`; no image or audit-only field may cross the boundary.
- Preserve RGB constructor, image validation, decode/resize, four-field samples, and collator behavior exactly.
- Preserve base order `(image_index, question_index, source_position)` and right-pad only with `<NULL>=0`, without truncation or reordering.
- Add no dependency, hash, generated data, real CLEVR access, scene/program read, network, GPU, training, smoke, or evaluation command.
- Follow TDD: observe focused failures before product code, then run the full allowed CPU suite with `CUDA_VISIBLE_DEVICES=''`.
- Keep `tests.test_cifar_training` and `tests.test_cifar_data` out of aggregate commands because they contain project-policy-forbidden hash logic.
- The spawn-worker DataLoader regression needs execution outside the filesystem sandbox because its resource-sharer Unix socket is blocked there; this is an environment constraint, not a code failure.

## Review Focus

- A noncanonical `image_path` in an unrequested partition must still invalidate Q-only construction; Task 1 adds an all-row corrupt-metadata test.
- Complete absence of the CLEVR image tree must not affect Q-only construction or access, and `PIL.Image.open` must remain uncalled; Task 1 adds a no-root/no-open test.
- RGB and Q-only ordering, tokens, lengths, and answers must not drift after parser refactoring; Task 1 adds an index-by-index projection test.
- An image or audit-only key in a Q-only sample must be rejected rather than ignored; Task 2 adds exact-key failure cases.
- Empty input, malformed token boundaries/dtypes, inconsistent lengths, or invalid answers must fail before padding; Task 2 adds bounded malformed-sample cases.

---

### Task 1: Image-independent question-only samples

**Files:**
- Modify: `tests/test_clevr_dataset.py`
- Modify: `src/clevr_dataset.py`

**Interfaces:**
- Consumes: existing `RuntimeDatasetFixture`, `Vocabulary`, `encode_question`, `ImageIdentity`, `_parse_row`, `_load_records`, and accepted audit artifacts.
- Produces: `CLEVRQuestionOnlyDataset(Dataset[dict[str, Tensor]])` with keyword-only `audit_dir: Path`, `partition: str`, `__len__()`, and exact three-key `__getitem__()`.
- Preserves: `CLEVRQuestionDataset(data_root: Path, audit_dir: Path, partition: str)` and all existing RGB outputs/errors.

- [ ] **Step 1: Add focused failing Q-only dataset tests**

Import `shutil`, `unittest.mock`, and `CLEVRQuestionOnlyDataset`. Add
`CLEVRQuestionOnlyDatasetTests` with these tests:

```python
def test_needs_no_data_root_or_images_and_never_opens_png(self): ...
def test_matches_rgb_question_projection_and_order(self): ...
def test_rejects_noncanonical_image_metadata_without_resolving_files(self): ...
def test_rejects_invalid_constructor_inputs_and_unrequested_bad_rows(self): ...
```

The no-image test removes the temporary `data_root`, patches
`src.clevr_dataset.Image.open` to raise if called, constructs from only
`audit_dir` and `partition="train"`, and asserts item 0 is exactly tokens
`[1,4,2]`, scalar length `3`, scalar answer `0`, all `int64`, with exactly the
three public keys. Assert the mock has zero calls.

The projection test constructs both classes while PNGs exist and uses
`torch.equal` at every index for `question_tokens`, `question_length`, and
`answer`. The corruption tests rewrite a row's `image_path` to a noncanonical
string and separately corrupt an unrequested partition; both must raise
`CLEVRDatasetError` even after images are absent. Also pin unknown partition and
missing audit directory errors.

- [ ] **Step 2: Run the new class and observe RED**

```bash
CUDA_VISIBLE_DEVICES='' /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  test_clevr_dataset.CLEVRQuestionOnlyDatasetTests -v
```

Run from `tests/` with `PYTHONPATH=/tmp/rope-clevr-qonly-data` if module
discovery requires it. Expected: import error because
`CLEVRQuestionOnlyDataset` does not exist.

- [ ] **Step 3: Parameterize private row loading without weakening RGB**

In `src/clevr_dataset.py`:

- change `_RuntimeRecord.image_path` to `Path | None`;
- change `_parse_row(..., data_root: Path | None, ...)` so it always checks
  `image_filename`, `ImageIdentity`, and exact canonical relative
  `image_path`, calls `_resolve_image_file` only when `data_root is not None`,
  and otherwise stores `None`;
- change `_load_records(..., data_root: Path | None, ...)` consistently;
- leave the RGB caller passing its resolved root and add an explicit internal
  invariant check before RGB access so an impossible `None` fails clearly
  rather than reaching `Path.resolve`.

Do not add an image-ignore mode to the public RGB class and do not skip
validation of unselected rows.

- [ ] **Step 4: Implement the dedicated Q-only dataset**

Add:

```python
class CLEVRQuestionOnlyDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, *, audit_dir: Path, partition: str) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Tensor]: ...
```

Resolve and validate only `audit_dir`; validate `partition`; load vocabulary,
report counts, and records through `_load_records(data_root=None, ...)`.
`__getitem__` creates only independent `int64` tensors for tokens, scalar true
length, and scalar answer. It must not reference Pillow, torchvision, or a
filesystem image path.

- [ ] **Step 5: Run focused dataset tests and observe GREEN**

```bash
cd /tmp/rope-clevr-qonly-data/tests
PYTHONPATH=/tmp/rope-clevr-qonly-data CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  test_clevr_dataset.CLEVRQuestionOnlyDatasetTests \
  test_clevr_dataset.RuntimeDatasetTests -v
```

Expected: all selected Q-only and existing RGB dataset tests pass. The existing
spawn-worker test may require the already approved outside-sandbox execution;
if it fails only with resource-sharer `PermissionError`, rerun the exact command
with controlled escalation rather than changing code.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/clevr_dataset.py tests/test_clevr_dataset.py
git commit -m "feat: add image-independent CLEVR question dataset"
```

---

### Task 2: Strict Q-only collation

**Files:**
- Modify: `tests/test_clevr_dataset.py`
- Modify: `src/clevr_dataset.py`

**Interfaces:**
- Consumes: exact three-key samples from `CLEVRQuestionOnlyDataset` and existing `pad_sequence`.
- Produces: `collate_clevr_question_only(samples: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]` with shapes `[N,L_max]`, `[N]`, `[N]`.
- Preserves: `collate_clevr_questions` exact four-key validation and output.

- [ ] **Step 1: Add focused failing collate tests**

Import `collate_clevr_question_only` and add
`CLEVRQuestionOnlyCollateTests` with a three-key sample helper and:

```python
def test_pads_without_truncation_and_preserves_order(self): ...
def test_dataloader_batch_contains_exactly_three_fields_without_images(self): ...
def test_rejects_empty_missing_extra_and_non_tensor_samples(self): ...
def test_rejects_invalid_token_length_and_answer_contracts(self): ...
```

For inputs `([1,7,2],1)` and `([1,5,6,2],0)`, assert exact output tokens
`[[1,7,2,0],[1,5,6,2]]`, lengths `[3,4]`, answers `[1,0]`, exact keys, and
`int64`. The DataLoader test uses the Q-only dataset after removing images and
asserts batch answers `[0,1]` and no `image` key. Failure cases cover an empty
sequence; missing/extra/image keys; a non-tensor value; wrong token dtype/rank;
missing START/END; embedded NULL or negative ID; non-scalar/wrong length; and
non-scalar or non-binary answer.

- [ ] **Step 2: Run the new collate class and observe RED**

```bash
cd /tmp/rope-clevr-qonly-data/tests
PYTHONPATH=/tmp/rope-clevr-qonly-data CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  test_clevr_dataset.CLEVRQuestionOnlyCollateTests -v
```

Expected: import error because `collate_clevr_question_only` does not exist.

- [ ] **Step 3: Share question tensor validation and implement Q-only collate**

In `src/clevr_dataset.py` add `_QUESTION_ONLY_MODEL_FIELDS` and:

```python
def collate_clevr_question_only(
    samples: Sequence[Mapping[str, Tensor]],
) -> dict[str, Tensor]: ...

def _validate_question_only_collate_sample(
    sample: Mapping[str, Tensor], *, index: int
) -> tuple[Tensor, Tensor, Tensor]: ...

def _validate_question_tensors(
    tokens: Tensor, length: Tensor, answer: Tensor, *, context: str
) -> tuple[Tensor, Tensor, Tensor]: ...
```

Both collators enforce their own exact key set and all-tensor requirement.
Move the existing token/length/answer checks unchanged into
`_validate_question_tensors`; the RGB validator first validates its image and
then delegates. Q-only delegates without constructing or accepting an image.
Use `pad_sequence(..., batch_first=True, padding_value=0)` and preserve supplied
sample order.

- [ ] **Step 4: Run focused collate and complete dataset tests**

```bash
cd /tmp/rope-clevr-qonly-data/tests
PYTHONPATH=/tmp/rope-clevr-qonly-data CUDA_VISIBLE_DEVICES='' \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest \
  test_clevr_dataset.CLEVRQuestionOnlyCollateTests \
  test_clevr_dataset.CLEVRQuestionOnlyDatasetTests \
  test_clevr_dataset.CLEVRQuestionCollateTests \
  test_clevr_dataset.RuntimeDatasetTests -v
```

Expected: all focused Q-only and RGB tests pass, subject only to the documented
spawn-worker sandbox rerun rule.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/clevr_dataset.py tests/test_clevr_dataset.py
git commit -m "feat: collate CLEVR question-only batches"
```

---

### Task 3: Regression, scope audit, and PR evidence

**Files:**
- Verify: `docs/superpowers/specs/2026-10-25-clevr-question-only-data-design.md`
- Verify: `docs/superpowers/plans/2026-10-25-clevr-question-only-data.md`
- Verify: `src/clevr_dataset.py`
- Verify: `tests/test_clevr_dataset.py`

**Interfaces:**
- Consumes: Tasks 1-2 committed implementation.
- Produces: reproducible CPU evidence and a reviewable branch diff; no additional feature.

- [ ] **Step 1: Run the complete allowed CPU suite**

Run the repository's established aggregate `unittest` command excluding only
`tests.test_cifar_training` and `tests.test_cifar_data`, with
`CUDA_VISIBLE_DEVICES=''`. Use the explicit flightgpt interpreter. Because the
suite includes the spawn-worker test, run the exact command with controlled
outside-sandbox permission if the sandbox reproduces the known resource-sharer
socket denial.

Expected: all allowed CPU tests pass; the existing CUDA-only test is skipped;
only previously understood dependency warnings remain.

- [ ] **Step 2: Audit scope and direct evidence**

```bash
git diff --check feat/clevr-runtime-dataset...HEAD
git diff --stat feat/clevr-runtime-dataset...HEAD
git status --short --branch
```

Inspect the source diff and assert directly that Q-only has no `data_root`,
Pillow/torchvision call, scene/program read, image output, dependency, hash,
training, or runner code; RGB API and output remain intact. Expected diff:
exactly the two docs, `src/clevr_dataset.py`, and
`tests/test_clevr_dataset.py`, with a clean worktree.

- [ ] **Step 3: Obtain independent whole-branch review**

Give a fresh reviewer the exact base/head SHAs, spec, plan, changed files, and
test evidence. Require severity-ranked findings and independent checks of
no-image I/O, all-row metadata validation, RGB preservation, projection
equality, and strict collate boundaries. Fix every Critical/Important finding
with a focused regression before proceeding; record any consciously deferred
minor limitation.

- [ ] **Step 4: Push and create the real stacked PR**

Push `feat/clevr-qonly-data`, then create a non-draft PR with base
`feat/clevr-runtime-dataset`. The description must include the trigger and
boundary, mathematical/data contract, mature-library and license review,
changes, RED/GREEN and full-suite evidence, independent review result, and
limitations: no runner integration, real data, GPU, training, evaluation, or
performance claim.

Verify the remote head SHA and PR state directly. Do not claim CI unless a real
check exists, and do not merge the branch.
