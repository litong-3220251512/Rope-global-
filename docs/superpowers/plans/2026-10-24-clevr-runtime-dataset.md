# CLEVR Runtime Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the missing PUB-R1.1 runtime boundary that reads accepted local audit artifacts and returns only deterministic RGB tensors, encoded questions, true lengths, and binary supervision.

**Architecture:** Add one focused map-style `CLEVRQuestionDataset` backed by the accepted audit directory and explicit CLEVR root, plus one strict dynamic-padding collate function. Constructor validation consumes and then discards audit-only fields; lazy item access performs the frozen Pillow/torchvision RGB transform, while batches expose exactly four tensor keys.

**Tech Stack:** Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, Pillow 12.0.0, standard-library JSON/path handling, `unittest` temporary fixtures.

**Spec:** `docs/superpowers/specs/2026-10-24-clevr-runtime-dataset-design.md`

## Global Constraints

- Work only on `feat/clevr-runtime-dataset` in `/tmp/rope-clevr-runtime-dataset`; never modify `master` directly.
- Create only `src/clevr_dataset.py` and `tests/test_clevr_dataset.py` beyond the already committed spec and this plan.
- Accept only explicit `data_root`, explicit `audit_dir`, and one of `train`, `dev`, `base_test`, `confirmation`; never search, download, or fall back.
- Read fixed artifacts `selected_questions.jsonl`, `vocabulary.json`, and `audit_report.json`; require protocol version `PUB-R1.1` where serialized.
- Return exactly `image`, `question_tokens`, `question_length`, and `answer`; audit-only fields never enter samples or batches.
- Image transform is exactly PNG → Pillow RGB → `pil_to_tensor` → float32 divide by 255 → tensor resize `[224,224]`, bilinear, `antialias=True`; no normalization or augmentation.
- Preserve base order `(image_index, question_index, source_position)` within a partition; collate preserves supplied sample order and pads only with `<NULL>=0`.
- Do not use hashes, real CLEVR data, GPU, training, evaluation, model code, sampler code, or network access in this PR.
- Follow TDD: observe each focused failure before writing the minimum production behavior; run tests with `CUDA_VISIBLE_DEVICES=''`.
- Keep `tests.test_cifar_training` and `tests.test_cifar_data` out of aggregate commands because they execute policy-forbidden hash logic.

## Review Focus

- A symlinked image resolving outside `data_root` must fail construction; pin in Task 1 path tests.
- JSON booleans in integer fields must be rejected even though `bool` subclasses `int`; pin in Task 1 schema tests.
- A malformed row in an unrequested partition must still invalidate the audit directory; pin in Task 1 all-row tests.
- Duplicate identities or a partition count disagreement must fail rather than be deduplicated; pin in Task 1 integrity tests.
- An image made unreadable after construction must fail item access with context and must not be skipped; pin in Task 1 lazy-decode tests.

---

### Task 1: Validated Manifest-Backed Runtime Samples

**Files:**
- Create: `src/clevr_dataset.py`
- Create: `tests/test_clevr_dataset.py`

**Interfaces:**
- Consumes: `src.clevr_text.Vocabulary`, `src.clevr_text.encode_question`, `src.clevr_partitions.ImageIdentity`, and the fixed artifacts emitted by `run_clevr_data_audit.py`.
- Produces: `CLEVRDatasetError(ValueError)` and `CLEVRQuestionDataset(Dataset[dict[str, Tensor]])` with keyword-only `data_root: Path`, `audit_dir: Path`, `partition: str`, plus `__len__()` and exact four-key `__getitem__()`.

- [ ] **Step 1: Write fixture helpers and failing happy-path tests**

In `tests/test_clevr_dataset.py`, add `RuntimeDatasetFixture` that writes all four valid partitions, accepted vocabulary/report schemas, and constant RGBA PNGs under a temporary explicit root. Put JSONL rows out of base order. Add:

```python
class RuntimeDatasetTests(unittest.TestCase):
    def test_sorts_selected_partition_and_returns_only_model_tensors(self): ...
    def test_rgb_transform_matches_the_frozen_torchvision_reference(self): ...
```

Use distinguishable constant image colors to prove ordering by `(image_index, question_index, source_position)` without returning IDs. Assert the exact four keys, image float32 `[3,224,224]` within the epsilon-derived numerical tolerance of `[0,1]` defined by the design, int64 question/scalars, exact values, and direct equality to the explicit functional transform. A constant `(255,128,0,alpha)` fixture remains `(1.0,128/255,0.0)`.

- [ ] **Step 2: Run happy-path tests and observe RED**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_sorts_selected_partition_and_returns_only_model_tensors \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_rgb_transform_matches_the_frozen_torchvision_reference -v
```

Expected: FAIL because `src.clevr_dataset` does not exist.

- [ ] **Step 3: Implement the minimum valid dataset and image transform**

Define in `src/clevr_dataset.py`:

```python
class CLEVRDatasetError(ValueError): ...

class CLEVRQuestionDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, *, data_root: Path, audit_dir: Path, partition: str) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Tensor]: ...
```

Use private frozen runtime records. Parse all fixed artifacts, reconstruct `Vocabulary`, validate every JSONL row, retain the requested partition, sort by the exact tuple, and lazily decode through Pillow plus explicit torchvision functionals. Do not expose configurable transforms or records.

- [ ] **Step 4: Run Step 2 and verify GREEN**

Expected: both tests PASS.

- [ ] **Step 5: Write failing strict artifact/schema/integrity tests**

Add these table-driven tests:

```python
def test_requires_explicit_roots_partition_and_all_three_artifacts(self): ...
def test_rejects_protocol_vocabulary_and_report_count_disagreement(self): ...
def test_rejects_missing_extra_or_wrong_typed_manifest_fields(self): ...
def test_rejects_invalid_split_identity_tokens_answer_and_relation(self): ...
def test_rejects_duplicate_question_or_source_identity(self): ...
def test_validates_malformed_rows_even_outside_the_requested_partition(self): ...
```

Cover missing/non-directory roots, unknown/empty partition, missing files, malformed JSON/JSONL and blank lines, wrong protocol, invalid vocabulary inverse/special IDs, missing/duplicate/unknown summaries, count mismatch, missing/extra keys, boolean integer fields, split mismatch, noncanonical filename, negative IDs, invalid question/relation/answer, non-list/out-of-range/NULL-containing tokens, wrong START/END, token recomputation mismatch, and both duplicate identity keys. Require contextual `CLEVRDatasetError`; never skip or repair.

- [ ] **Step 6: Run the six Step 5 tests and observe focused failures**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_requires_explicit_roots_partition_and_all_three_artifacts \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_rejects_protocol_vocabulary_and_report_count_disagreement \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_rejects_missing_extra_or_wrong_typed_manifest_fields \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_rejects_invalid_split_identity_tokens_answer_and_relation \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_rejects_duplicate_question_or_source_identity \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_validates_malformed_rows_even_outside_the_requested_partition -v
```

Expected: FAIL only at newly asserted unimplemented validation paths.

- [ ] **Step 7: Implement complete constructor validation**

Add small private readers/validators. Use strict `type(value) is int`, exact key sets, existing `Vocabulary`/`ImageIdentity`, `encode_question` equality, one summary per required partition, and direct count comparison. Wrap JSON/path errors contextually; do not coerce or mutate artifacts.

- [ ] **Step 8: Run `RuntimeDatasetTests` and verify current tests GREEN**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset.RuntimeDatasetTests -v
```

- [ ] **Step 9: Write failing path-containment and lazy-decode tests**

Add:

```python
def test_rejects_absolute_traversing_noncanonical_missing_and_symlink_escape_paths(self): ...
def test_decode_failure_after_construction_is_contextual_and_not_skipped(self): ...
def test_multiple_questions_may_share_one_image_without_mutable_sample_storage(self): ...
```

Assert exact relative path, resolved containment, regular-file presence, and symlink escape rejection. For lazy decode, construct with a valid file, replace it with invalid PNG bytes, and require item access to name index/path. For shared images, mutate one returned tensor and prove later access has independent storage.

- [ ] **Step 10: Run the three Step 9 tests and observe focused failures**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_rejects_absolute_traversing_noncanonical_missing_and_symlink_escape_paths \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_decode_failure_after_construction_is_contextual_and_not_skipped \
  tests.test_clevr_dataset.RuntimeDatasetTests.test_multiple_questions_may_share_one_image_without_mutable_sample_storage -v
```

Expected: FAIL only at missing containment/decode/independence behavior.

- [ ] **Step 11: Implement containment and contextual lazy decode**

Resolve the explicit root once. Require the serialized POSIX path to equal the canonical path before resolution; require `resolve(strict=True).is_relative_to(data_root)` and `is_file()`. Open only PNG in item access with Pillow's explicit `formats=("PNG",)` restriction, convert RGB inside the context, create independent tensors, validate final shape/dtype/finiteness/range, and wrap decode/I/O failures.

- [ ] **Step 12: Run complete `RuntimeDatasetTests` and verify GREEN**

Use the Step 8 command. Expected: all Task 1 tests PASS with no GPU use.

- [ ] **Step 13: Commit Task 1**

```bash
git add src/clevr_dataset.py tests/test_clevr_dataset.py
git commit -m "feat: add validated CLEVR runtime dataset"
```

### Task 2: Strict Dynamic Padding and DataLoader Integration

**Files:**
- Modify: `src/clevr_dataset.py`
- Modify: `tests/test_clevr_dataset.py`

**Interfaces:**
- Consumes: Task 1's exact four-key sample mapping.
- Produces: `collate_clevr_questions(samples: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]` with the same keys and shapes `[N,3,224,224]`, `[N,L_max]`, `[N]`, `[N]`.

- [ ] **Step 1: Write failing padding and DataLoader tests**

Add:

```python
class CLEVRQuestionCollateTests(unittest.TestCase):
    def test_pads_with_null_without_truncation_and_preserves_sample_order(self): ...
    def test_dataloader_batch_contains_exactly_the_four_model_fields(self): ...
```

Use `[1,7,2]` and `[1,5,6,2]`; assert exact `[[1,7,2,0],[1,5,6,2]]`, lengths `[3,4]`, unchanged answer/image order, exact keys, and no offline field. Exercise `DataLoader(..., batch_size=2, shuffle=False, collate_fn=collate_clevr_questions)` on fixtures.

- [ ] **Step 2: Run both tests and observe RED**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset.CLEVRQuestionCollateTests.test_pads_with_null_without_truncation_and_preserves_sample_order \
  tests.test_clevr_dataset.CLEVRQuestionCollateTests.test_dataloader_batch_contains_exactly_the_four_model_fields -v
```

Expected: FAIL because `collate_clevr_questions` does not exist.

- [ ] **Step 3: Implement minimum dynamic padding**

Define the exact signature from Interfaces. Require non-empty input, preserve order, stack images/answers/lengths, and use standard PyTorch padding with `batch_first=True`, `padding_value=0`. Return only four keys.

- [ ] **Step 4: Run Step 2 and verify GREEN**

- [ ] **Step 5: Write failing malformed-sample tests**

Add:

```python
def test_rejects_empty_batches_and_missing_or_extra_keys(self): ...
def test_rejects_wrong_sample_dtypes_shapes_lengths_and_answers(self): ...
def test_rejects_tokens_without_exact_start_end_or_with_null_before_padding(self): ...
```

Cover non-tensors; wrong keys; image dtype/shape/finiteness/range; token rank/dtype/emptiness; scalar rank/dtype; length mismatch; START/END mismatch; NULL before padding; and answer outside `{0,1}`. Require explicit `CLEVRDatasetError`, not downstream `stack` errors.

- [ ] **Step 6: Run Step 5 tests and observe focused failures**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset.CLEVRQuestionCollateTests.test_rejects_empty_batches_and_missing_or_extra_keys \
  tests.test_clevr_dataset.CLEVRQuestionCollateTests.test_rejects_wrong_sample_dtypes_shapes_lengths_and_answers \
  tests.test_clevr_dataset.CLEVRQuestionCollateTests.test_rejects_tokens_without_exact_start_end_or_with_null_before_padding -v
```

Expected: FAIL only at the missing explicit validation paths.

- [ ] **Step 7: Implement strict collate validation**

Validate each sample against Task 1's public contract in a small private helper. Do not widen the public API or accept alternate names/types.

- [ ] **Step 8: Run focused dataset suite**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset -v
```

Expected: all focused tests PASS.

- [ ] **Step 9: Run allowed whole-project CPU-only suite**

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_dataset tests.test_clevr_audit tests.test_clevr_text \
  tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_schema \
  tests.test_local_rope tests.test_classification \
  tests.test_attention_compatibility tests.test_segmentation -v
```

Expected: all tests PASS; CUDA-only test skipped; only existing `timm` warnings acceptable.

- [ ] **Step 10: Verify scope and direct evidence**

```bash
git diff --check c60a8c9...HEAD
git diff --name-status c60a8c9...HEAD
git status --short --branch
```

Expected: no whitespace errors; implementation diff only `src/clevr_dataset.py` and `tests/test_clevr_dataset.py`; no real data/network/GPU/training command occurred.

- [ ] **Step 11: Commit Task 2**

```bash
git add src/clevr_dataset.py tests/test_clevr_dataset.py
git commit -m "feat: collate CLEVR question batches"
```

## Final Branch Review and PR Handoff

- [ ] Re-read spec and plan requirement by requirement and map each to a focused test or direct evidence.
- [ ] Rerun the complete Task 2 Step 9 suite after the final commit and record pass/skip/warning counts.
- [ ] Run `git show --check --stat HEAD`, `git diff --check master...HEAD`, and verify a clean worktree.
- [ ] Request one fresh read-only whole-branch review focused on model-boundary leakage, path escape, image numerics, strict schema, and collate invariants. Fix only confirmed Critical/Important findings with focused RED→GREEN evidence, then rerun verification.
- [ ] Push `feat/clevr-runtime-dataset` and create a real PR against `master` with problem, contract, mature-library/license choice, changes, tests, and limitations. If GitHub authentication blocks submission, record the exact blocker and preserve the verified branch/worktree for the user.
- [ ] Do not merge, tag, run real CLEVR, or begin the common VQA backbone until this unit is reviewable and the governing one-PR-at-a-time sequence permits the next source change.
