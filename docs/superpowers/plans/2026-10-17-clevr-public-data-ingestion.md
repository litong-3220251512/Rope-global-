# CLEVR Public-Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only, reproducible pipeline that selects the preregistered single-relation CLEVR questions, audits their official answers, creates image-level partitions, encodes question text, and emits reviewable manifests without exposing privileged scene/program fields to the future model.

**Architecture:** Keep this subsystem independent of PyTorch. A pure program module validates, selects, and evaluates official CLEVR programs; separate partition and text modules own deterministic image splitting and vocabulary construction; an audit module joins the user-provided files and writes manifests and counts. Deliver the work as four serial PRs, stopping after each real PR is reviewable.

**Tech Stack:** Python standard library, NumPy `Generator(PCG64)`, Pillow; `unittest` for data-free fixtures. PyTorch, torchvision, timm, GPU code, downloads, and pretrained weights are out of scope.

**Spec:** `docs/rope_public_data_experiment_plan.md`

## Global Constraints

- Read only the user-provided `data/CLEVR_v1.0`; never download or generate images, questions, labels, or weights.
- Never use SHA, MD5, archive digests, or custom hashes. Use IDs, filenames, sizes, counts, parsed fields, direct comparisons, and tests.
- Do not invoke the legacy CIFAR digest helpers or the two legacy test modules that execute them (`tests.test_cifar_training`, `tests.test_cifar_data`). Run the targeted tests plus the explicitly listed non-hash regression suite and disclose these policy exclusions.
- Select only full linear programs of the form `scene -> filters* -> unique -> relate(r) -> filters* -> exist`, with `r` in `left/right/front/behind` and official answer in `yes/no`.
- Split complete image mother sets before filtering questions. Use `PCG64(20261016)` for official train and `PCG64(20261017)` for official val.
- Build the vocabulary from selected training questions only. Preserve the exact tokenization contract in spec section 4.2.
- Programs and scenes are offline audit inputs only. Offline manifests may expose RGB path, encoded original question, label, relation slice, and traceability IDs; they are not model inputs. The later dataset builder must strip paths, relation slices, and IDs so model batches contain only RGB, question encoding, length, and the supervised answer.
- Malformed schema, answer disagreement, duplicate/conflicting IDs, train class/relation gaps, or an empty required evaluation partition are hard failures. Do not repair labels or relax selection.
- Every source PR starts from current `master` on a separate branch, uses red-green-refactor, records library/version/license decisions, creates a real PR, and stops before the next PR. PR B1 and PR B2 are independent once PR A is merged; dependency-consuming PR C starts only after PR A, PR B1, and PR B2 are all merged into `master`.

## Review Focus

- A structurally malformed program node must raise a schema error rather than disappear into a normal exclusion count; Task 1 tests malformed `inputs`, `value_inputs`, and node mappings.
- `unique` producing zero or multiple objects must fail official-answer auditing; Task 2 tests both cardinality failures.
- Duplicate image IDs, duplicate filenames, or an ID/filename disagreement across scenes and questions must stop manifest construction; Task 6 tests each collision.
- Consecutive spaces and unseen dev/test words must follow the literal-space and `<UNK>` contract; Task 5 tests both without normalizing the source text.
- An empty required partition is a hard failure, while an empty evaluation relation-answer cell is retained and reported as `NA`; Task 7 tests both branches.

---

## PR A — Program Selection and Official-Answer Audit

Branch: `feat/clevr-program-audit`

### Task 1: Validate and Select Target Programs

**Files:**
- Create: `src/clevr_programs.py`
- Create: `tests/test_clevr_programs.py`
- Reference: `docs/rope_public_data_experiment_plan.md:40`

**Interfaces:**
- Produces: `ProgramSchemaError(ValueError)`.
- Produces: immutable `ProgramNode(function: str, inputs: tuple[int, ...], value_inputs: tuple[str, ...])`.
- Produces: immutable `SelectionDecision(relation: str | None, exclusion_reason: str | None)` with exactly one populated field.
- Produces: `parse_program(raw: object) -> tuple[ProgramNode, ...]`.
- Produces: `select_single_relation_exist(nodes: tuple[ProgramNode, ...], answer: object) -> SelectionDecision`.

- [ ] **Step 1: Write failing schema tests**

  Add table-driven tests with literal fixtures for a valid chain and malformed node/list/input/value types. Name the production mutation each test catches: accepting a branch, accepting a missing field, or treating malformed data as ordinary exclusion.

- [ ] **Step 2: Verify the schema tests fail for the missing API**

  Run: `conda run -n flightgpt python -m unittest tests.test_clevr_programs -v`

  Expected: import failure for `src.clevr_programs`, not an environment/dependency failure.

- [ ] **Step 3: Implement `ProgramNode`, `ProgramSchemaError`, and `parse_program`**

  Require every node to contain typed `function`, `inputs`, and `value_inputs`. Preserve source order. Basic official node schema errors raise; valid non-target functions remain parseable for normal exclusion.

- [ ] **Step 4: Verify schema tests pass**

  Run the Task 1 unittest command. Expected: all schema tests pass.

- [ ] **Step 5: Write failing selection tests**

  Cover zero/multiple filters on either side, exact adjacency of `unique -> relate`, one predecessor per node, one `unique`, one `relate`, terminal `exist`, four allowed relations, filter value domains, and `yes/no`. The literal domains are colors `gray/red/blue/green/brown/purple/cyan/yellow`, shapes `cube/sphere/cylinder`, sizes `small/large`, and materials `rubber/metal`. Include valid official programs that are not this task and assert stable exclusion categories rather than exceptions.

- [ ] **Step 6: Verify selection tests fail for missing behavior**

  Run the Task 1 unittest command. Expected: failures identify unimplemented target selection.

- [ ] **Step 7: Implement `SelectionDecision` and `select_single_relation_exist`**

  Validate exact accepted-chain arguments from spec section 2.2. Return a relation only for accepted programs; return one documented exclusion code for well-formed non-target programs; raise for malformed target arguments.

- [ ] **Step 8: Run targeted tests and commit**

  Run: `conda run -n flightgpt python -m unittest tests.test_clevr_programs -v`

  Expected: PASS.

  Commit: `feat: select CLEVR relation-existence programs`

### Task 2: Evaluate Selected Programs Against Scene Graphs

**Files:**
- Modify: `src/clevr_programs.py`
- Modify: `tests/test_clevr_programs.py`

**Interfaces:**
- Consumes: parsed accepted `ProgramNode` tuples from Task 1.
- Produces: `SceneSchemaError(ValueError)` and `AnswerMismatchError(ValueError)`.
- Produces: `evaluate_selected_program(nodes: tuple[ProgramNode, ...], scene: object) -> bool`.
- Produces: `audit_official_answer(nodes: tuple[ProgramNode, ...], scene: object, answer: object) -> None`.

- [ ] **Step 1: Write failing evaluator tests from hand-checked scenes**

  Use small literal object lists and relationship adjacency lists. Exercise each filter attribute, each relation, empty/non-empty `exist`, and both `yes/no` answers. Expected values must be literal, not computed with production helpers.

- [ ] **Step 2: Verify evaluator tests fail for the missing API**

  Run the Task 1 unittest command. Expected: missing evaluator/audit behavior.

- [ ] **Step 3: Implement the minimal selected-chain evaluator**

  Implement only `scene`, the four allowed filters, `unique`, the four allowed relations, and `exist`. `unique` requires exactly one object. Validate relationship rows against object count. Do not implement general CLEVR execution.

- [ ] **Step 4: Verify evaluator tests pass**

  Run the Task 1 unittest command. Expected: PASS.

- [ ] **Step 5: Write failing integrity tests**

  Cover `unique` cardinality zero/multiple, out-of-range relationship indices, wrong relationship row count, invalid object attributes, and official-answer disagreement.

- [ ] **Step 6: Implement hard-failure paths and verify green**

  Run the Task 1 unittest command. Expected: PASS with every malformed/mismatched case rejected explicitly.

- [ ] **Step 7: Run branch verification and commit**

  Run targeted tests, then the non-hash regression suite:

  `conda run -n flightgpt python -m unittest tests.test_clevr_schema tests.test_local_rope tests.test_classification tests.test_attention_compatibility tests.test_segmentation -v`

  Expected: targeted and non-hash regression tests pass. Record that `tests.test_cifar_training` and `tests.test_cifar_data` are intentionally excluded because they execute legacy SHA/MD5 logic prohibited by the repository policy.

  Commit: `feat: audit CLEVR answers against scene graphs`

### Task 3: Deliver PR A

**Files:**
- Modify only if evidence requires it: `docs/rope_public_data_experiment_plan.md`

**Interfaces:**
- Produces: one reviewable PR implementing only program selection and answer audit.

- [ ] **Step 1: Record mature-library decision in the PR description**

  Record official CLEVR schema/engine as the semantic authority (archived BSD generator code), Python standard library as runtime, and why no general loader/executor is copied or downloaded. Include actual Python version and licenses.

- [ ] **Step 2: Verify diff and repository state**

  Run `git diff --check`, targeted tests, the non-hash regression suite from Task 2, and `git status --short --branch`. Confirm no data, weights, generated manifests, or unrelated edits are tracked.

- [ ] **Step 3: Push and create the real PR**

  PR description must include trigger/problem, mathematical/data contract, library review, changes, red-green evidence, full-suite status, and limitations. Stop for review before PR B1.

---

## PR B1 — Deterministic Image Partitions

Branch: `feat/clevr-image-partitions`

### Task 4: Partition Complete Image Mother Sets

**Files:**
- Create: `src/clevr_partitions.py`
- Create: `tests/test_clevr_partitions.py`

**Interfaces:**
- Produces: immutable `ImageIdentity(official_split: str, image_index: int, image_filename: str)`.
- Produces: immutable `ImagePartitions(train, dev, base_test, confirmation)` as tuples of `ImageIdentity`.
- Produces: `build_image_partitions(train_images: Sequence[ImageIdentity], val_images: Sequence[ImageIdentity]) -> ImagePartitions` using fixed seeds and floor boundaries from spec section 2.3.

- [ ] **Step 1: Write failing literal partition tests**

  Use small unsorted mother sets and independently recorded PCG64 expected permutations. Test sort-before-permute, floor boundaries, complete/disjoint coverage, and deterministic repeatability.

- [ ] **Step 2: Verify red, implement minimal partitioning, verify green**

  Run: `conda run -n flightgpt python -m unittest tests.test_clevr_partitions -v`

  Reject duplicate indices, duplicate filenames, invalid split names, and index/filename prefix disagreement.

- [ ] **Step 3: Run branch verification, commit, and deliver PR B1**

  Run the targeted module and the non-hash regression suite from Task 2; inspect diff/status; document the installed NumPy version, the `Generator(PCG64).permutation` API, and NumPy's BSD license.

  Commit: `feat: partition CLEVR images deterministically`

  Push and create one real PR. Stop once PR B1 is reviewable before starting PR B2.

---

## PR B2 — Original-Question Text Encoding

Branch: `feat/clevr-text-encoding`

### Task 5: Encode Original Question Text

**Files:**
- Create: `src/clevr_text.py`
- Create: `tests/test_clevr_text.py`

**Interfaces:**
- Produces: `tokenize_question(question: str) -> tuple[str, ...]` including `<START>` and `<END>`.
- Produces: immutable `Vocabulary(token_to_id: Mapping[str, int], id_to_token: tuple[str, ...])`; the mapping must reject mutation.
- Produces: `build_vocabulary(training_questions: Sequence[str]) -> Vocabulary`.
- Produces: `encode_question(question: str, vocabulary: Vocabulary) -> tuple[int, ...]`.

- [ ] **Step 1: Write failing tokenization tests**

  Assert literal outputs for case preservation, comma/semicolon separation, removal of question marks/periods, consecutive spaces yielding empty tokens, and START/END placement. The exact operation order is separator insertion, punctuation removal, literal `split(" ")`, then START/END insertion.

- [ ] **Step 2: Verify red, implement tokenizer, verify green**

  Run: `conda run -n flightgpt python -m unittest tests.test_clevr_text -v`

- [ ] **Step 3: Write failing vocabulary/encoding tests**

  Assert fixed special IDs `0..3`, lexicographic ordinary-token IDs, train-only vocabulary behavior, and `<UNK>` mapping for unseen dev/test tokens.

- [ ] **Step 4: Verify red, implement vocabulary/encoding, verify green**

  Run the Task 5 unittest command. Expected: PASS.

- [ ] **Step 5: Run branch verification, commit, and deliver PR B2**

  Run the targeted module and the non-hash regression suite from Task 2; inspect diff/status; document why no CLEVR-IEP source is copied under its CC BY-NC 4.0 license.

  Commit: `feat: encode CLEVR question text`

  Push and create one real PR. Stop for review before PR C.

---

## PR C — Local Data Audit and Manifest Builder

Branch: `feat/clevr-data-audit`

### Task 6: Load and Join Official Files Safely

**Files:**
- Create: `src/clevr_audit.py`
- Create: `tests/test_clevr_audit.py`

**Interfaces:**
- Consumes: APIs from merged PR A, PR B1, and PR B2.
- Produces: `CLEVRDataError(ValueError)`.
- Produces: immutable source records for scenes/questions with traceability IDs.
- Produces: `load_source_index(data_root: Path) -> SourceIndex`, reading only train/val official files and paths.

- [ ] **Step 1: Write failing temporary-directory integration tests**

  Build tiny real JSON/PNG fixtures in a temporary directory. Test valid joins plus missing files, duplicate/conflicting IDs, split mismatch, question-image mismatch, and unreadable PNGs.

- [ ] **Step 2: Verify red, implement the minimal loader/index, verify green**

  Run: `conda run -n flightgpt python -m unittest tests.test_clevr_audit -v`

  The loader accepts an explicit root only; it has no URL, download flag, fallback search, or weight path.

- [ ] **Step 3: Commit**

  Commit: `feat: validate local CLEVR source files`

### Task 7: Build Manifests and Enforce Coverage Gates

**Files:**
- Modify: `src/clevr_audit.py`
- Modify: `tests/test_clevr_audit.py`

**Interfaces:**
- Produces: immutable offline-manifest `SelectedQuestion` with partition, official split, image/question IDs, source array position, original question, encoded tokens, `no=0/yes=1`, and relation. This record is not a dataset item or model batch.
- Produces: immutable `AuditResult` containing selected records, vocabulary, image partitions, cell counts, exclusion counts, and source/file summaries.
- Produces: `build_audit(source: SourceIndex) -> AuditResult`.

- [ ] **Step 1: Write failing end-to-end fixture tests**

  Assert split-before-filter behavior, same-image question co-location, train-only vocabulary, answer auditing, exact exclusion counts, eight relation-answer cells, and absence of scene/program/object fields from offline manifest records. Relation and traceability IDs remain offline; this PR exposes no dataset or model-batch API.

- [ ] **Step 2: Verify red, implement manifest construction, verify green**

  Run the Task 6 unittest command. Expected: PASS.

- [ ] **Step 3: Write failing coverage-gate tests**

  Test an empty required partition, missing train answer class, missing train relation, and missing evaluation relation-answer cell. The first three are hard failures; the empty evaluation cell remains in the eight-cell report with count zero and accuracy `NA`.

- [ ] **Step 4: Implement gates, run targeted/full tests, and commit**

  Commit: `feat: build audited CLEVR manifests`

### Task 8: Add the Explicit Audit CLI and Run the Real-Data Acceptance

**Files:**
- Create: `run_clevr_data_audit.py`
- Modify: `tests/test_clevr_audit.py`
- Generate outside Git: `runs/clevr_data_audit/` only after tests pass.

**Interfaces:**
- Produces CLI: `python run_clevr_data_audit.py --data-root PATH --output-dir PATH`.
- Produces: `image_partitions.json`, `vocabulary.json`, `selected_questions.jsonl`, and `audit_report.json` using direct IDs/counts/sizes and no digests.

- [ ] **Step 1: Write failing CLI behavior tests**

  Invoke the real CLI entry function on temporary fixtures. Assert explicit-root requirement, refusal to overwrite a nonempty output directory, complete files on success, no partial final files on failure, and no download/network option.

- [ ] **Step 2: Verify red, implement CLI serialization, verify green**

  Run the Task 6 unittest command. Write to a temporary sibling directory and rename only after the complete audit succeeds.

- [ ] **Step 3: Run all verification and commit**

  Run targeted tests, the non-hash regression suite from Task 2, `git diff --check`, and status inspection.

  Commit: `feat: add CLEVR data audit command`

- [ ] **Step 4: Run the approved real-data audit without training**

  Run under the locked environment against `data/CLEVR_v1.0`, writing to ignored `runs/clevr_data_audit/`. Record actual image/question/cell/exclusion counts, schema failures, readability results, elapsed CPU/I/O time, and environment versions. Do not inspect confirmation-set model outputs because none exist.

- [ ] **Step 5: Deliver PR C**

  Push and create a real PR with library/license decisions, red-green evidence, real-data report summary, full-suite status, and limitations. Stop before any VQA model implementation or GPU smoke.

## Completion Boundary

This plan is complete only when PRs A, B1, B2, and C are separately reviewable and the real-data audit succeeds. It does **not** authorize or implement the VQA model, G/B/M10/Q-only arms, optimizer, checkpointing, GPU smoke, pilot, test-set evaluation, or formal training. Those require their own approved plans and PRs.
