# CLEVR PUB-R1.1 Statistics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure, manifest-anchored statistics core that independently recomputes every PUB-R1.1 formal run metric, five-seed comparison, and common image-cluster bootstrap interval.

**Architecture:** Immutable target and prediction records form the trust boundary: authoritative manifest targets define the train and base-test cohorts, while every prediction run must exactly cover and relabel-check the base-test IDs. One pure module computes per-run classification metrics, fixed five-seed summaries, preregistered paired contrasts, and a single shared PCG64 image-weight matrix for bootstrap replicates; file parsing and report publication remain a later PR.

**Tech Stack:** Python 3.13 standard library (`dataclasses`, `math`, `statistics`), installed NumPy 2.5.3 (`Generator`, `PCG64`, `multinomial`, `quantile`), `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-statistics-design.md`

## Global Constraints

- Authority is PUB-R1.1 section 7; this unit does not run training, evaluation, RGB loading, model loading, or GPU work.
- Formal methods are exactly `G`, `B`, `M10`, `Q-only`; formal seeds are exactly `101,102,103,104,105`.
- Inputs must include nonempty authoritative `train` and `base_test` targets. Every run must exactly cover the authoritative base-test question IDs and copy their labels exactly.
- Prediction is `1` only when logit 1 is strictly greater than logit 0; ties predict no (`0`).
- Empty relation-answer cells remain present with `accuracy=None`; if any cell is empty, eight-cell macro is `None`.
- Train-majority uses only authoritative train targets and predicts no on a tie.
- Five-seed summaries use sample SD and `t_{4,0.975}=2.7764451051977987`.
- Bootstrap is exactly seed 31001, 2000 draws, image-cluster multinomial weights, common across all methods/seeds, and NumPy `quantile(..., method="linear")`.
- No new dependency, download, hash, file I/O, source-data mutation, model result, or scientific performance claim.
- The current Python environment lacks PyTorch. The passing regression gate is the full non-PyTorch suite; also run full discovery and preserve its seven pre-existing `ModuleNotFoundError: torch` collection errors as an environment limitation.

## Review Focus

- All 20 runs share the same wrong or truncated cohort: compare each run to authoritative base-test targets, not merely to each other (Task 1 tests exact missing/extra IDs).
- Finite but extreme equal logits: cross-entropy remains `log(2)` without large-value cancellation, while an unrepresentable loss is rejected (Task 1 extreme-logit tests).
- Python equality aliases (`True == 1`) and permissive numeric coercion: reject booleans for IDs, labels, seeds, and logits; reject string/nested/non-length-two logits (Task 1 schema tests).
- Bootstrap accidentally advances one RNG per run or averages per-image accuracies: one common weight matrix and unequal image question counts must reproduce literal weighted examples (Task 3 bootstrap tests).
- Reordering mappings/rows changes RNG-to-image assignment: canonicalize targets, runs, questions, and images before all metrics and draws (Task 3 permutation-invariance test).

---

### Task 1: Manifest-Anchored Per-Run Metrics

**Files:**
- Create: `src/clevr_statistics.py`
- Create: `tests/test_clevr_statistics.py`

**Interfaces:**
- Consumes: no earlier task interfaces; follows the design and existing CLEVR `no=0/yes=1`, relation order, ID, and filename contracts.
- Produces: immutable records with these exact fields:
  - `TargetRow(partition: str, official_split: str, image_index: int, image_filename: str, question_index: int, relation: str, label: int)`;
  - `PredictionRow(question_index: int, label: int, logits: tuple[float, float])`;
  - `RunPredictions(method: str, seed: int, rows: tuple[PredictionRow, ...])`;
  - `CellMetrics(relation: str, label: int, count: int, correct_count: int, accuracy: float | None)`;
  - `ClassificationMetrics(question_count: int, correct_count: int, accuracy: float, cross_entropy: float | None, cells: tuple[CellMetrics, ...], macro_accuracy: float | None)`;
  - `RunMetrics(method: str, seed: int, metrics: ClassificationMetrics)`;
  - `evaluate_run(targets: Sequence[TargetRow], run: RunPredictions) -> RunMetrics`.

- [ ] **Step 1: Write record and schema tests before production code**

Add `RecordContractTests` proving records are frozen and reject: boolean/negative IDs; partition/split mismatch; filename/index mismatch; unknown relation; boolean/nonbinary label; non-sequence, wrong-length, nested, boolean, NaN, or infinite logits; unknown method; boolean/nonformal seed; list fields where immutable tuple is required.

- [ ] **Step 2: Run the record tests and verify RED**

Run: `python -m unittest tests.test_clevr_statistics.RecordContractTests -v`  
Expected: ERROR importing missing `src.clevr_statistics`.

- [ ] **Step 3: Implement immutable input/output records and scalar validation**

In `src/clevr_statistics.py`, define exact constants:

```python
METHODS = ("G", "B", "M10", "Q-only")
SEEDS = (101, 102, 103, 104, 105)
RELATIONS = ("left", "right", "front", "behind")
```

Define the produced dataclasses with tuple fields and strict `type(value) is int` / finite-real checks. `TargetRow` accepts only `(train, train)` or `(base_test, val)` partition/split pairs and validates `CLEVR_<split>_<index:06d>.png`. Output records reject internally inconsistent counts/ranges so later tasks cannot construct invalid reports.

- [ ] **Step 4: Run record tests and verify GREEN**

Run: `python -m unittest tests.test_clevr_statistics.RecordContractTests -v`  
Expected: PASS.

- [ ] **Step 5: Write per-run metric tests**

Add `RunMetricTests` with literal fixtures and assertions for:

- eight base-test rows (all relation × label cells), zero logits, tie→0, `correct=4`, `accuracy=0.5`, `cross_entropy=math.log(2)`, answer-0 cell accuracy 1, answer-1 cell accuracy 0, macro 0.5;
- an empty-cell cohort retaining all eight cells and `macro_accuracy is None`;
- extreme equal logits `(1e308,1e308)` returning `log(2)`, and `(-1000,1000)` for label 1 returning a finite near-zero loss;
- target/run/row reordering producing equal results without mutating the input tuples;
- duplicate target IDs, missing train or base-test targets, duplicate prediction IDs, missing/extra prediction IDs, and copied-label mismatch each raising `ValueError` with the offending condition;
- finite logits whose target-vs-other subtraction overflows and yields nonfinite loss raising `ValueError` rather than clipping.

- [ ] **Step 6: Run per-run metric tests and verify RED**

Run: `python -m unittest tests.test_clevr_statistics.RunMetricTests -v`  
Expected: FAIL because `evaluate_run` is missing.

- [ ] **Step 7: Implement target anchoring and `evaluate_run`**

Canonicalize targets by `(partition, question_index)` and predictions by `question_index`. Validate both target partitions once per call, derive the authoritative base-test map, require exact ID-set and label equality, then compute `predicted = int(logit1 > logit0)`. Compute loss as `numpy.logaddexp(0.0, other_logit - target_logit)`, reject nonfinite losses, and build eight cells in `RELATIONS` then `(0,1)` order. Macro is defined only when all eight cell accuracies are non-`None`.

- [ ] **Step 8: Run Task 1 tests and non-PyTorch regressions**

Run: `python -m unittest tests.test_clevr_statistics.RecordContractTests tests.test_clevr_statistics.RunMetricTests -v`  
Expected: PASS.

Run: `python -m unittest tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_text tests.test_clevr_audit tests.test_clevr_statistics -v`  
Expected: PASS with zero failures/errors.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/clevr_statistics.py tests/test_clevr_statistics.py
git commit -m "feat: compute manifest-anchored CLEVR metrics"
```

### Task 2: Majority Baseline and Five-Seed Comparisons

**Files:**
- Modify: `src/clevr_statistics.py`
- Modify: `tests/test_clevr_statistics.py`

**Interfaces:**
- Consumes: Task 1 `TargetRow`, `RunMetrics`, target validation and fixed constants.
- Produces: immutable `SeedSummary(name: str, seeds: tuple[int, ...], values: tuple[float, ...], mean: float, sample_sd: float, interval: tuple[float, float])`; immutable `PairedSummary` with the same fields; `evaluate_train_majority(targets: Sequence[TargetRow]) -> ClassificationMetrics`; `summarize_seed_values(name: str, values: Mapping[int, float]) -> SeedSummary`; `summarize_paired_differences(name: str, minuend: Mapping[int, float], subtrahend: Mapping[int, float]) -> PairedSummary`.

- [ ] **Step 1: Write majority-baseline tests**

Add `MajorityBaselineTests` proving a 2/2 train tie predicts 0 on every base-test row, a 2/1 train majority predicts 1, only train labels choose the class, cross-entropy is `None`, cells/macro use base-test targets, and target reordering is invariant.

- [ ] **Step 2: Run majority tests and verify RED**

Run: `python -m unittest tests.test_clevr_statistics.MajorityBaselineTests -v`  
Expected: FAIL because `evaluate_train_majority` is missing.

- [ ] **Step 3: Implement `evaluate_train_majority`**

Reuse authoritative target validation and the Task 1 classification aggregation path with constant predictions. Do not manufacture logits or derive the class from base-test labels.

- [ ] **Step 4: Run majority tests and verify GREEN**

Run: `python -m unittest tests.test_clevr_statistics.MajorityBaselineTests -v`  
Expected: PASS.

- [ ] **Step 5: Write seed and paired-summary tests**

Add `SeedSummaryTests` asserting for seeds 101..105 and values `(-0.02,-0.01,0,0.01,0.02)`:

- values are emitted in seed order, mean is 0, sample SD is `sqrt(0.001/4)`, and interval is exactly mean ± `2.7764451051977987 * sd / sqrt(5)` within `assertAlmostEqual` tolerance;
- constant values have zero SD and degenerate interval;
- missing/extra/boolean seed, empty/blank name, and boolean/NaN/infinite values fail;
- paired summaries subtract same-seed values before summarizing and reject mismatched seed sets.

- [ ] **Step 6: Run seed tests and verify RED**

Run: `python -m unittest tests.test_clevr_statistics.SeedSummaryTests -v`  
Expected: FAIL because summary functions are missing.

- [ ] **Step 7: Implement five-seed summaries**

Use `statistics.fmean`, `statistics.stdev`, `math.sqrt(5)`, and the named t constant. Require exact seed keys before converting to ordered tuples. Compute paired values seed-by-seed, never by subtracting independent means.

- [ ] **Step 8: Run Task 2 tests and non-PyTorch regressions**

Run: `python -m unittest tests.test_clevr_statistics.MajorityBaselineTests tests.test_clevr_statistics.SeedSummaryTests -v`  
Expected: PASS.

Run: `python -m unittest tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_text tests.test_clevr_audit tests.test_clevr_statistics -v`  
Expected: PASS with zero failures/errors.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/clevr_statistics.py tests/test_clevr_statistics.py
git commit -m "feat: summarize CLEVR seed comparisons"
```

### Task 3: Shared Image-Cluster Bootstrap and Formal Report Object

**Files:**
- Modify: `src/clevr_statistics.py`
- Modify: `tests/test_clevr_statistics.py`

**Interfaces:**
- Consumes: Task 1 records/evaluator; Task 2 majority and summary functions.
- Produces: immutable `ClusterBootstrapSummary(seed: int, draws: int, quantile_method: str, replicates: tuple[float, ...], interval: tuple[float, float])`; immutable `CLEVRStatistics(run_metrics: tuple[RunMetrics, ...], method_summaries: tuple[SeedSummary, ...], b_minus_g: PairedSummary, m10_minus_g: PairedSummary, train_majority: ClassificationMetrics, bootstrap: ClusterBootstrapSummary)`; `compute_clevr_statistics(targets: Sequence[TargetRow], runs: Sequence[RunPredictions]) -> CLEVRStatistics`.

- [ ] **Step 1: Write formal-grid and cluster-bootstrap tests**

Add `FormalStatisticsTests` that create all 20 run fixtures and assert fixed method/seed ordering, 20 run metrics, four method summaries, `B-G` and `M10-G` same-seed values, majority metrics, and rejection of missing, extra, duplicate, or unknown method-seed entries.

Add `ClusterBootstrapTests` proving:

- two images with question counts `(1,3)`, G correct `(1,0)`, B correct `(0,3)`, and the literal first PCG64 weight rows `[(0,2),(1,1),(0,2),(1,1),(0,2)]` produce first replicate differences `(1,0.5,1,0.5,1)` for identical five seeds;
- 2000 replicates are returned with seed 31001 and `quantile_method="linear"`, and interval endpoints equal `numpy.quantile` on the returned replicates;
- a one-image cohort yields every replicate equal to the five-seed mean main difference and a degenerate interval;
- unequal question counts differ from the incorrect equal-per-image mean on the hand example;
- reversing target, run, and prediction-row order leaves the entire `CLEVRStatistics` object exactly equal and does not change NumPy global RNG state (compare scalar tuple members directly and the state array with `numpy.array_equal`, never with a hash).

- [ ] **Step 2: Run Task 3 tests and verify RED**

Run: `python -m unittest tests.test_clevr_statistics.FormalStatisticsTests tests.test_clevr_statistics.ClusterBootstrapTests -v`  
Expected: FAIL because `compute_clevr_statistics` is missing.

- [ ] **Step 3: Implement formal-grid orchestration and one common bootstrap weight matrix**

Canonicalize runs by `(METHODS index, seed)` and require the exact Cartesian product. Evaluate every run against the same authoritative targets, form per-method accuracy mappings, then call Task 2 summaries and majority evaluator. Sort base-test images by `(image_index, image_filename)`, build integer question counts `[I]` and correct-count matrices `[5,I]` for G and B, and generate exactly one `[2000,I]` matrix with `Generator(PCG64(31001)).multinomial(I, [1/I]*I, size=2000)`. Use matrix products to obtain per-draw per-seed weighted micro accuracies, average the five B−G differences per draw, convert all outputs to Python tuples/floats, and compute linear 2.5%/97.5% quantiles.

- [ ] **Step 4: Run all statistics tests and non-PyTorch regressions**

Run: `python -m unittest tests.test_clevr_statistics -v`  
Expected: PASS with zero failures/errors.

Run: `python -m unittest tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_text tests.test_clevr_audit tests.test_clevr_statistics -v`  
Expected: PASS with zero failures/errors.

- [ ] **Step 5: Run full discovery and verify only the baseline environment errors remain**

Run: `python -m unittest discover -s tests -v`  
Expected: the new statistics tests and all non-PyTorch tests PASS; exactly the same seven pre-existing modules fail collection with `ModuleNotFoundError: No module named 'torch'`. Any new failure is a Task 3 failure.

- [ ] **Step 6: Run static repository checks**

Run: `python -m compileall -q src/clevr_statistics.py tests/test_clevr_statistics.py`  
Expected: exit 0.

Run: `git diff --check`  
Expected: exit 0 with no output.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/clevr_statistics.py tests/test_clevr_statistics.py
git commit -m "feat: add CLEVR image-cluster bootstrap"
```

### Task 4: Whole-Branch Verification and Review Handoff

**Files:**
- Modify only if the fresh reviewer finds a Critical or Important issue, always with a new failing regression test first.

**Interfaces:**
- Consumes: Tasks 1–3 and the design document.
- Produces: review package, final evidence, PR-ready branch; no new runtime interface.

- [ ] **Step 1: Verify plan/spec diff scope and full non-PyTorch suite**

Run: `git diff --check`  
Expected: exit 0.

Run: `python -m unittest tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_text tests.test_clevr_audit tests.test_clevr_statistics -v`  
Expected: PASS with zero failures/errors.

- [ ] **Step 2: Run full discovery and record the known dependency boundary**

Run: `python -m unittest discover -s tests -v`  
Expected: statistics/non-PyTorch tests PASS and only the seven baseline `torch` import errors remain. Do not claim the full suite passes.

- [ ] **Step 3: Create the executing-plans review package and obtain a fresh whole-branch review**

Review against `docs/superpowers/specs/2026-10-25-clevr-statistics-design.md`, this plan, its Review Focus, and all ledger rulings. Re-grade findings by user impact. Fix Critical/Important items in one RED→GREEN pass; ledger Minor findings without implementing them.

- [ ] **Step 4: Re-run verification after any fix and commit**

Run Task 4 Steps 1–2 and `python -m compileall -q src/clevr_statistics.py tests/test_clevr_statistics.py`.  
Expected: same verified state, with no new failures.

- [ ] **Step 5: Prepare and attempt the real PR**

Push `feat/clevr-statistics` and create a non-draft PR to `master` whose body includes problem/trigger, formulas, official library/License sources, changes, RED→GREEN evidence, baseline dependency limitation, fresh review, and exclusions. If push/PR tooling or authentication is blocked, record exact commands/state and continue to the next independent plan unit as authorized; do not fabricate a PR.
