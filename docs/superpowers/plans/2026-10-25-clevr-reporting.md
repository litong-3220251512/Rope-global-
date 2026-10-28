# CLEVR PUB-R1.1 Reporting Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strictly parse one published PUB-R1.1 audit bundle and twenty self-described per-question prediction files, independently recompute all frozen statistics, and atomically publish a self-contained reproducible report bundle.

**Architecture:** `src/clevr_reporting.py` is the trust boundary: strict standard-library JSON readers cross-check the audit artifacts, parse self-described predictions into the immutable statistics API, build a typed report/bundle, serialize only the frozen output schema, and publish canonical targets/predictions plus the report through a same-parent staging directory. `run_clevr_statistics.py` is a thin explicit-path CLI and never discovers files or computes metrics itself.

**Tech Stack:** Python 3.13.11 standard library (`json`, `argparse`, `dataclasses`, `os`, `pathlib`, `platform`, `tempfile`, `shutil`, `importlib.metadata`), installed NumPy 2.5.3 through `src.clevr_statistics`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-reporting-design.md`

## Global Constraints

- PUB-R1.1 is the only protocol; no data/weight download, RGB/scene/checkpoint access, training, evaluation execution, GPU work, process monitoring, or scientific performance claim.
- Inputs are exactly one explicit audit directory and twenty explicit, distinct prediction paths; no recursive search, naming guess, best-seed selection, or runner metric ingestion.
- Audit bundle fixed inputs are `image_partitions.json`, `selected_questions.jsonl`, and `audit_report.json`; protocol, image partition, selected counts, and summary counts must agree.
- Prediction rows have exactly `protocol_version,method,seed,question_index,label,logits`; every row self-declares one consistent formal method/seed and exact protocol.
- Formal grid is exactly `G/B/M10/Q-only × 101..105`; every run still passes the statistics core's exact base-test ID and copied-label checks.
- Published bundle retains canonical `targets.jsonl`, all twenty canonical prediction JSONLs, and complete `statistics_report.json`, including all 2,000 bootstrap replicates. No hash/digest is computed.
- Only metric fields `CellMetrics.accuracy`, `ClassificationMetrics.macro_accuracy`, and `ClassificationMetrics.cross_entropy` may serialize `None` as `"NA"`; no JSON null is accepted as an input shortcut.
- Publication targets a new directory. The `O_EXCL` lock only guarantees no-overwrite among cooperating publishers; rename is the commit point, and cleanup errors must report whether commit occurred.
- No new dependency. Python standard library is PSF-2.0; NumPy is BSD-3-Clause. The exact library/API rationale is in the spec.
- The current Python environment lacks PyTorch. The passing regression gate is the full non-PyTorch suite; full discovery must separately preserve and report the seven pre-existing `ModuleNotFoundError: torch` collection errors.

## Review Focus

- **Caller relabels or reuses a run:** prediction method/seed/protocol must come from every row, one file must be internally consistent, and duplicate resolved source paths must fail before any prediction file is opened (Task 2 tests).
- **Schema-valid but unrelated manifest:** protocol-bearing audit files, image membership, mother-image counts, selected-image counts, and selected-question counts must cross-check before targets are trusted (Task 1 tests).
- **Python JSON extensions hide corruption:** duplicate keys at any nesting depth, NaN/Infinity, blank JSONL lines, booleans as integers, missing fields, and extra runner summaries must fail with source context (Tasks 1–2 tests).
- **Published report cannot be independently recomputed:** canonical targets and all twenty canonical prediction files must be present, ordered, self-described, and byte-invariant under input reordering (Task 3 tests).
- **Atomicity claim crosses the commit point:** pre-rename failures leave no final/staging, while lock-unlink failure explicitly distinguishes `not committed` from `committed` and preserves the undeletable lock (Task 3 tests).

---

### Task 1: Strict Audit-Bundle Projection

**Files:**
- Create: `src/clevr_reporting.py`
- Create: `tests/test_clevr_reporting.py`

**Interfaces:**
- Consumes: `src.clevr_partitions.ImageIdentity`; `src.clevr_statistics.TargetRow`.
- Produces:
  - `CLEVRReportError(ValueError)`;
  - frozen `ArtifactEvidence(path: str, size_bytes: int, record_count: int)`;
  - frozen `PartitionCounts(train: int, dev: int, base_test: int, confirmation: int)`;
  - `load_audit_targets(audit_dir: Path) -> tuple[tuple[TargetRow, ...], tuple[ArtifactEvidence, ...], PartitionCounts]`.

- [ ] **Step 1: Write strict audit-bundle tests before production code**

Add `AuditBundleTests` with a real temporary directory containing literal audit JSON/JSONL. Assert a valid bundle returns targets in fixed `train,base_test` then question-ID order; evidence order is `image_partitions.json,selected_questions.jsonl,audit_report.json`; evidence sizes come from actual files; record counts are total partition images, manifest rows, and one report document; and partition counts match literal rows. Prove output records are frozen and reject boolean/negative counts, blank/nonabsolute evidence paths, and inconsistent field types.

Add focused cases proving hard failure, with path/line context where applicable, for: absent/not-file audit inputs; wrong/boolean protocol fields; duplicate JSON keys including one nested object; NaN/Infinity; blank manifest line; non-object row; exact-key mismatch; booleans/negative IDs; split/filename/image-path mismatch; nonstring question; invalid encoded token; nonbinary answer; unknown relation/partition; duplicate question/source positions; duplicate image identities; wrong official split per partition; selected image absent from or assigned to another image partition; wrong mother/selected-image/selected-question summary; wrong summary order/schema; and missing train or base-test targets.

Name the production mutation each grouped test catches in a short test comment. Expected targets and counts must be literal, not generated through production helpers.

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run: `python -m unittest tests.test_clevr_reporting.AuditBundleTests -v`

Expected: import error because `src.clevr_reporting` does not exist.

- [ ] **Step 3: Implement immutable evidence records and strict JSON readers**

In `src/clevr_reporting.py`, define `PROTOCOL_VERSION = "PUB-R1.1"`, fixed partition/key tuples, strict nonnegative-int checks, a duplicate-rejecting `object_pairs_hook`, and a `parse_constant` callback that raises. Full-document and JSONL readers must use an opened UTF-8 descriptor, record `os.fstat(fd).st_size`, reject blank lines, and wrap JSON/Unicode/OSError/domain failures in `CLEVRReportError` with resolved path and 1-based line when relevant.

Do not add a size threshold, hash, schema dependency, global cache, or source mutation.

- [ ] **Step 4: Implement `load_audit_targets` and cross-artifact checks**

Require `audit_dir` to be a `Path` resolving to a directory. Parse the three fixed files only. Construct `ImageIdentity` records, enforce global image uniqueness and official split per partition, validate the exact audit report top-level and partition-summary schema, validate every selected manifest field, cross-check image membership and all three summary counts, then project only train/base-test rows into `TargetRow`.

Canonical target order is partition rank `train=0, base_test=1`, then `question_index`. Dev/confirmation rows are validated and counted but never projected.

- [ ] **Step 5: Run Task 1 and non-PyTorch regression tests**

Run: `python -m unittest tests.test_clevr_reporting.AuditBundleTests -v`

Expected: PASS.

Run: `python -m unittest tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_text tests.test_clevr_audit tests.test_clevr_statistics tests.test_clevr_reporting -v`

Expected: all tests pass with zero failures/errors.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/clevr_reporting.py tests/test_clevr_reporting.py
git commit -m "feat: parse authoritative CLEVR audit bundles"
```

### Task 2: Self-Described Predictions and Typed Report

**Files:**
- Modify: `src/clevr_reporting.py`
- Modify: `tests/test_clevr_reporting.py`

**Interfaces:**
- Consumes: Task 1 `load_audit_targets`; `src.clevr_statistics.{METHODS,SEEDS,PredictionRow,RunPredictions,CLEVRStatistics,compute_clevr_statistics}`.
- Produces:
  - frozen `PredictionSource(path: Path)`;
  - frozen `PredictionEvidence(method: str, seed: int, source_path: str, size_bytes: int, row_count: int, canonical_path: str)`;
  - frozen `EnvironmentEvidence(python: str, numpy: str)`;
  - frozen `StatisticsReport(protocol_version: str, report_schema_version: int, audit_sources: tuple[ArtifactEvidence, ...], targets_path: str, manifest_partition_counts: PartitionCounts, predictions: tuple[PredictionEvidence, ...], statistics: CLEVRStatistics, environment: EnvironmentEvidence)`;
  - frozen `StatisticsBundle(report: StatisticsReport, targets: tuple[TargetRow, ...], runs: tuple[RunPredictions, ...])`;
  - `load_prediction_run(source: PredictionSource) -> tuple[RunPredictions, PredictionEvidence]`;
  - `build_statistics_bundle(audit_dir: Path, prediction_sources: Sequence[PredictionSource]) -> StatisticsBundle`;
  - `statistics_report_payload(report: StatisticsReport) -> dict[str, object]`.

- [ ] **Step 1: Write prediction-file tests and verify RED**

Add `PredictionFileTests` for a literal valid self-described file and assert exact `RunPredictions`, source evidence, and canonical name `<method>_seed_<seed>.jsonl`. Prove every new public record is frozen and rejects inconsistent direct construction. Add failures for wrong source type/path; empty file; blank line; non-object; duplicate/nonnormal JSON values; exact-key mismatch including a runner `accuracy`; wrong protocol; unknown/boolean method or seed; method/seed disagreement between rows; duplicate question ID; and every invalid `PredictionRow` boundary.

Run: `python -m unittest tests.test_clevr_reporting.PredictionFileTests -v`

Expected: FAIL because the prediction interfaces are missing.

- [ ] **Step 2: Implement and verify one prediction loader**

Use Task 1's strict JSONL reader. Parse all rows before constructing one `RunPredictions`; derive method/seed from the first row and require exact agreement on every subsequent row. Convert only the three statistical fields into `PredictionRow`. Sort rows by question ID. Record resolved source path, descriptor size, row count, and deterministic canonical relative path.

Run the focused class again; expected PASS.

- [ ] **Step 3: Write complete-grid/report/payload tests and verify RED**

Add `StatisticsBundleTests` using one literal audit bundle and twenty real prediction files. The balanced eight-cell `(0,0)`-logit fixture must hand-check: every run accuracy `0.5`, CE `log(2)`, no-cell accuracy `1`, yes-cell accuracy `0`, macro `0.5`, tied train-majority predicts no with CE `"NA"`, both paired summaries are all zero, and all 2,000 bootstrap replicates are zero.

Assert exact method/seed/evidence ordering, protocol/schema/environment fields, all fixed top-level payload keys, all 2,000 replicates, and `"NA"` only at the explicitly nullable metric paths with no JSON null. Reordering manifest lines, source arguments, and prediction rows must preserve statistics, ordered records, and the payload.

Add pre-I/O rejection for 19, 21, non-`PredictionSource`, and duplicate resolved paths using otherwise nonexistent files. Add post-read rejection for duplicate/missing formal identities. Prove a file copied to a distinct path cannot occupy another grid cell unless its own row metadata changes; do not reject legitimately equal logits across different runs.

Run: `python -m unittest tests.test_clevr_reporting.StatisticsBundleTests -v`

Expected: FAIL because bundle/report builders are missing.

- [ ] **Step 4: Implement exact-grid orchestration and typed serialization**

Validate sequence type, exact length 20, source types, and unique resolved paths before opening any prediction file. Load all files, require the exact Cartesian grid, canonicalize by `METHODS` then `SEEDS`, call `compute_clevr_statistics`, and construct the frozen report and bundle with `platform.python_version()` and `importlib.metadata.version("numpy")`.

Implement explicit payload functions per statistics dataclass rather than generic `asdict`/recursive-None replacement. Emit `"NA"` only for cell accuracy, classification macro accuracy, and classification cross-entropy. Use tuples internally and JSON-compatible lists/dicts in the payload.

- [ ] **Step 5: Run Task 2 and regression tests**

Run the two reporting classes; expected PASS.

Run the full non-PyTorch command from Task 1; expected all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/clevr_reporting.py tests/test_clevr_reporting.py
git commit -m "feat: build independent CLEVR statistics reports"
```

### Task 3: Canonical Bundle Publication and Explicit CLI

**Files:**
- Modify: `src/clevr_reporting.py`
- Create: `run_clevr_statistics.py`
- Modify: `tests/test_clevr_reporting.py`

**Interfaces:**
- Consumes: Task 2 `StatisticsBundle`, `build_statistics_bundle`, `statistics_report_payload`.
- Produces:
  - `ReportPublicationError(CLEVRReportError)` carrying `committed: bool`;
  - `publish_statistics_bundle(output_dir: Path, bundle: StatisticsBundle) -> None`;
  - CLI `main(argv: Sequence[str] | None = None) -> int` with required `--audit-dir`, exactly twenty repeatable `--prediction-file`, and required `--output-dir`.

- [ ] **Step 1: Write canonical-bundle and CLI success tests; verify RED**

Add `PublicationTests` that run the real CLI against the Task 2 fixture and assert exit code 0 and exactly: root `statistics_report.json`, root `targets.jsonl`, and `predictions/` with twenty fixed filenames. Parse every output as strict JSON; assert canonical target/run row order and exact six/seven-field schemas, final newline, no digest-like keys, no JSON null, and report evidence paths/counts.

Independently reconstruct `TargetRow`/`RunPredictions` from the published canonical files, call `compute_clevr_statistics`, and assert equality with the published report's hand-checked scalar/list values. Reverse all source and row orders into a second invocation under a different output path, normalize only source absolute paths in the two report payloads, and assert every canonical artifact byte is otherwise identical.

Run: `python -m unittest tests.test_clevr_reporting.PublicationTests.test_cli_publishes_complete_recomputable_canonical_bundle -v`

Expected: import/failure because publisher and CLI are missing.

- [ ] **Step 2: Implement deterministic artifact writers and thin CLI**

Canonical JSONL uses `ensure_ascii=False, sort_keys=True, allow_nan=False` and one newline per row. Report JSON adds `indent=2` and one terminal newline. Write targets/runs from `StatisticsBundle`, never reread input or accept runner metrics. `run_clevr_statistics.py` parses `Path` values with `allow_abbrev=False`, constructs `PredictionSource` records, then only calls build and publish.

Run the success test; expected PASS.

- [ ] **Step 3: Write publication failure/commit-point tests; verify RED**

Add real-filesystem tests for existing file, existing empty/nonempty directory, output symlink, pre-existing lock, and output appearing after lock acquisition. Assert no existing target changes.

Inject failures separately during canonical JSONL writing, report JSON writing, and rename; each must leave no final directory or staging and, when unlink works, no lock. Inject lock unlink failure before commit together with a writer failure and assert `ReportPublicationError.committed is False`, no final/staging, and retained lock. Inject lock unlink failure after successful rename and assert `committed is True`, complete final bundle, no staging, and retained lock. Do not assert mock calls; assert filesystem state and exception fields/messages.

Run the failure tests; expected FAIL until transaction handling is implemented.

- [ ] **Step 4: Implement cooperative-lock transaction and explicit commit status**

Validate a new non-symlink target before lock acquisition; create parent and `.<output-name>.lock` with `os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600`; create staging via `tempfile.mkdtemp` in that parent; write all artifacts; recheck target; rename staging to output; mark committed; then unlink lock.

On primary error, remove only this invocation's staging. If lock unlink succeeds, re-raise the primary error. If lock unlink fails, raise `ReportPublicationError` chained from the cleanup error and preserve `committed`. Never remove a committed output or an existing target. Document in the class docstring that noncooperating target creation between final check and POSIX rename is outside the guarantee.

- [ ] **Step 5: Run Task 3, static checks, and full regressions**

Run: `python -m unittest tests.test_clevr_reporting -v`

Expected: all reporting tests PASS.

Run: `python -m unittest tests.test_clevr_programs tests.test_clevr_partitions tests.test_clevr_text tests.test_clevr_audit tests.test_clevr_statistics tests.test_clevr_reporting -v`

Expected: all non-PyTorch tests PASS.

Run: `python -m compileall -q src tests run_clevr_statistics.py`

Expected: exit 0.

Run: `git diff --check 15a768d6594c9cbb61a363772c064ef9bbb25d79...HEAD`

Expected: no output, exit 0.

Run: `python -m unittest discover -s tests -v`

Expected: all non-PyTorch tests PASS and exactly the same seven pre-existing modules fail collection with `ModuleNotFoundError: No module named 'torch'`; any new failure is a blocker.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/clevr_reporting.py run_clevr_statistics.py tests/test_clevr_reporting.py
git commit -m "feat: publish canonical CLEVR statistics bundles"
```

### Task 4: Fresh Review, Final Verification, and PR Handoff

**Files:**
- Modify only if a reviewer-confirmed Important/Critical defect requires one minimal TDD fix.

**Interfaces:**
- Consumes: Tasks 1–3 committed branch.
- Produces: a reviewable feature branch and real stacked PR targeting `feat/clevr-statistics` when the GitHub PR interface is available.

- [ ] **Step 1: Dispatch a fresh whole-branch reviewer**

Give the reviewer the spec, plan, base-to-head diff, and explicit focus on self-declared run identity, audit cross-checks, strict JSON, canonical artifact sufficiency, serializer NA locations, and transaction commit/cleanup semantics. Require independent focused test execution and both spec-compliance and code-quality verdicts.

- [ ] **Step 2: Resolve review findings once**

Fix Critical/Important findings with a new failing regression first, then minimal code and full focused regression. Record Minor findings and scope rulings without silently expanding into runner/checkpoint provenance. Do not claim the report proves how a model was trained.

- [ ] **Step 3: Re-run final evidence from committed HEAD**

Run all Task 3 verification commands, `git status --short --branch`, `git log --oneline 15a768d6594c9cbb61a363772c064ef9bbb25d79..HEAD`, and confirm the diff contains only the design, plan, reporting module, CLI, and reporting tests.

- [ ] **Step 4: Push and create a real stacked PR when possible**

Push `feat/clevr-reporting`; create a non-draft PR with base `feat/clevr-statistics`. The PR body must include: trigger/problem, trust-boundary and mathematical basis, standard-library/NumPy versions and licenses, exact changes, TDD/review/test evidence, the seven pre-existing torch collection errors, and limitations (declared method/seed is not proof of training provenance; audit correctness remains conditional on the upstream bundle; cooperative-lock/rename and no-fsync boundaries; no real predictions or scientific result).

Verify remote/local head equality and PR state/base/head. Do not merge, tag, or claim CI. If PR creation remains unavailable, retain the pushed branch and exact compare URL and continue under the user's explicit skip-blocker instruction.
