# Cross-run CLEVR data contract

## Failure, definition, and invariants

At baseline, fresh G and B runs in the same phase/seed may carry different
audit snapshots while sharing identical initialized tensors. The statistics
loader accepts predictions without training provenance. This does not establish
the paired comparison required by PUB-R1.1.

Let A be the ordered mapping of four UTF-8 source texts: selected_questions.jsonl,
vocabulary.json, audit_report.json, image_partitions.json. For every arm m and
seed s in phase p require A[p,m,s] = A[p], using direct text equality, not hashes.
For RGB arms also require the same resolved explicit data_root; Q-only has no
RGB root. E.g. vocabularies with [NULL,START,END,UNK,red] and
[NULL,START,END,UNK,blue] have the same size but must fail equality. Matching
copies in different audit directories may pass; even whitespace-only edits to
the frozen text are rejected, so retaining the original audit files is required.

Model inputs, labels, sampling, G/B rotations, optimization and metric formulas
are unchanged. This is provenance enforcement, not a new evaluation slice or
sample filter. No arrays/tensors or trainable parameters are added to the model.

## Implementation plan

1. Reuse the existing atomic no-overwrite artifact publication to establish a
   phase-level paired_data.json and, for RGB arms, rgb_data_root.json. All
   production train/resume/evaluate entries validate them before checkpoint or
   model computation. Existing run_config files in that phase must also agree;
   missing old provenance is an error, never inferred from predictions.
2. Include image_partitions.json in the existing runner audit snapshot.
3. Before statistics, require each prediction's original run_config, verify
   formal method/seed/budget and all four snapshots against the supplied audit,
   then compare RGB roots across arms. Accept an explicit config path, the
   runner's fixed relative config location, or the exported adjacent config.
4. Publish the verified configs alongside predictions; schema v2 records their
   paths. Old reports stay intact. No silent upgrade of old evidence.

Direct snapshot mismatch, missing config, wrong method/seed/phase, or changed
RGB root falsifies pairing. Verify those failures, Q-only first, concurrent
first writers, unchanged reuse, report export/reload and runner ordering.

## Mature-library review (2026-10-28)

Python 3.11 stdlib json/pathlib (PSF license) and existing project publication
helpers cover the required mapping, path and atomic-write operations; no
third-party data/versioning system is needed. Python is maintained upstream;
no assertion that 3.11 is the newest release. References:
[json](https://docs.python.org/3.11/library/json.html),
[pathlib](https://docs.python.org/3.11/library/pathlib.html),
src/clevr_run_artifacts.py `_publish`, src/clevr_reporting.py `_read_json_document`.
Reuse these functions; add only the domain-specific pairing invariant. The
official CLEVR source interface cannot bind separate local experiment runs.

## Limits

Recorded files and explicit roots are compared directly; this does not prove
pixel identity if someone mutates image bytes in place after the offline audit.
It is not protection against deliberately forged run configurations. Tests use
CPU synthetic fixtures only. There is no change to the scientific evaluation
definitions and no experiment launch or user-data access.

## Report invocation and compatibility

`run_clevr_statistics.py` retains the explicit `--audit-dir`, repeated
`--prediction-file`, and `--output-dir` interface. For a runner prediction at
`<run>/predictions/base_test/<method>_seed_<seed>.jsonl`, it reads
`<run>/run_config.json`. For an exported or flat file it requires the adjacent
same-stem `.run_config.json`. An alternate location can be supplied by repeating
`--run-config-file` once per prediction, in the same argument order.

The newly produced report uses schema 2 and exports those verified original
configurations next to its predictions. Missing provenance is a hard error;
historical schema-1 files remain readable as historical artifacts, but cannot
be asserted to satisfy this new pairing check without their original configs.
Adding image_partitions.json to the frozen runtime snapshot means old run
configs with only three texts cannot resume under this stricter runner. Do not
edit old configs to force acceptance; retain their evidence and use a new run
root after the appropriate user authorization.
