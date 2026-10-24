# CLEVR Runtime Dataset Design

Date: 2026-10-24
Protocol: PUB-R1.1
Status: proposed for approval; no product code is authorized by this document alone

## 1. Intent

The accepted Stage 1 protocol already produces an offline-audited
`selected_questions.jsonl`, `vocabulary.json`, and `audit_report.json`. The
repository does not yet have the runtime boundary required by `AGENTS.md`:
an indexable dataset whose model-facing sample contains only official RGB,
encoded original-question tokens, the true token length, and the supervised
binary answer.

This unit fills that boundary before the common RGB+question backbone is
implemented. It does not repeat task selection, scene/program execution, or
image partitioning, and it does not train or evaluate a model.

## 2. Goals, non-goals, and anti-goals

### Goals

- Read only caller-specified local `data_root` and caller-specified audit
  artifact directory.
- Validate that runtime artifacts still satisfy the frozen PUB-R1.1 schema.
- Preserve the protocol's base order `(image_index, question_index,
  source_position)` within each requested partition.
- Decode official PNG files deterministically into `float32` RGB tensors of
  shape `[3, 224, 224]`.
- Return exactly four model-facing values: image, encoded question, true
  question length, and answer.
- Dynamically pad a batch with `<NULL>=0` without truncating any question.
- Keep traceability and audit-only fields out of every sample and batch.

### Non-goals

- No program filtering, answer recomputation, scene loading, vocabulary
  construction, partition construction, sampling, augmentation, optimization,
  checkpointing, model code, or result reporting.
- No Q-only optimization. Its requirement to avoid RGB I/O belongs to the
  later Q-only unit and must not weaken the RGB dataset contract here.
- No real-data smoke, pilot, training, or evaluation run in this PR. Tests use
  temporary fixtures only; any run over user-provided CLEVR requires separate
  authorization.
- No prediction-to-ID reporting API. A later evaluation runner may pair its
  fixed sequential evaluation order with the immutable audit manifest, but IDs
  do not enter this model batch.

### Anti-goals

- Never search for another CLEVR root, generate data, download data or weights,
  or silently fall back to a different manifest.
- Never return question text, relation, partition, official split, image path,
  image/question/source indices, program, scene graph, object data, or any
  filtering result from `__getitem__` or the collate function.
- Never accept path traversal, an absolute manifest image path, or a symlink
  that resolves outside the explicit data root.
- Never add normalization, crop, flip, color augmentation, Mixup, CutMix,
  random resize, or test-time augmentation.
- Never use hashes to bind or validate artifacts.

## 3. Considered approaches

### A. Manifest-backed map-style Dataset — selected

Use `torch.utils.data.Dataset` over the accepted audit artifacts. Constructor
validation is deterministic and inexpensive relative to the completed source
audit; image decoding remains lazy in `__getitem__`. The indexable contract is
compatible with the later explicit sampler, exact sample sequence, and
checkpoint cursor requirements.

### B. Re-run the complete source audit in every dataset constructor — rejected

This would reload scenes and programs, revalidate 85,000 images, and retain or
transitively expose offline-only structures at the runtime boundary. It is
unnecessary because the accepted audit artifacts already record the selected
task, and it would repeat the observed high-memory, multi-minute audit before
ordinary runtime use.

### C. Stream JSONL through IterableDataset — rejected

Streaming could reduce constructor storage, but the selected set is only 9,506
rows and fits comfortably in memory. IterableDataset would complicate exact
index-based sampling, worker sharding, resume cursors, and out-of-band
traceability. Those costs conflict with the frozen paired-run protocol.

## 4. Public API and ownership boundary

Add `src/clevr_dataset.py` with the conceptual API:

```python
class CLEVRQuestionDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        data_root: Path,
        audit_dir: Path,
        partition: str,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: ...


def collate_clevr_questions(
    samples: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]: ...
```

`partition` is exactly one of `train`, `dev`, `base_test`, or `confirmation`.
The constructor resolves only the two explicit roots and fixed artifact names:

- `<audit_dir>/selected_questions.jsonl`
- `<audit_dir>/vocabulary.json`
- `<audit_dir>/audit_report.json`

It neither searches parent/sibling directories nor invokes the audit builder.
The dataset retains only a private immutable tuple of minimum runtime records:
resolved image path, the three integer sort fields, encoded token tuple, and
binary answer. Audit-only row values are validated and then discarded.

## 5. Artifact contract

### 5.1 Protocol and vocabulary

Both `vocabulary.json` and `audit_report.json` must be JSON mappings with
`protocol_version == "PUB-R1.1"`. The vocabulary is reconstructed through the
existing immutable `Vocabulary` validator, which enforces:

- `<NULL>=0`, `<START>=1`, `<END>=2`, `<UNK>=3`;
- `token_to_id` exactly inverts `id_to_token`;
- no duplicate token;
- lexicographically sorted ordinary tokens.

No new tokenization or vocabulary implementation is added.

### 5.2 Selected-question rows

Every non-empty JSONL line must decode to a mapping with exactly the fields
emitted by the accepted audit command:

```text
partition, official_split, image_index, image_filename, image_path,
question_index, source_position, question, encoded_tokens, answer, relation
```

Extra or missing fields are a hard error. Types use strict JSON/Python checks;
in particular, booleans are not accepted as integer IDs or labels. Validate:

- partition is in the four-value set;
- `train`/`dev` rows use official split `train`, while
  `base_test`/`confirmation` rows use `val`;
- image, question, and source indices are non-negative integers;
- `ImageIdentity` accepts the official split/index/filename tuple;
- `image_path` is exactly `images/<official_split>/<image_filename>` as a
  relative POSIX path;
- resolving that path under the resolved `data_root` stays under the root and
  identifies an existing regular file;
- question is a string, relation is one of left/right/front/behind, and answer
  is the integer 0 or 1;
- encoded tokens are a JSON list of ordinary integers in vocabulary range,
  contain at least START and END, begin with 1, end with 2, and contain no 0
  before collation;
- recomputing `encode_question(question, vocabulary)` exactly equals the stored
  token sequence;
- `(official_split, question_index)` and
  `(official_split, source_position)` are unique;
- every partition's observed row count equals its
  `audit_report.json` `selected_question_count`, and each requested partition
  is non-empty.

Malformed JSON, blank lines, duplicate summary rows, missing artifacts, or an
unknown summary partition are hard errors. Errors identify the artifact and
line or field; they do not skip a row or repair it.

### 5.3 Ordering

After validating all rows, retain only the requested partition and sort by

\[
(\text{image\_index},\ \text{question\_index},\ \text{source\_position}).
\]

This is the protocol's base order. The later sampler may permute dataset
indices, but the dataset itself is deterministic and independent of JSONL
physical line order.

## 6. Image transform

For selected row \(i\), let the decoded PNG after Pillow `convert("RGB")` be

\[
I_i\in\{0,\ldots,255\}^{H_i\times W_i\times3}.
\]

Apply the operations in this exact order:

1. `torchvision.transforms.functional.pil_to_tensor`, producing a copied
   `uint8` CHW tensor;
2. convert to `torch.float32` and divide by 255;
3. `torchvision.transforms.functional.resize` to `[224, 224]` with
   `InterpolationMode.BILINEAR` and `antialias=True`.

The result is

\[
X_i\in\mathbb R^{3\times224\times224}.
\]

No mean/std normalization is applied. The implementation checks the final
shape, dtype, finiteness, and range `[0,1]`. It opens only PNG through Pillow,
closes the file before returning, and converts decode/I/O failures into a
dataset-contract error that names the selected index/path. Constructor path
checks do not eagerly decode all images.

## 7. Sample and batch contract

For a question token sequence

\[
q_i=(q_{i,0},\ldots,q_{i,L_i-1})\in\mathbb N^{L_i},\quad
q_{i,0}=1,\quad q_{i,L_i-1}=2,
\]

and answer \(y_i\in\{0,1\}\), `__getitem__` returns a mapping with exactly:

| Key | dtype | shape | Meaning |
| --- | --- | --- | --- |
| `image` | `torch.float32` | `[3,224,224]` | frozen RGB transform |
| `question_tokens` | `torch.int64` | `[L_i]` | START…END, unpadded |
| `question_length` | `torch.int64` | `[]` | scalar `L_i` |
| `answer` | `torch.int64` | `[]` | no=0 / yes=1 |

For non-empty samples \(i=1,\ldots,N\), the collate function sets
\(L_{\max}=\max_i L_i\) and constructs

\[
Q_{ij}=\begin{cases}
q_{i,j},&j<L_i,\\
0,&j\ge L_i.
\end{cases}
\]

It returns the same four keys with shapes `[N,3,224,224]`, `[N,L_max]`,
`[N]`, and `[N]`. It never truncates or sorts the supplied batch. An empty
sample sequence, wrong/extra key, wrong dtype/shape, inconsistent declared
length, or invalid answer is a hard error.

Hand example:

```text
q1 = [1, 7, 2],       L1 = 3
q2 = [1, 5, 6, 2],    L2 = 4
Q  = [[1, 7, 2, 0],
      [1, 5, 6, 2]]
L  = [3, 4]
```

For a constant RGBA fixture whose RGB bytes are `(255,128,0)`, dropping alpha,
scaling, and resizing must keep every output pixel equal to
`(1.0, 128/255, 0.0)` up to the library's exact float32 representation.

## 8. Invariants, boundaries, and falsification

The implementation is invalid if any focused test demonstrates one of these:

- an absolute, traversing, or root-escaping image path is accepted;
- a model-facing sample/batch contains a fifth field or any audit-only value;
- JSONL line order changes dataset order;
- stored tokens disagree with the original question and frozen vocabulary but
  are accepted;
- a sequence is truncated, padded with a nonzero ID, or its true length points
  after padding rather than END;
- image decoding returns RGBA, integer data, a size other than 224×224, or a
  transform differing from the explicit torchvision functional reference;
- a malformed row is skipped, coerced, or repaired;
- construction reads scene/program files or invokes data/network discovery;
- any model parameter, G/B phase, or training behavior changes in this PR.

Boundary conditions include one-row partitions in fixtures, unequal question
lengths, duplicate IDs, out-of-range token IDs, blank/malformed JSONL,
non-PNG/unreadable files, symlink escape, missing count summaries, and multiple
questions sharing one image. Shared images are decoded independently on access;
no mutable image cache is introduced.

## 9. Mature-library and license decision

The implementation uses only the installed project stack:

| Component | Local version | License evidence | Adopted API |
| --- | --- | --- | --- |
| PyTorch | 2.6.0+cu124 | installed metadata: BSD-3-Clause | map-style `Dataset`, tensors, `pad_sequence` or equivalent standard collation |
| torchvision | 0.21.0+cu124 | installed metadata/license: BSD | `pil_to_tensor`, tensor `resize`, `InterpolationMode.BILINEAR` |
| Pillow | 12.0.0 | installed metadata: MIT-CMU | PNG `Image.open`, `convert("RGB")` |
| Python standard library | project interpreter | PSF ecosystem | `json`, `pathlib`, immutable records |

Official PyTorch 2.6 documentation defines a map-style dataset through
`__getitem__` and `__len__`, with sampler-controlled index order:
<https://docs.pytorch.org/docs/2.6/data.html>. Official torchvision 0.21
documentation states that `pil_to_tensor` copies a PIL image without scaling
and that tensor `resize` accepts exact `[H,W]`, bilinear interpolation, and
`antialias=True`:
<https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.functional.pil_to_tensor.html>
and
<https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.functional.resize.html>.

No torchvision CLEVR dataset implements this filtered question/manifest
contract. A thin map-style adapter plus explicit functional transform is the
smallest gap. A configurable transform pipeline is deliberately not exposed,
because caller-provided transforms could silently violate the frozen input.

## 10. Cost and degradation behavior

The unit adds no trainable parameters and no GPU computation. Constructor cost
is linear in the 9,506-row manifest and retains only compact immutable runtime
records. Item cost is one local PNG decode and one CPU tensor resize. Batch
memory is dominated by `N×3×224×224` float32 images plus `N×L_max` int64 tokens.

There is no fallback mode. Any failed artifact, path, image, token, or count
check stops construction/access with a descriptive error. If the official
artifacts intentionally change, they require a new protocol/versioned audit,
not runtime coercion.

## 11. Test and PR boundary

The implementation PR will add focused tests using temporary JSON/JSONL and
small generated PNG fixtures. Tests cover schema rejection, root containment,
sorting, exact return keys/dtypes/shapes, image transform equality, dynamic
padding, shared-image rows, and error propagation. They also run the existing
CPU-only nonhash regression suite. The legacy hash-executing CIFAR tests remain
excluded under project policy.

The PR contains this dataset module and its tests only. It does not add the VQA
backbone, G/B comparison code, M10, Q-only, sampler, runner, or real-data run.
After this PR is reviewable and merged, the next independent unit is the common
RGB+question backbone and G/B integration specified by PUB-R1.1.
