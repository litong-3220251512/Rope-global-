# CLEVR Q-only Runtime Data Design

Date: 2026-10-25
Protocol: PUB-R1.1
Status: implementation basis; the user has instructed the agent to continue the
accepted plan without another design-question round

## 1. Intent

PUB-R1.1 defines Q-only as a question-prior control that must not call the RGB
loader or read scene data. The existing `CLEVRQuestionDataset` deliberately
implements the stronger RGB contract: it requires an explicit CLEVR root,
resolves every manifest image to an existing in-root regular file, and decodes
the selected PNG lazily. Reusing that public class for Q-only would therefore
retain an unnecessary image-filesystem dependency even if the caller discarded
the returned tensor.

This unit adds a distinct runtime dataset and collate boundary for Q-only. It
uses the same accepted audit artifacts, validates the same question identities,
text, labels, counts, and deterministic order, but exposes only encoded question
tokens, true token length, and supervised answer. It does not implement the
Q-only model, optimizer, runner, checkpointing, or any experiment run.

## 2. Goals, non-goals, and anti-goals

### Goals

- Construct Q-only samples from only a caller-specified `audit_dir` and one
  frozen partition name.
- Validate the complete manifest schema, including image identity and the exact
  canonical relative `image_path` string, without resolving or probing any
  image file.
- Preserve the RGB dataset's base order and question/answer values exactly.
- Return exactly `question_tokens`, `question_length`, and `answer` from each
  sample and batch.
- Dynamically right-pad with `<NULL>=0` without reordering or truncating.
- Preserve the existing RGB dataset's public API, image checks, sample fields,
  and collate behavior.

### Non-goals

- No filtering, partition construction, vocabulary construction, answer
  recomputation, or scene/program loading.
- No changes to model behavior. The already separate Q-only model unit owns the
  replacement of image patch embeddings with zero patch tokens.
- No sampler, optimizer, training runner, checkpoint, metric, or train-majority
  implementation.
- No real CLEVR access, engineering smoke, training, or evaluation. Tests use
  temporary artifacts and synthetic PNG fixtures only where the unchanged RGB
  regression needs them.

### Anti-goals

- Do not add a public `q_only=True` flag that weakens
  `CLEVRQuestionDataset`'s required `data_root` or file checks.
- Do not open, decode, stat, resolve, or require an image path on the Q-only
  path; do not accept a `data_root` there.
- Do not read a scene, program, relation table, object coordinate, or other
  offline-only value as a model feature.
- Do not return image metadata, question text, relation, IDs, partition, or
  paths in a model sample or batch.
- Do not add a dependency, download, hash check, fallback search, or generated
  data.

## 3. Approaches considered

### A. Dedicated public Q-only dataset with shared private validation — selected

Add `CLEVRQuestionOnlyDataset(audit_dir=..., partition=...)` and
`collate_clevr_question_only(...)`. Refactor only the private row-loading path
so both datasets share schema, text, answer, uniqueness, count, and order
checks. The image path string is always checked against the validated
`ImageIdentity`; filesystem resolution happens only when the RGB caller passes
its explicit root.

This gives the strongest observable no-RGB contract: Q-only has no root to
probe and still rejects corrupt audit metadata. It also makes accidental image
fields at the model boundary a strict error.

### B. Add a `q_only` mode to `CLEVRQuestionDataset` — rejected

This would overload whether `data_root` is required, whether missing images are
valid, and which keys `__getitem__` returns. A mode-dependent public contract is
easy to misconfigure in a later runner and weakens the already reviewed RGB
boundary.

### C. Feed blank images through the RGB dataset — rejected

This still opens and transforms every PNG, violates PUB-R1.1's no-RGB-I/O
control, and confounds question-only compute with the visual input pipeline.
It also fails when the image tree is absent even though Q-only needs only the
accepted audit artifacts.

### D. Duplicate the entire parser for Q-only — rejected

Duplication would let question, label, count, or ordering checks drift between
the RGB and Q-only arms. The difference belongs only at image-file validation
and the model-facing field boundary.

## 4. Public API and data flow

The new public surface in `src/clevr_dataset.py` is:

```python
class CLEVRQuestionOnlyDataset(
    torch.utils.data.Dataset[dict[str, torch.Tensor]]
):
    def __init__(self, *, audit_dir: Path, partition: str) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: ...


def collate_clevr_question_only(
    samples: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]: ...
```

The constructor reads exactly these existing artifacts:

- `<audit_dir>/selected_questions.jsonl`
- `<audit_dir>/vocabulary.json`
- `<audit_dir>/audit_report.json`

It does not receive or infer a CLEVR data root. It validates all rows in the
selected-question artifact, checks all four report counts, retains only the
requested non-empty partition, and sorts records by
`(image_index, question_index, source_position)` exactly as the RGB dataset
does. The retained Q-only record needs only those sort fields, encoded token
IDs, and answer; it retains no usable image path.

The existing public RGB API remains:

```python
CLEVRQuestionDataset(data_root=..., audit_dir=..., partition=...)
collate_clevr_questions(samples)
```

and continues to require every referenced image to resolve to an existing
regular file under the explicit root before retaining resolved paths.

## 5. Mathematical contract

Let an accepted manifest partition contain rows

\[
r_i=(m_i, q_i, s_i, f_i, u_i, y_i),
\]

where `m` is image index, `q` question index, `s` source position, `f` the
canonical image identity/path metadata, `u_i=(u_{i1},\ldots,u_{i\ell_i})` the
audited token sequence, and \(y_i\in\{0,1\}\). Each `u_i` begins with START=1,
ends with END=2, contains no NULL=0, and exactly equals re-encoding the original
question under the accepted training vocabulary.

Define \(\sigma\) as ascending lexicographic order on `(m, q, s)`. The Q-only
dataset is

\[
D_Q(j)=(u_{\sigma(j)},\ \ell_{\sigma(j)},\ y_{\sigma(j)}).
\]

For a batch of \(N>0\) samples, let
\(L=\max_j\ell_{\sigma(j)}\). The collate output is

\[
T_{jk}=\begin{cases}
u_{\sigma(j),k},& k<\ell_{\sigma(j)},\\
0,& \ell_{\sigma(j)}\le k<L,
\end{cases}
\quad
\boldsymbol{\ell}_j=\ell_{\sigma(j)},
\quad
\mathbf y_j=y_{\sigma(j)}.
\]

Shapes and dtypes are:

- `question_tokens`: \(T\in\mathbb Z^{N\times L}\), `torch.int64`;
- `question_length`: \(\boldsymbol{\ell}\in\mathbb Z^N\), `torch.int64`;
- `answer`: \(\mathbf y\in\{0,1\}^N\), `torch.int64`.

All sample tensors are CPU tensors at this boundary. Device transfer belongs to
the later runner.

### Relation to the G/RGB data baseline

When the same audit artifacts are used and all RGB files satisfy the existing
RGB contract, for every index \(j\),

\[
\pi_{u,\ell,y}(D_{RGB}(j)) = D_Q(j),
\]

where \(\pi\) drops only the RGB tensor. The records, order, question encoding,
length, and label must remain equal. Q-only additionally remains defined when
the entire image tree is absent. This data-layer equality does not claim that G
and Q-only model activations are equal: the Q-only model separately substitutes
zero patch tokens.

## 6. Invariants, boundaries, and falsification

The implementation must preserve these invariants:

1. **Artifact integrity:** both modes reject malformed protocol, vocabulary,
   row schema, identity, canonical image-path string, text encoding, answer,
   uniqueness, partition isolation, counts, and empty selected partitions.
2. **RGB preservation:** the RGB class still rejects missing/out-of-root/non-file
   paths at construction and rechecks the selected file at access; image output
   and collate validation are unchanged.
3. **Q-only independence:** Q-only construction, access, and collation succeed
   after the image tree is removed, and a patched `PIL.Image.open` that raises
   is never reached.
4. **Exact boundary:** Q-only samples/batches contain exactly three keys; image
   or metadata keys are rejected rather than silently discarded.
5. **Projection equality:** with valid images, Q-only and RGB datasets have
   equal length and identical token/length/answer tensors at every tested index.
6. **No label/order change:** the shared sort and encoded values are direct
   audit values, not reconstructed labels or answer-balanced selections.

The design is falsified by any test or inspection showing image/scene access on
the Q-only path, acceptance of noncanonical manifest image metadata, divergence
from RGB question order/labels, an image or audit-only field in a Q-only batch,
or changed RGB behavior.

## 7. Hand-worked example

Suppose physical JSONL order contains these two train rows:

```text
(image_index=2, question_index=20, source_position=20,
 tokens=[1,5,6,2], answer=1)
(image_index=1, question_index=10, source_position=10,
 tokens=[1,4,2], answer=0)
```

Sorting yields image 1 then image 2. Therefore Q-only item 0 is
`([1,4,2], 3, 0)` and item 1 is `([1,5,6,2], 4, 1)`. Collation must produce:

```text
question_tokens = [[1,4,2,0], [1,5,6,2]]
question_length = [3,4]
answer          = [0,1]
```

No image tensor or path appears. Removing `images/` cannot change this output.

## 8. Mature-library and license review

Local environment inspection on 2026-10-25 found:

| Need | Candidate | Local version/license evidence | Decision |
| --- | --- | --- | --- |
| Map-style dataset | `torch.utils.data.Dataset` | PyTorch `2.6.0+cu124`; installed metadata says BSD-3-Clause | Reuse; no custom loader framework |
| Variable-length padding | `torch.nn.utils.rnn.pad_sequence` | PyTorch `2.6.0+cu124`; signature supports `batch_first` and `padding_value` | Reuse with right padding and NULL=0 |
| RGB decode/resize | Existing Pillow/torchvision path | torchvision `0.21.0+cu124` metadata says BSD; Pillow `12.0.0` declares MIT-CMU | Preserve for RGB; do not invoke for Q-only |
| JSON/schema validation | Python `json` plus existing `Vocabulary`, `ImageIdentity`, and `encode_question` | Repository-owned protocol code | Reuse; no general schema package needed |

No maintained library can supply this repository-specific accepted-manifest
contract or its exact field whitelist. The minimum custom code is therefore one
thin dataset, one collator, and a private parser parameterization. No dependency
or external source is added or copied.

## 9. Tests and implementation boundary

Add focused tests in `tests/test_clevr_dataset.py` before product code:

- Q-only constructs and returns the exact three tensors when `data_root` and
  all PNGs are absent, with `PIL.Image.open` patched to fail if called.
- Q-only and RGB records have exact token/length/answer projection equality and
  order when PNGs exist.
- Q-only still rejects malformed canonical image metadata even though it does
  not resolve files.
- Q-only collate pads and preserves order, exposes exactly three keys, and
  rejects empty input, extra/missing fields, wrong tensor types/dtypes/shapes,
  inconsistent lengths, NULL within a sample, and invalid answers.
- `DataLoader(..., shuffle=False, collate_fn=collate_clevr_question_only)`
  yields the exact expected batch without image fields.
- Existing RGB tests and the full CPU suite remain green.

Implementation is limited to this design document, its implementation plan,
`src/clevr_dataset.py`, and `tests/test_clevr_dataset.py`. It adds no training,
GPU, network, real-data, or performance claim. A later runner PR will join this
three-field batch with the Q-only model by passing `image=None` explicitly.
