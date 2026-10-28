# CLEVR PUB-R1.1 Runner Integration Design

Date: 2026-10-25
Protocol: PUB-R1.1
Status: implementation basis; the user instructed the agent to continue the
accepted plan autonomously and to defer blocked PR publication

## 1. Intent

The accepted PUB-R1.1 components currently stop at separate dataset, model,
training-configuration, committed-sampler, checkpoint, and statistics APIs.
This unit connects them into an explicit command-line runner without starting
an experiment. It must preserve the protocol's paired sample order, exact
gradient accumulation, update-indexed learning rate, checkpoint boundary, and
per-question evidence while keeping audit identities outside the model input.

This unit is an integration boundary, not a new model or scientific mechanism.
It does not change G, B, M10, Q-only, their losses, their budget, or the frozen
statistics. No command is executed against user data or a GPU as part of this
implementation.

## 2. Scope and permission boundary

The runner supports three separately invoked operations:

1. `train`: a caller-authorized pilot or formal training run, including the
   protocol-required dev evaluations and final train evaluation;
2. `evaluate`: an explicitly requested evaluation of one checkpoint, with
   base-test prediction publication available only for formal seeds and the
   fixed update-8000 checkpoint;
3. `inspect`: configuration and artifact preflight only, with no dataset item
   access, model construction, CUDA initialization, or training.

The command never downloads, discovers, or substitutes data or weights. Paths
must be explicit. Merely implementing or testing this module does not authorize
`train` or `evaluate`; those commands remain user-controlled under AGENTS.md.
The runner does not poll or monitor a launched process.

Confirmation-set model evaluation is deliberately absent. The confirmation
partition remains reserved for a future separately preregistered candidate.

## 3. Integration approach

The implementation branch is a local integration top over the already
reviewed feature heads for question data, all four models/training config,
sampler/checkpoint, and reporting/statistics. Those histories remain distinct;
the runner adds only its own dataset provenance accessor, integration module,
entry point, tests, and design/plan. If prerequisite PRs merge first, this
branch can be rebased so its final PR contains only the runner delta.

Alternatives rejected:

- Waiting for every prerequisite PR would leave the only missing protocol unit
  unimplemented despite the user's instruction to continue past PR blockers.
- Reimplementing narrow copies of unmerged APIs would create two contracts and
  make later integration evidence meaningless.
- Passing question/image IDs through dataset batches would violate the frozen
  model-input boundary and create an avoidable leakage surface.

## 4. Public data provenance without model leakage

Both runtime datasets retain validated records in protocol order. Add an
immutable public record:

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

and these read-only members to both RGB and Q-only datasets:

```python
@property
def vocab_size(self) -> int: ...

def sample_identity(self, index: int) -> CLEVRSampleIdentity: ...
```

For dataset index `j`, `sample_identity(j)` returns metadata from the same
validated `_RuntimeRecord` as `__getitem__(j)`. It performs no image read and
returns no mutable internal object. Python sequence indexing semantics are
used consistently by both methods, including valid negative indices and
`IndexError` outside the range.

The invariant is structural:

\[
\operatorname{keys}(D_{RGB}[j])=
\{\texttt{image,question_tokens,question_length,answer}\},
\]

\[
\operatorname{keys}(D_Q[j])=
\{\texttt{question_tokens,question_length,answer}\}.
\]

No identity field enters either sample, collated batch, model call, loss, or
gradient. The runner holds sampler indices separately and queries identities
only when writing sample-order or prediction evidence.

## 5. Frozen run configurations

The evidence-producing configurations are exact:

| phase | allowed seed | target optimizer updates | role |
| --- | --- | ---: | --- |
| smoke | 9001 | 1 | wiring check only; non-evidentiary |
| pilot | 9001 | 800 | engineering/numerical check only |
| formal | 101,102,103,104,105 | 8000 | frozen comparison |

Methods are exactly `G`, `B`, `M10`, and `Q-only`. A seed-level training
invocation constructs one `CLEVRPairedInitialization`, then builds requested
arms sequentially so one owned common non-position state is the source for all
fresh arms in that process. Resuming reconstructs the same factory and then
loads the explicit checkpoint transactionally.

The runner exposes no arbitrary result-producing update budget. The one-update
`smoke` mode uses seed 9001 and is marked
`evidence_role="engineering-only"`; it cannot publish base-test predictions or
enter the statistics command.

The seed directory owns one `common_state.pt`. Its restricted payload contains
only protocol version, run seed, locked environment strings, and cloned CPU
tensors from `CLEVRPairedInitialization.common_state_dict()`. The first fresh
arm publishes it atomically without overwrite. Every later fresh or resumed arm
loads it with `weights_only=True` and requires exact ordered keys, shape, dtype,
and `torch.equal` values before training. This makes the common source a direct
artifact rather than relying only on deterministic reconstruction.

## 6. Training mathematics and state transition

Let effective batch size `B=128`, micro-batch size `b=16`, and accumulation
count `A=8`, so `B=Ab`. At completed-update boundary `k`, the committed state is

\[
S_k=(\theta_k,O_k,L_k,I_k,R_k),
\]

matching the checkpoint contract. The sampler first previews

\[
J_k=I_k.\operatorname{preview}(128)
=(j_{k,0},\ldots,j_{k,127})
\]

without mutation. For micro-batch `a`, collate exactly the contiguous indices
`J_k[16a:16(a+1)]`. If `z_{a,n}\in\mathbb R^2` are logits and `y_{a,n}` labels,
the backward scalar is

\[
\ell_a={1\over 8}\left[-{1\over16}
\sum_{n=1}^{16}\log
{\exp z_{a,n,y_{a,n}}\over\sum_{c=0}^1\exp z_{a,n,c}}\right].
\]

After all eight backward calls, clip the global parameter-gradient norm to
`1.0` using PyTorch's maintained `clip_grad_norm_` with non-finite values as a
hard error. Then execute, in order:

```text
optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)
sampler.commit(J_k)
```

The transition is complete only after commit, yielding `S_(k+1)`. The learning
rate used by `optimizer.step()` is already the frozen `eta_(k+1)` supplied by
the existing scheduler convention. Q-only receives `image=None`; the other
three methods receive the RGB tensor. No audit metadata is passed.

If forward, loss, backward, clipping, or optimizer/scheduler step raises,
indices are not committed. The process is considered failed and must resume
from a previously published complete checkpoint; in-memory optimizer changes
after a failing optimizer call are not claimed rollback-safe. Every invocation
therefore records its attempt separately and preserves failed attempts.

### Hand-worked boundary example

For a dataset of size five and a preview beginning `(1,4,2,0,3,...)`, the
effective batch is the next 128 entries of the concatenated epoch permutation
stream. The first micro-batch consumes entries 0--15 only in computation; the
sampler cursor remains unchanged throughout all eight micro-batches. If the
eighth backward fails, `state_dict()` still reports the old cursor. Only after
the successful update sequence is the exact 128-entry tuple committed. Thus
DataLoader look-ahead cannot become false consumed progress because the runner
does not use a prefetching DataLoader.

## 7. Evaluation and prediction contract

Evaluation uses deterministic ascending dataset indices, micro-batches of at
most 16, `model.eval()`, and `torch.inference_mode()`. It computes sums over
questions, never unweighted means of batch means:

\[
\operatorname{accuracy}={\sum_i [\hat y_i=y_i]\over N},\qquad
\operatorname{crossentropy}={\sum_i \operatorname{CE}(z_i,y_i)\over N},
\]

with tie prediction `argmax([z_0,z_1])=0`, equivalent to `int(z_1>z_0)`.
The per-question output row has exactly:

```text
protocol_version, method, seed, question_index, label, logits
```

Rows are sorted by ordinary integer `question_index`; logits are finite Python
floats copied from FP32 model output. Before publication, every dataset index
must contribute exactly once, identities must be unique, and labels must equal
the model batch labels. Formal base-test output uses the canonical filename
`<method>_seed_<seed>.jsonl`, directly consumable by the existing independent
reporter.

Training automatically evaluates dev after update 1000 and each subsequent
1000 updates. Pilot evaluates dev at its final update 800. Formal training also
evaluates the complete train partition after update 8000. No training command
constructs the base-test dataset.

## 8. Checkpoints, attempts, and append-only evidence

Fresh runs save update 0 and then every 1000 complete updates; pilot also saves
its final update 800. Formal update 8000 is both periodic and final. Each path
is new-file-only and uses the reviewed checkpoint publisher.

A seed directory contains immutable `common_state.pt`; each method run
directory contains one immutable `run_config.json`, append-only
per-attempt journals under `attempts/`, new checkpoint files, and new
prediction files. Every process invocation claims a new monotonically
increasing `attempt_N.jsonl` with exclusive new-file creation; concurrent
callers cannot share one journal. Records include method, seed, phase, attempt,
completed update, event type, and protocol-relevant values. Update records
contain the 128 dataset indices and matching 128 question indices, the
summed/mean loss, unclipped gradient norm, and LR used. This is direct
sample-order evidence, not a digest.

Resume always requires an explicit checkpoint path. The immutable config,
checkpoint metadata, dataset size, method, seed, and requested target are
validated before any mutation. Existing records are never deleted or rewritten;
a resume event identifies the checkpoint boundary from which a new attempt
continues. A successful run publishes a new completion record only after its
required final checkpoint and evaluations exist. Failures append their type and
message when possible, then re-raise; a logging failure cannot convert a failed
training action into success.

## 9. Runtime and device contract

Production `train` and `evaluate` require one explicit CUDA device index. They
do not auto-select a device. Inputs and model are FP32; AMP is absent and CUDA
and cuDNN TF32 are disabled before computation. The expanded run config records
Python, NumPy, PyTorch, torchvision, timm, Pillow, CUDA runtime, device name,
deterministic-algorithm flags, cuDNN flags, and exact protocol constants.

The runner seeds Python, NumPy legacy RNG, PyTorch CPU, and the selected CUDA
device from the run seed before each fresh arm. NumPy's seed API is valid for
all frozen seeds. Resume instead restores all recorded streams from the
checkpoint after object construction. The runner neither promises cross-device
bitwise identity nor changes the frozen sampler's independent `10000+s` stream
or M10's `20000+s` stream.

Unit tests exercise CPU seams through injected tiny models/datasets and mocks;
they run with `CUDA_VISIBLE_DEVICES=""`, are not experiment runs, and do not
weaken the production CUDA check.

## 10. Mature-library and license review

Local locked-stack evidence from the prerequisite units is reused:

| Need | Candidate | Decision |
| --- | --- | --- |
| CE loss | PyTorch 2.6 `torch.nn.functional.cross_entropy` | reuse with `reduction="mean"`; divide each fixed-size micro-batch by 8 |
| clipping | PyTorch 2.6 `torch.nn.utils.clip_grad_norm_` | reuse with `max_norm=1.0`, `error_if_nonfinite=True` |
| inference | PyTorch `inference_mode` and module `eval/train` | reuse; no custom autograd suppression |
| batch loading | PyTorch `DataLoader` versus direct map-style indexing | direct indexing selected because the committed sampler already owns order and DataLoader prefetch adds an unnecessary second cursor |
| JSON records | Python 3.11 `json` and filesystem primitives | reuse with strict schemas and new-file outputs; no logging dependency |
| model/data/optimizer/checkpoint | reviewed repository APIs over PyTorch, torchvision, timm, Pillow, NumPy | compose directly; do not rewrite their core behavior |

The runner-specific gap is orchestration and artifact state, for which no
general library knows PUB-R1.1's method matrix, committed-index semantics, or
authorization split. No dependency or external source is added.

## 11. Invariants and falsification

The implementation is invalid if any test or inspection shows:

1. an identity/path/relation/program field enters a model-facing sample or
   call;
2. paired methods receive different dataset-index or question-index streams;
3. sampler state advances before all eight micro-batches and the update finish;
4. loss scaling differs from eight means divided by eight, clipping differs
   from global norm 1.0, or scheduler order shifts the frozen LR;
5. a checkpoint is written with gradients present or a non-committed cursor;
6. Q-only opens an image, receives an image tensor, or gains a patch-projection
   gradient;
7. a training command touches base_test/confirmation or an evaluation command
   publishes a partial/duplicate prediction set;
8. resume silently restarts an epoch, auto-selects a checkpoint, rewrites old
   evidence, or accepts a mismatched method/seed/environment;
9. a pilot/smoke file can be consumed as a formal base-test prediction; or
10. implementation tests require user data, downloads, weights, CUDA, or a
    long-running training process.

## 12. Implementation boundary

The runner PR may add or modify only:

- this design and its implementation plan;
- the provenance accessor in `src/clevr_dataset.py` and focused tests;
- `src/clevr_run_artifacts.py` and `tests/test_clevr_run_artifacts.py`, including
  restricted common-state publication/verification;
- `src/clevr_runner.py` and `tests/test_clevr_runner.py`;
- `run_clevr_experiment.py` and focused CLI tests.

It may merge prerequisite feature histories but does not change their model,
sampler, checkpoint, or statistics formulas. Actual data validation, smoke,
pilot, formal training, dev/base-test evaluation, report generation, GPU use,
push, PR creation, and tagging remain outside implementation-time execution
unless separately authorized.
