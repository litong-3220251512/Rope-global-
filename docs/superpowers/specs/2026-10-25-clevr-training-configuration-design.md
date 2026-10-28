# CLEVR Paired Initialization and Optimizer Configuration Design

Date: 2026-10-25
Protocol: PUB-R1.1
Status: implementation basis; the user instructed the agent to continue the accepted plan without another design-question round

## 1. Intent

PUB-R1.1 requires G, B, M10, and Q-only runs for one training seed to start
from one bitwise-equal non-position model state. It also freezes AdamW weight
decay membership and the learning rate used by each of 8,000 optimizer updates.
The existing answerer implements all four modes and tests same-seed constructor
behavior, but no runtime object yet owns the one-time common state, copies it
into independently constructed arms, records optimizer membership, or exposes
the update-indexed schedule.

This unit adds that configuration boundary. It constructs models and optimizer
objects on CPU for unit verification, but it does not perform a training run,
read data, move a model to a GPU, execute a checkpoint file, or decide when a
runner saves and resumes.

## 2. Goals, non-goals, and anti-goals

### Goals

- Generate exactly one common non-position state for `(vocab_size, run_seed)`.
- Build G, B, M10, or Q-only on demand and copy that state into the arm after
  checking exact key order, shape, and dtype.
- Give M10 only the independent position seed `20000 + run_seed`.
- Keep construction from changing process-global Python, NumPy, PyTorch CPU,
  or accelerator RNG streams.
- Return fresh common-state snapshots whose tensors do not alias the factory or
  any built model.
- Build AdamW groups that cover every trainable parameter exactly once and
  retain the parameter names in the optimizer state for direct audit.
- Put protocol-listed matrices in weight decay 0.05 and every bias,
  LayerNorm parameter, CLS token, LayerScale parameter, and M10 frequency in
  weight decay 0.
- Expose the exact finite-budget learning-rate formula and a standard PyTorch
  scheduler whose current LR is the LR for the next update.

### Non-goals

- No dataset, DataLoader, sampler, batch transfer, loss, backward pass,
  accumulation, clipping, optimizer update loop, evaluation, or metrics.
- No model/optimizer/scheduler checkpoint file and no restore transaction. A
  later runner unit will combine those states with committed sampler and RNG
  state at complete update boundaries.
- No formal-seed allowlist. The later run-config validator owns pilot/formal
  matrices. This unit accepts ordinary Python integers in
  `0 <= run_seed <= 2**64 - 1 - 20000`, so both the common seed and exact M10
  seed `20000 + run_seed` stay inside PyTorch's accepted unsigned 64-bit range.
  It does not wrap or reduce a seed modulo that range.
- No claim that the frozen optimizer or schedule is optimal or converged.

### Anti-goals

- Do not create all four models at once in production merely to compare them;
  one factory must support one arm at a time.
- Do not rely only on reseeding four constructors as evidence of copying one
  common state.
- Do not compare weights with hashes, summaries, norms, or tolerances.
- Do not use parameter dimensionality without recording the resulting names and
  verifying complete, disjoint coverage.
- Do not give Q-only's unused patch projection a synthetic zero gradient. It
  remains in the declared decay group but AdamW must skip it when `grad is None`.
- Do not implement a hand-written optimizer when PyTorch AdamW is available.

## 3. Approaches considered

### A. One common-state factory with on-demand arms — selected

Construct one G answerer inside an isolated CPU RNG context, clone every state
entry except `position.*`, and retain that private mapping. `build(mode)`
constructs one requested answerer in another isolated context, checks its
non-position schema, loads the private clones into a complete state mapping,
then verifies every copied tensor with `torch.equal`.

This implements “generate once, copy to all groups” literally while keeping
peak production memory to one model plus one common-state snapshot. M10's
position module remains outside the copy and uses its already-isolated seed.

### B. Construct and return all four arms together — rejected

This makes cross-arm comparison convenient but keeps four full transformers in
memory even though the single-GPU protocol trains one run at a time. Tests may
temporarily build multiple arms, but the production API should not require it.

### C. Reset the seed before every constructor and compare afterward — rejected

Existing model tests already show that this currently produces equal common
tensors. It does not make one owned initial state available for saving, and a
future mode-specific constructor change could consume a different random
sequence before a common module while still looking superficially paired.

### Optimizer and schedule choices

Use timm's maintained `param_groups_weight_decay` for the conventional
one-dimensional/bias split, with explicit skip names for CLS and all position
parameters. Reconstruct named groups from its parameter identities, verify
coverage against `named_parameters()`, and store `group_name` plus
`param_names` in each AdamW group. A fully custom classifier was rejected
because the fixed model's only trainable one-dimensional tensors are exactly
biases, LayerNorm, and LayerScale.

Use PyTorch `LambdaLR` around one pure protocol formula. `SequentialLR` with
`LinearLR` and `CosineAnnealingLR` was rejected because their inclusive
milestones make the required update-1/update-400/update-401 convention less
direct to audit. A custom scheduler was also rejected because `LambdaLR`
already serializes the required standard scheduler counters and group state.

## 4. Public API

Add `src/clevr_training.py` with:

```python
class CLEVRTrainingConfigurationError(ValueError): ...


class CLEVRPairedInitialization:
    def __init__(self, *, vocab_size: int, run_seed: int) -> None: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def run_seed(self) -> int: ...

    @property
    def m10_seed(self) -> int: ...

    def common_state_dict(self) -> OrderedDict[str, Tensor]: ...

    def build(self, mode: str) -> CLEVRQuestionAnswerer: ...


def build_clevr_adamw(model: CLEVRQuestionAnswerer) -> AdamW: ...


def clevr_learning_rate(update: int) -> float: ...


def build_clevr_lr_scheduler(optimizer: AdamW) -> LambdaLR: ...
```

`vocab_size` is an ordinary Python integer at least 4. `run_seed` is an
ordinary Python integer in `[0, 2**64 - 1 - 20000]`; booleans and NumPy integer
scalars are invalid. The upper bound is required because local verification of
PyTorch 2.6 found that `Generator.manual_seed(2**64 - 1)` succeeds and
`manual_seed(2**64)` raises overflow, while M10 must use the unreduced sum
`20000 + run_seed`. `mode` is exactly one of `G`, `B`, `M10`, and `Q-only`.

The implementation decision for the common PyTorch stream is to use
`run_seed` directly. PUB-R1.1 names the experiment seed and separately freezes
only the sampler stream as `10000+s` and M10 as `20000+s`; it gives no other
common-initialization offset. This decision is part of this design rather than
a claim from an external paper.

## 5. Common initialization contract

Let `State(M)` be the ordered `state_dict()` of answerer `M`. Define the common
key sequence

\[
K_{common}=(k\in State(G): \neg k.\mathrm{startswith}(\texttt{"position."})).
\]

Inside `torch.random.fork_rng(devices=[])`, set only the CPU default generator
to `run_seed`, construct a G model, and store

\[
C_s[k]=\operatorname{clone}(State(G_s)[k]),\quad k\in K_{common}.
\]

For requested arm `a`, construct `M_{a,s}` in a fresh isolated CPU RNG context;
pass `m10_seed=20000+s` only when `a=M10`. Before copying, require its common
keys to equal `K_common` in order, and require each target tensor's shape and
dtype to match. Load a complete state mapping in which common entries are
replaced by fresh clones of `C_s`, then require

\[
\forall k\in K_{common},\quad
\operatorname{torch.equal}(State(M_{a,s})[k],C_s[k]).
\]

G/B fixed position buffers and Q-only's G position buffers are deterministic
mode state, not part of `C_s`; M10 frequencies are independent trainable
position state. Every tensor returned by `common_state_dict()` is a fresh
clone. Mutating a returned snapshot or one built model therefore cannot change
the factory or another arm.

Construction must leave Python `random`, NumPy legacy RNG, PyTorch CPU RNG, and
available accelerator RNGs unchanged. The unit does not initialize or query a
CUDA device; tests patch accelerator seeding entry points to reject accidental
calls.

## 6. AdamW groups and direct manifest

Let `P` be the ordered set of `model.named_parameters()` with
`requires_grad=True`. The groups form a disjoint partition

\[
P=P_{decay}\ \dot\cup\ P_{zero}.
\]

For this frozen architecture:

\[
P_{zero}=\{p:\dim(p)\le1\}
\cup\{\texttt{backbone.cls_token}\}
\cup\{p:\text{name starts with }\texttt{position.}\}.
\]

Direct inspection establishes that the one-dimensional trainable tensors are
exactly all biases, LayerNorm weights, and LayerScale `gamma_1/gamma_2` for the
current answerer. The explicit skip entries cover the rank-3 CLS token and the
rank-3 M10 frequencies. Therefore `P_decay` contains the Conv, QKV/output/MLP,
and answer Linear weights; Embedding weight; and GRU `weight_ih_l0` and
`weight_hh_l0`.

The implementation calls `timm.optim.param_groups_weight_decay` with
`weight_decay=0.05` and the explicit skip set, then maps returned parameter
identities back to names. It rejects a missing, duplicate, non-trainable, or
unknown parameter before constructing AdamW. Each optimizer group records:

```text
group_name: "decay" or "no_decay"
param_names: tuple of exact fully-qualified names in model order
params: the matching Parameter objects
weight_decay: 0.05 or 0.0
```

AdamW uses `lr=3e-4`, `betas=(0.9, 0.999)`, and `eps=1e-8`. All other options
retain PyTorch 2.6 defaults. Q-only's patch projection remains in
`P_decay`; because its gradient stays `None`, PyTorch AdamW does not create
optimizer state for it or change it on a step.

## 7. Learning-rate and scheduler contract

For ordinary integer update `k` in `[1,8000]`, expose exactly

\[
\eta_k=\begin{cases}
3\times10^{-4}k/400,&1\le k\le400,\\
10^{-6}+\dfrac{3\times10^{-4}-10^{-6}}2
\left[1+\cos\left(\pi\dfrac{k-400}{7600}\right)\right],
&400<k\le8000.
\end{cases}
\]

Thus `eta_1=7.5e-7`, `eta_400=3e-4`, `eta_4200=1.505e-4`, and
`eta_8000=1e-6`. Values outside the finite budget are rejected by the pure
function.

PyTorch 2.6 initializes `LambdaLR` at `last_epoch=0`. Its multiplier for
scheduler epoch `j` is

\[
\eta_{\min(j+1,8000)}/(3\times10^{-4}).
\]

Immediately after construction, optimizer groups therefore hold `eta_1`. The
runner uses that LR for update 1 and calls `scheduler.step()` only after a
successful `optimizer.step()`. After `k<8000` completed updates, the current LR
is `eta_(k+1)`; after update 8000, one final scheduler step retains `eta_8000`
and records `last_epoch=8000`. The later run-budget validator forbids another
optimizer update; the clamp only gives the final scheduler checkpoint a
well-defined LR.

On restore, a caller must construct optimizer and scheduler first, then load
their recorded state dictionaries. Tests verify that restored counters and the
next LR equal an uninterrupted sequence.

## 8. Mature-library and license review

Local inspection on 2026-10-25 found:

| Need | Candidate | Evidence | Decision |
| --- | --- | --- | --- |
| Common model state | PyTorch `Module.state_dict` / `load_state_dict` and `torch.equal` | installed PyTorch `2.6.0+cu124`, BSD-3-Clause; strict state loading and direct tensor comparison already used in the repository | Reuse directly |
| RNG isolation | `torch.random.fork_rng(devices=[])` plus CPU default generator | installed PyTorch API; existing M10 code uses the same no-accelerator pattern | Reuse directly |
| Weight-decay partition | `timm.optim.param_groups_weight_decay` | installed timm `1.0.24`, Apache-2.0; public helper implements one-dimensional/bias/skip-list split | Reuse, then add name manifest and exhaustive identity validation |
| Optimizer | `torch.optim.AdamW` | installed PyTorch `2.6.0+cu124`, BSD-3-Clause; exact betas/epsilon/decoupled decay API | Reuse directly |
| Schedule | PyTorch `LambdaLR`, `SequentialLR`, `LinearLR`, `CosineAnnealingLR` | installed PyTorch scheduler APIs | Select `LambdaLR` because one pure update-index formula is easier to audit without milestone translation |

No dependency is added and no external source is copied. The custom code is
limited to the project-specific seed ownership, direct common-state copy and
verification, parameter-name audit, and frozen schedule function.

## 9. Invariants, edge conditions, and falsification

The implementation must maintain:

1. **One common source:** all arm common tensors equal the factory snapshot by
   `torch.equal`, not merely by equal constructors.
2. **Independent position:** only M10 has trainable position state, seeded by
   exactly `20000+s`; it is excluded from the common copy.
3. **No aliases:** snapshots and independently built arms share no tensor
   storage.
4. **RNG isolation:** factory creation, snapshots, and arm builds do not change
   global Python, NumPy, PyTorch CPU, or accelerator RNG state.
5. **Complete optimizer partition:** every trainable parameter appears once;
   every non-trainable parameter appears zero times.
6. **Semantic decay:** representative Conv/Linear/Embedding/GRU weights have
   decay 0.05; representative biases, LayerNorm, CLS, LayerScale, and M10
   frequencies have decay 0.
7. **Q-only absence of update:** an unused patch projection with `grad is None`
   is unchanged by AdamW and absent from optimizer state after a real Q-only
   loss/update example.
8. **Schedule boundaries:** updates 1, 400, 401, 4200, and 8000 match the frozen
   equation; invalid types and indices fail.
9. **Resume continuity:** optimizer/scheduler state restoration preserves the
   scheduler counter and next-update LR exactly.

The design is falsified by one unequal common tensor, an M10 seed other than
`20000+s`, any shared tensor storage, a changed global RNG, duplicate/missing
optimizer membership, a protocol-listed parameter in the wrong decay group,
Q-only patch projection movement without a gradient, or any schedule boundary
or restored-next-LR mismatch.

## 10. Test and implementation boundary

Add `tests/test_clevr_training.py` before `src/clevr_training.py`. Tests use
small vocabularies and model construction only; no dataset or experiment run:

- strict constructor and mode validation, including both seed bounds, before
  state mutation;
- all four arms directly equal one common snapshot for keys/order/shape/dtype;
- M10 position equals direct `20000+s` construction and stays outside common
  state;
- snapshot/model independence and global RNG isolation;
- exhaustive optimizer identity coverage plus literal representative names for
  every protocol category in G, M10, and Q-only;
- Q-only forward/backward plus one AdamW step proving patch projection stays
  gradient-free and unchanged;
- hand-derived schedule boundaries, LambdaLR step convention, and state restore.

The PR may add only this design, its implementation plan,
`src/clevr_training.py`, and `tests/test_clevr_training.py`. It does not merge
the separate data or sampler branches and does not add a command that accesses
CLEVR, launches GPU work, or starts training.
