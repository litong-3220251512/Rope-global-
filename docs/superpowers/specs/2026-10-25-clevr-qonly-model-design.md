# CLEVR Q-only Model Design

Status: `C-QONLY-MODEL-R1.0`. This is the model-side half of the PUB-R1.1
Q-only control. It does not authorize data access, an engineering smoke,
training, GPU use, evaluation, or result interpretation.

## Intent

PUB-R1.1 needs a question-prior and non-visual-computation control for the
G/B/M10 comparison. Q-only must preserve the common initialized model and the
full six-block token computation while removing every visual value from the
forward path. A later data-side unit must additionally ensure that a Q-only
run never opens an RGB file or reads a scene artifact.

This unit is split from that data-side work because the accepted model chain
is stacked through PR #11 while the accepted runtime dataset is independently
reviewable in PR #8. Combining either unmerged branch into the other would
duplicate an unrelated PR and violate the repository's one-problem-per-PR
rule. The later runner is the first component allowed to connect both APIs.

## Goals

- Add an explicit `Q-only` mode to `CLEVRQuestionAnswerer`.
- Require `image is None` in that mode so the model cannot accidentally
  consume a loaded RGB tensor.
- Bypass the complete patch projection and construct exactly 196 zero patch
  tokens of width 192.
- Keep G's fixed axial position structure and the same six transformer blocks,
  final normalization, question encoder, CLS fusion, and two-logit head.
- Retain the patch projection module and its initialized parameters in model
  state, but leave both projection weight and bias gradients as `None`.
- Preserve direct same-seed state equality and parameter-count equality with G.

## Non-goals and anti-goals

- Do not add a dataset, collator, sampler, optimizer, checkpoint, runner,
  metrics, train-majority baseline, CLI, or real-data check in this PR.
- Do not accept a dummy image, decode an image, load a scene, or use any
  offline manifest field as a model input.
- Do not feed zero RGB through the convolution, remove patch tokens, freeze
  the transformer, or replace the common answerer with a smaller language-only
  classifier.
- Do not claim that a passing contract test is evidence of task performance.

## Mathematical contract

Let batch size be `N > 0`, padded question width be `L >= 2`, question tokens
be `Q in int64[N,L]`, and true lengths be `ell in int64[N]`. The unchanged
question encoder produces

\[
h=\operatorname{GRU}(\operatorname{Embedding}(Q))_{\ell-1}
\in\mathbb R^{N\times192}.
\]

For the shared trainable CLS vector `c in R^192`, Q-only forms

\[
C_0=c+h\in\mathbb R^{N\times1\times192},\qquad
Z_0=0\in\mathbb R^{N\times196\times192},
\]

and sends `T_0=[C_0;Z_0] in R^{N x 197 x 192}` through the same six Pre-LN
blocks as G. Every block receives G's fixed base-100 axial phase table
`R_G in C^{196 x 32}`; only patch Q/K are rotated, CLS Q/K and every V are
unchanged. The output remains

\[
\operatorname{Linear}_{192\to2}
\left(\operatorname{LN}(T_6)_{:,0}\right)
\in\mathbb R^{N\times2}.
\]

The public model call is
`model(None, question_tokens, question_length)` for Q-only. Passing any image
object is rejected before patch-token construction. Conversely G, B, and M10
continue to require an RGB tensor of shape `[N,3,224,224]`; `None` is rejected.
`lam` is not valid for Q-only because this mode is fixed to G position, and
`m10_seed` remains valid only for M10.

### Exact difference from G

At equal common initialization seed, G and Q-only have exactly the same state
keys, tensor shapes, dtypes, values, trainable parameter names, and trainable
parameter count. Their only forward difference is

\[
Z_G=\operatorname{PatchEmbed}(X)
\quad\longrightarrow\quad
Z_{Q}=0.
\]

Q-only adds no parameter and changes no position tensor. The retained patch
projection is not called, so autograd does not allocate gradients for either
its weight or bias. Every later block still receives 197 tokens; zero patch
states may become nonzero after block 0 and participate in subsequent
attention and MLP computation.

### Invariants, equivalences, and boundary cases

- `Z_0` is exactly zero, with the same dtype and device as `C_0`; it is not a
  persistent parameter or buffer.
- All six blocks receive the same exact G phase object for a forward, matching
  the existing G path.
- The question encoder, CLS, all six blocks, final norm, and answer head remain
  on the answer-loss gradient path. Only the unused patch projection is
  excluded.
- A zero RGB image passed through patch projection is not equivalent: the
  projection has a trainable bias, so `Conv(0)=bias`, and it would still
  require image construction/loading.
- Removing the 196 patches is not equivalent because it changes attention
  normalization and sequence length from 197 to 1.
- Replacing the answerer with `head(c+h)` is not equivalent because it removes
  six blocks and their non-visual computation.
- A one-example batch must work. Empty question batches and malformed question
  tensors remain invalid. Q-only never infers batch size from an image.

### Hand calculation

For `N=1`, the first block input has shape `[1,197,192]`. If the question
encoder returns `h` and the learned CLS vector is `c`, token 0 is exactly
`c+h`; tokens 1 through 196 are exactly 192 zeros each. A zero RGB image would
instead produce each patch as the patch-convolution bias before flattening,
which is generally nonzero. Therefore zero patch tokens must be created after
bypassing the projection, not by synthesizing a black image.

## Approaches considered

1. **Explicit Q-only mode with `image=None` and zero tokens at the patch-token
   boundary — selected.** This makes accidental visual consumption an API
   error, preserves all common state and downstream computation, and directly
   implements PUB-R1.1.
2. **Feed a zero RGB tensor through patch projection — rejected.** It violates
   the no-RGB-I/O intent and differs mathematically because the convolution has
   a bias.
3. **Delete patches or use a separate language-only classifier — rejected.**
   Either changes sequence length, attention normalization, capacity, and
   computation, so the result would not isolate visual information.

The selected approach is an implementation decision by the agent under the
user's standing instruction to continue the accepted PUB-R1.1 sequence. It
does not revise the research protocol.

## Mature-library review

Review date: 2026-10-25. No package, data, or weight was downloaded.

| Candidate | Inspected version/API/license | Decision |
| --- | --- | --- |
| PyTorch `Tensor.new_zeros` | Locally installed PyTorch `2.6.0+cu124`, package metadata `BSD-3-Clause`; the installed docstring specifies a zero tensor that inherits dtype and device by default. | Use `cls.new_zeros((N,196,192))` so zero patches follow the live question/CLS tensor without a custom allocator or device branch. |
| PyTorch `torch.zeros` | Same installed package and license; supports explicit dtype and device. | Not selected because repeating dtype/device arguments creates an avoidable mismatch surface. |
| Existing `PatchEmbed` convolution | Already part of the common model and carries bias. | Retain its parameters/state for paired initialization, but bypass the call in Q-only. It cannot create the required exact zero tokens from zero RGB. |

There is no missing reusable Q-only algorithm: this is an explicit tensor
ablation composed from the installed framework primitive and the accepted
answerer. No community package, pretrained constructor, network API, or custom
zero operator is warranted. Remote maintenance was not re-evaluated; this unit
adds no dependency beyond the project's locked local PyTorch environment.

## Error handling and data boundary

The constructor validates the mode before allocating model parameters.
`Q-only` rejects `m10_seed`; M10 still requires it. During forward, Q-only
rejects any non-`None` image and any `lam`; RGB modes reject `None` through the
existing image contract. Question tensor validation remains owned by
`CLEVRQuestionEncoder`, with an explicit nonempty batch check for the image-free
path.

This API proves only that the model cannot consume RGB in Q-only mode. The
separate data-side unit must provide question/length/answer samples and batches
without opening PNGs, returning image paths, or loading scenes. The runner must
route those batches only to a Q-only model. Until both units are integrated,
the repository must not claim end-to-end no-RGB I/O.

## Falsification and test obligations

Reject the implementation if any of these occurs:

- Q-only accepts or depends on an image, or calls `backbone.patch_embed`;
- the first block receives anything but exactly zero patch tokens of shape
  `[N,196,192]` beside the ordinary question-conditioned CLS;
- any block receives a position table different from G's fixed phase table;
- Q-only and G differ in same-seed state tensors or trainable parameter count;
- patch projection weight or bias receives a tensor gradient after answer loss;
- question embedding/GRU, CLS, every transformer block, final norm, or answer
  head loses its finite nonzero answer-loss gradient;
- patch representations are removed or remain outside later block computation;
- Q-only accepts `lam` or `m10_seed`, or an RGB mode silently accepts `None`;
- implementation adds data access, training, evaluation, or a new dependency.

Passing these tests establishes only the model-side ablation contract. It is
not evidence that question priors explain any CLEVR result.
