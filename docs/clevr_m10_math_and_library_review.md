# CLEVR M10 mathematical contract and library review

Status: `C-M10-R1.0`, pre-implementation contract for the PUB-R1.1 M10
comparison arm. This document does not authorize a data run, engineering
smoke, pilot, training, GPU use, or evaluation.

## 1. Scope and baseline

The frozen source of truth is
[`rope_public_data_experiment_plan.md`](rope_public_data_experiment_plan.md),
PUB-R1.1. The common model is the accepted G/B question-answering backbone
from PR #10: RGB `224 x 224`, `16 x 16` patches, 196 patch tokens, width 192,
six blocks, three heads, head dimension 64, question-conditioned CLS before
block 0, no APE/RPB/QK normalization/local mask, and a two-logit answer head.

This unit adds only the M10 position arm. It does not add Q-only, optimizer
groups, paired run construction, checkpointing, a CLI, metrics, or any new
candidate method. M10 is the published RoPE-Mixed parameterization adapted to
the frozen backbone. It is an existing-method reference, not a project
innovation and not a full reproduction of the paper's architecture or
training recipe.

## 2. Mathematical definition

Let layers be \(\ell\in\{0,\ldots,5\}\), heads be
\(h\in\{0,1,2\}\), complex planes be \(j\in\{0,\ldots,31\}\), and
patches be \(n\in\{0,\ldots,195\}\). The row-major patch coordinate is

\[
x_n=n\bmod 14,\qquad y_n=\left\lfloor n/14\right\rfloor.
\]

For \(t\in\{0,\ldots,15\}\), define the base-10 magnitude

\[
m_t=10^{-t/16}.
\]

For each layer and head, draw one independent initialization angle
\(\alpha_{\ell h}\sim\mathcal U[0,2\pi)\) from the M10-only seed stream and
initialize two orthogonal directions at every magnitude:

\[
\begin{aligned}
(f^x_{\ell h t},f^y_{\ell h t})
  &=m_t(\cos\alpha_{\ell h},\sin\alpha_{\ell h}),\\
(f^x_{\ell h,16+t},f^y_{\ell h,16+t})
  &=m_t(\cos(\alpha_{\ell h}+\pi/2),
         \sin(\alpha_{\ell h}+\pi/2)).
\end{aligned}
\]

Both components then become ordinary trainable parameters. The patch phase is

\[
R_{\ell hnj}
=\exp\left(i\left[x_nf^x_{\ell hj}+y_nf^y_{\ell hj}\right]\right)
\in\mathbb C.
\]

For block \(\ell\), the real Q and K tensors have shape
`[N, 3, 197, 64]` and are viewed as complex tensors of shape
`[N, 3, 197, 32]`. Only patch entries 1--196 are multiplied by
`R[ell]`, whose shape is `[3, 196, 32]`; CLS Q/K and all V entries remain
unrotated. The parameter tensor follows the official flattened storage layout
`[2, 6, 96]`, where 96 is `3 heads x 32 planes`, while the computed phases
have semantic shape `[6, 3, 196, 32]`. The exact trainable increment is

\[
2\times6\times3\times32=1152.
\]

The paper defines the mixed phase and relative attention term in Section 3.2,
Eqs. (14)--(15), and states that frequencies are learned separately for every
head and self-attention layer
([Heo et al., 2024, Section 3.2](https://arxiv.org/html/2403.13298v2#S3.SS2)).
The implementation constants above are the frozen PUB-R1.1 adaptation.

### 2.1 Difference from G and B

G uses fixed axial frequency vectors: half of the 32 complex planes use only
x and half use only y, with base 100, and the same phase table is supplied to
all six blocks. B keeps that layout and base but replaces coordinates in half
of each axis's planes with fixed 2-by-2 within-region coordinates. Neither G
nor B has trainable position parameters.

M10 instead gives every plane a two-dimensional direction, uses base 10 at
initialization, learns both components independently, and has a separate set
for each layer and head. Therefore M10 minus G jointly changes the initial
spectrum, direction parameterization, per-layer/per-head sharing, and
learnability. No result may attribute an M10 difference to just one of those
factors.

### 2.2 Invariants, equivalences, and limits

- Every phase has unit modulus, so applying one phase preserves each complex
  Q/K plane's norm.
- For fixed frequencies, a Q/K pair depends on displacement only:
  `R(p) * conj(R(q)) = exp(i ((p-q) dot f))`. Translating both patches by the
  same offset leaves this relative phase unchanged.
- A self-pair has relative phase one.
- If one component of a frequency vector is zero, that plane is axial. This
  is a representational special case, not the M10 initialization and not an
  end-to-end equivalence to G, which has a different plane allocation, base,
  and sharing pattern.
- Frequency updates preserve unit-modulus phases but need not preserve the
  initialized magnitudes or orthogonality. The implementation must not project
  them back or add constraints absent from PUB-R1.1.
- CLS remains unrotated. M10 supplies no region ID, object coordinate,
  program field, scene graph, or answer-derived feature.

### 2.3 Hand calculation

Take one head/layer with \(t=0\), so \(m_0=1\), and choose the illustrative
angle \(\alpha=0\). The two initialized frequency vectors are `(1, 0)` and
`(0, 1)`. Patch index 44 has `(x, y) = (2, 3)`, hence its two phases are
`exp(2i)` and `exp(3i)`. Against patch `(1, 1)`, the relative phases are
`exp(i)` and `exp(2i)`. Translating both positions by `(4, 5)` leaves the
differences `(1, 2)` and therefore both relative phases unchanged.

## 3. Randomness, gradients, and numerical behavior

The public constructor requires an explicit non-negative integer `m10_seed`
for mode M10. The M10 initializer executes the official random-frequency
routine inside a CPU `torch.random.fork_rng` context seeded only with that
value. Construction order remains backbone, question encoder, then M10
frequencies. Consequently:

1. the same ordinary model seed yields bitwise-equal non-position state for G,
   B, and M10;
2. M10 frequency initialization does not advance the caller's global CPU RNG;
3. the later run builder can pass the preregistered `20000 + s` without
   coupling it to common initialization or sample order.

Coordinates are registered buffers and are cast to the current frequency
dtype before phase construction, enabling an independent float64 gradient
check. Phases are recomputed from the live parameter on every forward call.
They are never detached, stored across optimizer updates, or cached as a
completed autograd graph. The existing non-in-place attention adapter remains
the only attention path, so gradients must reach patch Q, patch K, and M10
frequencies.

Production precision policy and tolerances remain those in PUB-R1.1. A
float64 analytical-vs-finite-difference check tests mathematical gradients;
it does not replace the production-precision regression, and neither is a
claimed error bound.

## 4. Mature implementation search and selection

Review date: 2026-10-25. No package, data, or weight was downloaded.

| Candidate | Version/license and inspected API | Decision |
| --- | --- | --- |
| Official NAVER RoPE-ViT | Vendored commit `48d8df50250056a837672609bf764d80fb545869`. The project root carries [`LICENSE_rope-vit`](../vendor/rope-vit/LICENSE_rope-vit), while the vendored [`NOTICE`](../vendor/rope-vit/NOTICE) explicitly lists `models/vit_rope.py` among files originating from Code Llama and records separate Llama 2 terms; [`LICENSE_llama`](../vendor/rope-vit/LICENSE_llama) is retained. This is a file-level license record, not a new legal-compatibility conclusion. `models/vit_rope.py` provides `init_random_2d_freqs`, `init_t_xy`, `compute_mixed_cis`, the `[2, depth, heads*planes]` parameter layout, and per-block phase selection. The remote official file was also inspected at [`naver-ai/rope-vit`](https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py). | Adopt the three pure frequency/grid functions through a thin project module. Keep the existing non-in-place attention adapter and question-conditioned forward, rather than calling the upstream full model forward. |
| timm | Locally installed `timm 1.0.24`, Apache-2.0. `timm.layers.pos_embed_sincos.RotaryEmbeddingMixed` is maintained as a general mixed-RoPE module and exposes per-depth trainable frequencies. | Do not adopt in this unit. It emits timm's real concatenated sine/cosine representation for a different rotation API and grid-indexing interface, while the frozen backbone consumes the official complex phase tensor. Adapting it would add a conversion and a second plane-order contract with no task benefit. |
| PyTorch | Locally installed `torch 2.6.0+cu124`, BSD-3-Clause. `nn.Parameter`, `torch.random.fork_rng`, complex tensors, `torch.polar`, and autograd provide the underlying primitives. | Adopt these primitives; PyTorch does not replace the task-specific official Mixed layout. No custom RNG, complex arithmetic engine, or optimizer is introduced here. |

The selected path is the smallest composition of the published official
implementation and the already tested project attention adapter. No
pretrained constructor or network-capable API is called.

## 5. Rejected implementation approaches

1. **Instantiate the complete upstream mixed backbone.** Rejected because its
   `forward_features` owns patch/CLS flow and would bypass the frozen
   question-conditioned CLS path. It also couples model construction to
   upstream architecture defaults outside this comparison.
2. **Use timm `RotaryEmbeddingMixed` directly.** Rejected for the incompatible
   phase representation and rotation interface described above.
3. **Rewrite the mixed initializer and phase operator.** Rejected because the
   official functions are already vendored under a compatible license and
   define the exact layout this comparison intends to preserve.
4. **Cache phase tensors.** Rejected because M10 frequencies are trainable;
   cross-update caching can reuse stale values or an invalid autograd graph.

## 6. Falsification and test obligations

The implementation is rejected if any of the following occurs:

- the parameter count, storage layout, phase shape, row-major coordinates, or
  per-block/head mapping differs from the official mapping above;
- initialized magnitudes or orthogonal direction pairs differ from base 10;
- equal common seeds do not give bitwise-equal non-position tensors across G
  and M10, or M10 construction advances the caller's CPU RNG;
- a selected phase differs from a hand-computed `cos(angle) + i sin(angle)`;
- phase norm, self-pair, or common-translation relative-phase invariants fail;
- float64 finite differences disagree with autograd for frequencies or test
  inputs under the preregistered tolerance;
- an answer loss fails to produce finite nonzero gradients for M10 frequency,
  Q, or K parameters;
- a frequency update does not alter a newly computed phase, or the completed
  phase tensor is reused across forwards;
- mode M10 silently accepts `lam`, omits its dedicated seed, or G/B silently
  accept an irrelevant M10 seed;
- the change adds any input beyond RGB plus encoded original question and
  supervised answer.

Passing these checks establishes the mathematical and engineering contract
only. It is not evidence that M10 improves CLEVR accuracy or that G/B is
effective.
