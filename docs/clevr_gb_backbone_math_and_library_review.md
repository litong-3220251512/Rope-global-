# CLEVR G/B RGB+Question Backbone Mathematical and Library Review

Version: **CGB-BACKBONE-R1.0**. Audit date: 2026-10-24. Applicable protocol: PUB-R1.1.

> **Late-correction notice.** This prerequisite was written only after the RGB+question implementation had reached reviewed commit `ff01ba7651ed713a48bddbfbb90fcf3a087e34c0`. Whole-branch final review found that the implementation plan had incorrectly named `docs/stage1_clevr_math_and_library_review.md`, which freezes the withdrawn 4/5-channel marker task. That older document remains unchanged as historical evidence and is not applicable to this backbone. This document cannot retroactively have preceded coding: it records the corrected prerequisite and a post-implementation re-audit. Future changes to this unit must review this version, or an explicit successor, before code is changed.

This review defines only the common RGB+question backbone and its G/B positional difference. It neither reports CLEVR performance nor claims novelty, superiority, robustness, or task success.

## 1. Scope, symbols, and domains

Let $N\geq1$ be the batch size, $V\geq4$ the training-vocabulary size, and $L\geq2$ the padded question width. The model contract is:

- RGB input $X\in\mathbb{R}^{N\times3\times224\times224}$, supplied as FP32 by PUB-R1.1 preprocessing;
- question IDs $Q\in\{0,\ldots,V-1\}^{N\times L}$, stored as `torch.int64`, with `<NULL>=0`, `<START>=1`, `<END>=2`, and `<UNK>=3`;
- true lengths $\ell\in\{2,\ldots,L\}^N$, stored as `torch.int64`, where each length includes START and END;
- logits $s\in\mathbb{R}^{N\times2}$, ordered as no $=0$, yes $=1$.

The dataset/collator must guarantee token value ranges, START/END placement, $2\leq\ell_n\leq L$, and compatible devices. `CLEVRQuestionEncoder` deliberately validates tensor type, dtype, rank, batch agreement, and $L\geq2$, but does not read token contents or synchronize length values during forward.

### 1.1 Language path and END-state extraction

The selected PyTorch modules are

\[
E=\operatorname{Embedding}_{V\to64}(Q)\in\mathbb{R}^{N\times L\times64},
\qquad
O,\_ = \operatorname{GRU}_{64\to192}(E),
\]

where the GRU is one-layer, unidirectional, and batch-first. The question representation is gathered row by row:

\[
h_n=O_{n,\ell_n-1,:},\qquad h\in\mathbb{R}^{N\times192}.
\]

Thus the state at END is used even when later columns contain NULL padding. Neither `O[:, -1]` nor the hidden state after processing padding is an allowed replacement.

### 1.2 RGB patch path, fusion, and output

`PatchEmbed` is a biased `Conv2d(3,192,kernel_size=16,stride=16)` followed by flatten/transpose, producing

\[
Z=\operatorname{PatchEmbed}(X)\in\mathbb{R}^{N\times196\times192}.
\]

For learned CLS tensor $c\in\mathbb{R}^{1\times1\times192}$, language enters exactly once, before the first transformer block:

\[
C_0=\operatorname{expand}_N(c)+h[:,\mathrm{None},:]
\in\mathbb{R}^{N\times1\times192},
\qquad
T_0=[C_0;Z]\in\mathbb{R}^{N\times197\times192}.
\]

There are six Pre-LN blocks. Each has width 192, three heads of width 64, a 768-wide GELU MLP, LayerNorm epsilon $10^{-6}$, LayerScale vectors initialized to $10^{-4}$, and zero dropout/drop-path. Schematically, for block $r$,

\[
U_r=T_r+\gamma_{r,1}\odot
  \operatorname{Attn}(\operatorname{LN}_{r,1}(T_r);\Phi),
\qquad
T_{r+1}=U_r+\gamma_{r,2}\odot
  \operatorname{MLP}(\operatorname{LN}_{r,2}(U_r)).
\]

There is no absolute positional embedding, relative-position bias, Q/K normalization, or local attention mask. The public outputs are

\[
Y=\operatorname{LN}(T_6)\in\mathbb{R}^{N\times197\times192},
\quad f=Y_{:,0,:}\in\mathbb{R}^{N\times192},
\quad s=\operatorname{Linear}_{192\to2}(f).
\]

## 2. Exact G/B mathematical contract

Patch coordinates are $p=(x,y)\in\{0,\ldots,13\}^2$. Each 64-dimensional head contains 32 complex rotation planes: x planes 0--15 and y planes 16--31. For within-axis plane $t\in\{0,\ldots,15\}$,

\[
\omega_t=100^{-t/16},\qquad o(a)=7\left\lfloor\frac{a}{7}\right\rfloor,
\quad a\in\{x,y\}.
\]

Define the coordinate used by mode $m$ as

\[
g_{G,t}(a)=a,
\qquad
g_{B,t}(a;\lambda)=
\begin{cases}
a,&t<8,\\
a-\lambda o(a),&t\geq8.
\end{cases}
\]

The x-plane phase is $\exp(i\omega_tg_{m,t}(x))$, and the corresponding y-plane phase is $\exp(i\omega_tg_{m,t}(y))$. Production B fixes $\lambda=1$; planes 0--7 on each axis remain global, while low-frequency planes 8--15 use the coordinate within one of the $2\times2$ regions. G and B add no trainable positional tensor.

In each block, Q/K/V have shape $N\times3\times197\times64$. The phase tensor has shape $196\times32$ complex values and broadcasts over batch and heads. `differentiable_attention` rotates only patch Q and patch K (token indices 1--196). CLS Q/K and every V remain unrotated.

### 2.1 Invariants, equivalences, and limitations

- Every phase has unit modulus, so each two-channel rotation preserves norm; a self-pair has zero relative phase.
- If two patches share the same x region and the same y region, their B relative phase equals G on the reset planes because the common origin cancels. The retained planes are already global.
- This same-region statement is only a patch--patch phase fact for a fixed pair. It does not imply equal block outputs or model outputs: cross-region patches and unrotated CLS interactions remain in attention, and altered earlier states propagate to later blocks.
- At $\lambda=0$, $g_{B,t}=g_{G,t}$ for every coordinate and plane. With paired non-position state and identical inputs, B must therefore degenerate to G. The current regression requires exact phase, normalized-token, and logit equality in the tested CPU path.
- At the production $\lambda=1$, local-plane aliases such as $x=0$ and $x=7$ are intentional. Global planes 0--7 remain different, so B is not an absolute region-ID, quadrant anchor, object-aware encoding, or total loss of global position.
- G/B equivalence also requires identical non-position parameters and buffers, execution order, inputs, and backend. Equal formulas alone do not repair an unpaired initialization.

### 2.2 Hand-worked question-length example

Take $N=2,L=5$:

```text
Q[0] = [START, red, END, NULL, NULL],  length[0] = 3
Q[1] = [START, is, left, cube, END],   length[1] = 5
```

Embedding produces `[2, 5, 64]`; the batch-first GRU produces $O$ with shape `[2, 5, 192]`. Zero-based gather indices are `length - 1 = [2, 4]`, so

\[
h_0=O[0,2,:],\qquad h_1=O[1,4,:].
\]

For the shorter row, columns 3 and 4 are computed but cannot replace $O[0,2,:]$. The resulting $h_0$ is added only to row 0's expanded CLS; neither row's 196 patch tokens is changed by fusion.

### 2.3 Hand-worked boundary and phase example

Consider x-plane $t=8$, where $\omega_8=100^{-8/16}=0.1$, with query patch $x_q=6$ and key patch $x_k=7$. Using the relative-angle convention in the attention identity $q^TR(\phi_k-\phi_q)k$:

\[
\Delta\phi_G=0.1(7-6)=0.1,
\qquad
\Delta\phi_B=0.1[(7-7)-(6-0)]=-0.6.
\]

Crossing the region boundary therefore shifts this local-plane relative angle by $-0.7$ rad; it does not preserve the global displacement. By contrast, $x_q=8,x_k=10$ share origin 7:

\[
\Delta\phi_B=0.1[(10-7)-(8-7)]=0.2
=0.1(10-8)=\Delta\phi_G.
\]

These are phase calculations, not evidence that either mode performs better on CLEVR.

### 2.4 Boundary conditions and falsification criteria

The contract is falsified if any of the following occurs:

1. a padded row uses a state after END, or the language modules do not map 64 to 192 as specified;
2. fusion changes patch tokens before block 0, occurs after a block, or does anything other than $c+h$ to CLS;
3. an input other than RGB/question tensors enters forward, the grid is not $14\times14$, or logits are not `[N, 2]`;
4. B changes planes 0--7, fails to reset planes 8--15 at $\lambda=1$, rotates CLS or V, or introduces a trainable position parameter;
5. B at $\lambda=0$ fails to match G under paired state, or same-seed construction changes any non-position state tensor;
6. loss gradients cannot reach the language encoder, CLS, patch projection, every block's QKV, final norm, or answer head in an ordinary B forward;
7. a claimed same-region equality is extended to a cross-boundary pair or to an entire layer/network.

Valid edge cases include $\ell_n=2$, $\ell_n=L$, patches at coordinates 0 and 13, the boundary pair 6/7, and diagnostic $\lambda=0$. Invalid vocabulary sizes, tensor ranks/dtypes, image shapes, empty answerer batches, and batch mismatches must be rejected as specified by the implementation interface. Token-content and numeric-length validation remains an upstream collator obligation, as stated in Section 1.

## 3. Initialization and G/B pairing

Initialization order is part of the comparison contract:

1. `CLEVRQuestionAnswerer` rejects modes other than G/B before random allocation.
2. It constructs `RoPEClassifier`. That adapter first constructs fixed `LocalAxialRoPE`; for G/B this adds buffers but consumes no random numbers. It then constructs the vendored `rope_vit_models` backbone and binds `differentiable_attention` to all six attention blocks.
3. The answerer retains the classifier's `backbone` and `position`, then constructs `CLEVRQuestionEncoder`: Embedding first, GRU second. No whole-model reinitializer runs afterward.

The vendored constructor's internal order is also retained: its base builds and initializes patch embedding, provisional CLS/APE state, blocks, norm, and head; `rope_vit_models` then replaces CLS with its RoPE-ViT CLS initialization and disables APE when `use_ape=False`. Even random draws for subsequently replaced tensors affect the stream, so refactoring this order is not neutral to pairing.

For every experimental seed, G and B must start from one common non-position state, or from construction with the random seed reset immediately before each identical constructor sequence followed by direct `torch.equal` checks. Names, shapes, dtypes, and every non-position tensor must match. B's only additional state is its fixed `[1.0]` lambda buffer; it owns no positional parameter. The current unit tests exercise same-seed construction, while the later training unit remains responsible for saving and copying the common initial state.

## 4. Library and API selection

All evidence below was inspected locally on 2026-10-24; no network query, package installation, model download, or pretrained constructor was used.

| Component | Fixed local version/revision and exact used API | Decision and rejected alternative |
| --- | --- | --- |
| PyTorch language modules | `torch 2.6.0+cu124`; `nn.Embedding(V, 64, padding_idx=0)` and `nn.GRU(64, 192, num_layers=1, batch_first=True, bidirectional=False)` | Adopt the library-provided tensor/RNN APIs and their default initialization. Do not reimplement embedding or GRU. `pack_padded_sequence` is not needed for correctness: the one-way GRU runs once and the model gathers the END output directly; content/range checks remain with the collator. |
| timm vision primitives | `timm 1.0.24`; vendored code reaches `PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=192)` and `Mlp(in_features=192, hidden_features=768, act_layer=nn.GELU, drop=0.0)` through `deit.models_v2.vit_models` | Adopt these primitives through the fixed vendored constructor, rather than rewriting patch projection or MLP. Do not call a timm model registry/pretrained path because it would not preserve the frozen six-layer backbone and could introduce weights or architecture changes. |
| Vendored RoPE-ViT | `naver-ai/rope-vit` commit `48d8df50250056a837672609bf764d80fb545869`; `rope_vit_models(...)`, `RoPE_Layer_scale_init_Block`, `RoPEAttention`, and `apply_rotary_emb` | Adopt the fixed backbone/block/rotation implementation without its registered pretrained constructors. Do not call vendored `forward_features`, because this unit must insert question-conditioned CLS and project-specific G/B phases before all blocks. Do not use upstream `RoPEAttention.forward` directly because its in-place Q/K view assignment is replaced by the already tested non-in-place adapter. |
| Existing project adapters | `RoPEClassifier(mode, fixed_lambda=1.0, image_size=224, patch_size=16, embed_dim=192, depth=6, heads=3, num_classes=2, attention_backend="manual", init_scale=1e-4, drop_path_rate=0, drop_rate=0, attn_drop_rate=0, theta=100)`, `LocalAxialRoPE(head_dim=64, mode, theta=100, fixed_lambda=1.0)`, and `differentiable_attention(self, x, freqs_cis)` | Adopt as thin adapters so G/B share the existing backbone, phase layout, and non-in-place attention. New code is limited to question encoding, CLS fusion, fixed shape validation, and flow through the reused head. Reimplementing Transformer blocks, rotation, or attention would create a second numerical path without a task-specific need. |

The separately reviewed broader architecture survey in PR #9 considers FiLM, MAC, LCGN, RN, and other references. None is adopted here because replacing the frozen backbone would also change conditioning, capacity, coordinates, pretraining, or supervision and would no longer isolate G versus B. This prerequisite does not depend on that survey: the APIs and decisions needed by this unit are fully stated above.

### 4.1 Maintenance status, bounded to local evidence

- Local package metadata on 2026-10-24 reports PyTorch `2.6.0+cu124` and timm `1.0.24`, and the reviewed implementation imports and exercises the named APIs. This establishes the pinned local environment only; without a network check it does not establish that either is the latest release or characterize upstream activity after the installed versions.
- The vendored Git object is exactly commit `48d8df50250056a837672609bf764d80fb545869`, whose local commit date is 2025-10-29. The locally available history shows earlier commits, but cannot establish remote maintenance after that revision. Pinning is intentional for reproducibility.
- The vendored code imports legacy `timm.models.layers` and `timm.models.registry` paths that emit deprecation `FutureWarning`s in the recorded branch tests. They remained functional in the pinned environment; future dependency upgrades require a separate compatibility review rather than silent substitution.
- `RoPEClassifier`, `LocalAxialRoPE`, and `differentiable_attention` are repository-maintained adapters with local regression coverage. This is not a claim of external maintenance or general-purpose API stability.

### 4.2 License record and file-level caveat

- Installed PyTorch metadata declares **BSD-3-Clause**.
- Installed timm metadata declares **Apache-2.0**.
- `vendor/rope-vit/LICENSE_rope-vit` is the RoPE-ViT root **Apache-2.0** license. That root label is not the complete file-level account: `vendor/rope-vit/NOTICE` explicitly lists `models/vit_rope.py` among files originating from `meta-llama/codellama`, states that subcomponents are subject to their separate terms, and reproduces the **Llama 2 Community License** terms. The repository also retains `vendor/rope-vit/LICENSE_llama`. Because this unit imports `models/vit_rope.py`, its use and redistribution must preserve and assess the root license, NOTICE, and applicable Llama 2 terms together; the file must not be described simply as Apache-2.0.

This is a source and license inventory, not legal advice. The implementation uses the pinned vendored file and thin local adapters; this correction copies no third-party source.

## 5. Retrospective code and test re-audit

The audit target is reviewed head `ff01ba7651ed713a48bddbfbb90fcf3a087e34c0`, chiefly `src/clevr_model.py` and `tests/test_clevr_model.py`, with reused-operator evidence in the named dependency files/tests. **PASS** below means the current source and regression assertions conform on direct review; it does not mean this document existed before implementation. No tests were rerun for this documentation-only correction. Existing branch reports record 13 focused CLEVR model tests passing and, before this correction, 126 CPU tests passing with one existing CUDA-only skip.

| Contract | Source evidence | Regression evidence | Result / gap |
| --- | --- | --- | --- |
| Embedding 64, one-way batch-first GRU 192, untouched defaults | `src/clevr_model.py`: `CLEVRQuestionEncoder.__init__` | `test_uses_frozen_embedding_and_gru_dimensions_without_custom_reinitialization`; one-call hook tests | **PASS** |
| Gather each row's END state | `gru_output[batch_indices, question_length - 1]` | padded short/long rows are compared with separately encoded prefixes | **PASS** |
| Only metadata validation inside encoder | explicit type/dtype/rank/batch/width guards | invalid-contract cases | **PASS with bounded upstream precondition:** token IDs, START/END contents, and numeric length bounds are not runtime-validated here by design; malformed caller data remains a collator failure. |
| RGB-only 224/16 patch path yields 196×192 | `backbone.patch_embed(image)` after fixed image checks | exact patch module/grid/bias assertions and invalid image cases | **PASS** |
| Fusion is exactly learned CLS + question state; patches unchanged | `cls = cls_token.expand(...); cls = cls + question.unsqueeze(1); cat` | first-block pre-hook compares CLS and all patches exactly | **PASS** |
| Six-block frozen architecture and `[N,2]` logits | fixed `RoPEClassifier` arguments; final norm and reused head | exact module/epsilon/LayerScale/dropout/QKV/MLP/head assertions; output-shape test | **PASS** |
| Patch Q/K rotate; CLS Q/K and all V do not | `attention_backend="manual"` binds `differentiable_attention` | CLEVR test pins the bound adapter; `tests/test_attention_compatibility.py` directly checks patch-only Q/K rotation | **PASS** |
| B uses exact planes and one shared 14×14 phase tensor in all blocks | `self.position(14,14)` once, then six block calls | exact B phase, retained/reset plane slices, same object in every block | **PASS** |
| $\lambda=0$ degenerates B to G | diagnostic `lam` is forwarded only to `position` | paired G/B normalized tokens and logits require `torch.equal` | **PASS** |
| No trainable B position state and paired G/B initialization | fixed lambda and constructor ordering | all non-position state tensors equal under reset seed; parameter names/counts match; position parameter lists empty | **PASS** |
| Boundary/same-region/local-alias phase behavior | reused `build_axial_cis` in `src/local_rope.py` | `tests/test_local_rope.py` covers same-region logits, boundary/cross-region hand derivation, fixed-one aliasing, and zero override | **PASS for reused dependency** |
| Answer gradient reaches all required non-position paths | ordinary answerer forward | embedding, both GRU weight families, CLS, patch projection, six QKV weights, final norm, and head have finite nonzero gradients | **PASS** |
| Only G/B accepted before allocation | constructor checks mode before `RoPEClassifier` | C/unknown raise and leave seeded next random draw unchanged | **PASS** |
| Performance, data correctness, and training behavior | absent from this unit | absent by scope | **NOT CLAIMED / not a gap in this prerequisite** |

No production or test defect was found in this re-audit, so this correction changes neither file. The remaining explicit boundaries are upstream value/content validation and later training-state copying; neither authorizes a performance or novelty claim.
