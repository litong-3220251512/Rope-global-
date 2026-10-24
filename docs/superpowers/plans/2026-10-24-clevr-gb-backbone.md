# CLEVR G/B Question-Answering Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen PUB-R1.1 RGB+question answerer and connect only the G and B axial position variants, with data-free architecture, phase, initialization, and gradient regressions.

**Architecture:** A standalone `CLEVRQuestionEncoder` maps padded int64 question tokens through `Embedding(vocab_size, 64, padding_idx=0)` and a one-layer unidirectional `GRU(64, 192)`, then gathers the output at each true final token. `CLEVRQuestionAnswerer` reuses the existing from-scratch `RoPEClassifier` constructor to obtain the exact upstream-initialized 224/16, width-192, six-layer, three-head backbone and two-logit head; it adds the question state to the expanded learned CLS before all transformer blocks. G and B share every trainable tensor and differ only in fixed axial phases from `LocalAxialRoPE`, with B fixed at lambda 1 and no trainable position parameter.

**Tech Stack:** Python 3.11, PyTorch 2.6.0, timm 1.0.24, vendored `naver-ai/rope-vit` commit `48d8df5`, `unittest`.

**Spec:** `docs/rope_public_data_experiment_plan.md` (PUB-R1.1), especially §§3, 4.1–4.3, 6, and 10. Mathematical/library prerequisite: `docs/stage1_clevr_math_and_library_review.md`.

## Global Constraints

- This PR implements PUB-R1.1 implementation unit 2 only: common RGB+question backbone, G/B integration, and mathematical/numerical regressions.
- Do not implement M10, Q-only, optimizer groups, sampling, checkpoint/resume, training, evaluation, statistics, or a CLI in this branch.
- Do not import the unmerged runtime dataset PR. The final model consumes only `image`, `question_tokens`, and `question_length` tensors matching that published batch contract.
- Do not read real CLEVR files, download data or weights, invoke a pretrained constructor, run training, or use a GPU. Tests use hand-constructed tensors with `CUDA_VISIBLE_DEVICES=''`.
- Fixed architecture: RGB `3×224×224`, patch16, 196 patch tokens, width 192, depth 6, 3 heads, head dimension 64, MLP width 768, GELU, Pre-LN epsilon `1e-6`, LayerScale `1e-4`, all dropout/drop-path zero, qkv and patch-projection bias enabled, no APE/RPB/QK norm/local mask, two output logits.
- Fixed language path: `Embedding(vocab_size, 64, padding_idx=0)` then one-layer unidirectional `GRU(64, 192)` using the state at `question_length - 1`; only these new modules use their locked PyTorch default initialization.
- Reuse `RoPEClassifier(..., pretrained-free defaults)` and the existing non-in-place manual attention adapter. Do not copy or rewrite upstream patch embedding, transformer blocks, attention, or head.
- Only patch Q/K rotate. CLS and V stay unrotated. G uses fixed base100 global axial phases; B resets each axis's low-frequency planes 8–15 with the 2×2 local coordinate at fixed lambda 1 and adds no trainable PE parameters.
- Keep the legacy Stage 0 classification and segmentation APIs unchanged.
- Use `apply_patch` for every file edit. Each production behavior is preceded by a failing test and a witnessed RED result.

## Review Focus

- Padded batches with unequal question lengths: the shorter example must use its END-state, not the state after NULL padding; Task 1 pins this against a separately run unpadded prefix.
- Language/vision fusion at the first block: the incoming CLS must equal `backbone.cls_token + h`, while all 196 patch tokens must equal the unmodified patch projection; Task 2 captures the first block input.
- Same-seed G/B construction: every non-position state tensor must be `torch.equal`, total trainable parameter counts must match, and B must contain no trainable positional tensor; Task 2 compares named states directly.
- B's fixed phase routing: all six blocks must receive the exact 14×14 B phase tensor at lambda 1; a B forward with diagnostic lambda 0 must be bitwise equal to G under identical non-position state; Task 2 captures block arguments and runs the end-to-end equality.
- Gradient reachability: answer loss must reach Embedding, both GRU weight families, CLS, patch projection, every block's QKV, final norm, and answer head, while B position remains parameter-free; Task 2 checks finite nonzero gradients.

---

### Task 1: Encode Variable-Length CLEVR Questions

**Files:**
- Create: `src/clevr_model.py`
- Create: `tests/test_clevr_model.py`

**Interfaces:**
- Consumes: collated `question_tokens: Tensor[int64, N×L]` padded with NULL=0 and `question_length: Tensor[int64, N]` whose values include START and END.
- Produces: `CLEVRQuestionEncoder(vocab_size: int)` with public modules `embedding: nn.Embedding`, `gru: nn.GRU`, and `forward(question_tokens: Tensor, question_length: Tensor) -> Tensor[N,192]`.
- Later task consumes the encoder without changing its names or output semantics.

- [ ] **Step 1: Add RED tests for the exact language architecture and true-length state**

Add `CLEVRQuestionEncoderTests` with:

- `test_uses_frozen_embedding_and_gru_dimensions_without_custom_reinitialization`: with a fixed seed, assert `embedding.num_embeddings`, width 64, `padding_idx == 0`, one unidirectional GRU layer with input 64 and hidden 192, no dropout, and output shape `[2,192]`. Reset the seed, construct a reference `nn.Embedding` followed by `nn.GRU`, and require every state tensor to be `torch.equal`; this proves no post-construction reinitializer changed the locked PyTorch defaults.
- `test_padded_batch_uses_each_last_valid_token_state`: compare the short row in a padded two-row batch against the same prefix encoded alone; require exact equality and separately compare the long row. This catches using `output[:, -1]` or `h_n` after padded computation.
- `test_rejects_invalid_vocabulary_and_tensor_contracts`: reject bool/non-int/fewer-than-four vocabulary sizes, non-int64/rank-mismatched tokens or lengths, batch mismatch, and sequence width below two.

- [ ] **Step 2: Run the new tests and witness RED**

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest tests.test_clevr_model.CLEVRQuestionEncoderTests -v
```

Expected: FAIL because `src.clevr_model` / `CLEVRQuestionEncoder` does not exist.

- [ ] **Step 3: Implement `CLEVRQuestionEncoder` minimally**

In `src/clevr_model.py`:

- Validate constructor and tensor rank/dtype/batch/width contracts without inspecting START/END/NULL contents or synchronizing tensor values; content and length-range checks belong to the audited dataset/collator. Document that `question_length` must satisfy `2 <= length <= L`.
- Apply the embedding and GRU once. Gather `gru_output[n, question_length[n]-1]` for each batch item and return `[N,192]`.
- Do not call any initialization routine after constructing `nn.Embedding` and `nn.GRU`.

- [ ] **Step 4: Run the focused language tests and witness GREEN**

Run the Step 2 command.

Expected: 3 tests pass, 0 failures.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/clevr_model.py tests/test_clevr_model.py
git commit -m "feat: encode variable-length CLEVR questions"
```

### Task 2: Connect the Frozen RGB Backbone to G and B

**Files:**
- Modify: `src/clevr_model.py`
- Modify: `tests/test_clevr_model.py`

**Interfaces:**
- Consumes: `CLEVRQuestionEncoder.forward(...) -> Tensor[N,192]`, `RoPEClassifier`, `LocalAxialRoPE`, and the exact three tensor fields `image`, `question_tokens`, `question_length`.
- Produces: `CLEVRQuestionAnswerer(vocab_size: int, *, mode: str = "G")` with:
  - `question_encoder: CLEVRQuestionEncoder`
  - `backbone`: the exact six-layer upstream `rope_vit_models`
  - `position`: fixed `LocalAxialRoPE` accepting only mode G or B
  - `forward_tokens(image, question_tokens, question_length, *, lam: Lambda | None = None) -> Tensor[N,197,192]`
  - `forward_features(...) -> Tensor[N,192]`
  - `forward(...) -> Tensor[N,2]`
- Later M10/Q-only and training PRs consume these names; do not add those behaviors here.

- [ ] **Step 1: Add RED tests for the exact frozen architecture and CLS fusion**

Add `CLEVRQuestionAnswererArchitectureTests`:

- `test_constructs_the_exact_pub_r1_1_backbone`: assert all fixed dimensions and module settings from Global Constraints, including 196 patches, six blocks, 3×64 heads, MLP 768, `qkv.bias`, patch bias, no `pos_embed`, zero dropout/drop-path, LayerNorm epsilon, LayerScale values, and `Linear(192,2)` head.
- `test_first_block_receives_question_conditioned_cls_and_unchanged_patches`: capture the first block input. Assert token 0 equals expanded learned CLS plus `question_encoder(...)`; assert tokens 1–196 equal `backbone.patch_embed(image)` exactly.
- `test_forward_returns_two_logits_and_rejects_non_contract_images_or_batches`: assert `[N,2]` logits for valid tensors and reject wrong image rank/channel/height/width, empty batch, or batch disagreement with question tensors.

- [ ] **Step 2: Run the architecture tests and witness RED**

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest tests.test_clevr_model.CLEVRQuestionAnswererArchitectureTests -v
```

Expected: FAIL because `CLEVRQuestionAnswerer` does not exist.

- [ ] **Step 3: Implement the common answerer and G path**

- Expose the final `mode` argument but, for this first GREEN, accept only `"G"`; reject B, C, and unknown strings before allocating the backbone. Step 7 is the only change that enables B after its tests have failed.
- Construct a temporary `RoPEClassifier` with exact arguments G mode, fixed lambda 1, image 224, patch 16, width 192, depth 6, heads 3, classes 2, manual attention, LayerScale `1e-4`, all dropout/drop-path zero, and theta 100. Retain its `backbone` and `position`; this reuses the upstream Linear initialization and never invokes a pretrained path.
- Construct `question_encoder` after the backbone and do not run a whole-model reinitializer.
- In `forward_tokens`, validate only the image/batch interface, create patch tokens, add encoded question state to the expanded CLS, concatenate, build one phase tensor for the 14×14 grid, pass that same tensor to all six blocks, and normalize all 197 tokens.
- `forward_features` returns normalized token 0. `forward` applies the reused `backbone.head` with no additional dropout.

- [ ] **Step 4: Run architecture tests and witness GREEN**

Run the Step 2 command.

Expected: 3 tests pass, 0 failures.

- [ ] **Step 5: Add RED tests for G/B isolation, phase routing, lambda-zero equality, and gradients**

Add `CLEVRQuestionAnswererPositionTests`:

- `test_same_seed_g_and_b_have_equal_non_position_state_and_parameter_count`: reset one seed before each constructor; compare every state entry except `position.*` with `torch.equal`; assert identical names, shapes, dtypes, trainable parameter count, and no position parameters. Assert B's checkpointed fixed lambda is exactly `[1.0]`.
- `test_every_b_block_receives_the_exact_fixed_local_phase`: capture each block's `freqs_cis`; compare all six with `build_axial_cis(14,14,mode="B",lam=1.0,theta=100.0)` using exact equality. Also prove that the captured tensor differs from G in the prescribed low-frequency planes and matches G in the retained planes.
- `test_b_lambda_zero_matches_g_end_to_end_with_paired_state`: construct same-seed eval G/B, run identical image/question tensors with G default and B `lam=0.0`, and require bitwise-equal normalized tokens and logits.
- `test_answer_loss_reaches_every_non_position_component`: on B, backpropagate two-class cross-entropy and assert finite nonzero gradients for embedding weight, GRU `weight_ih_l0` and `weight_hh_l0`, CLS, patch projection, every block QKV, final norm, and answer head; assert `list(model.position.parameters()) == []`.
- `test_rejects_c_and_unknown_modes`: require `ValueError` before construction for C and arbitrary modes.

- [ ] **Step 6: Run the position tests and witness RED**

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest tests.test_clevr_model.CLEVRQuestionAnswererPositionTests -v
```

Expected: FAIL because B construction is still rejected by the G-only implementation from Step 3. Tests unrelated to B may already pass; they are not counted as RED evidence for B.

- [ ] **Step 7: Complete the minimal G/B integration**

Change mode validation to accept exactly `{"G", "B"}` and pass the selected mode into the existing `RoPEClassifier` constructor. Implement only what the failing tests require. Preserve the existing manual attention adapter, pass diagnostic `lam` only to `self.position`, and do not add caches or detach phases. Do not add any trainable region ID, absolute quadrant anchor, object feature, or new PE tensor.

- [ ] **Step 8: Run all new tests and the full CPU suite**

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest tests.test_clevr_model -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest discover -s tests -v
```

Expected: all new tests pass; the full suite passes with exactly the existing CUDA-only skip permitted and no new warning class.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/clevr_model.py tests/test_clevr_model.py
git commit -m "feat: add PUB-R1.1 G and B question answerer"
```

## Final Verification and PR Contract

- [ ] Confirm `git diff --check <merge-base>..HEAD` and that changed paths are only this plan, `src/clevr_model.py`, and `tests/test_clevr_model.py`.
- [ ] Confirm no command accessed real CLEVR, downloaded data/weights, ran training, or exposed a GPU.
- [ ] Generate a whole-branch review package and request one independent read-only review focused on the five Review Focus cases.
- [ ] Fix any Critical/Important finding in one RED→GREEN pass, rerun the focused tests and full CPU suite, and commit the fix.
- [ ] Push `feat/clevr-gb-backbone` and create a real PR against `master`. The PR body must include the trigger, mathematical basis, upstream/library/license decision, exact changes, fresh test evidence, and explicit non-goals.
