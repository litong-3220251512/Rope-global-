# CLEVR Q-only Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the model-side PUB-R1.1 Q-only control that accepts no image, injects exact zero patch tokens, and otherwise preserves G's initialized model and computation.

**Architecture:** Extend `CLEVRQuestionAnswerer` with one explicit `Q-only` mode. It retains the complete G model state but replaces the patch-projection call with `cls.new_zeros((N,196,192))`, then sends the ordinary question-conditioned CLS plus those patches through G position and all six blocks. Runtime no-RGB dataset access remains a separate PR on the independently reviewable dataset branch.

**Tech Stack:** Python 3.11, PyTorch 2.6.0+cu124, `unittest`, existing vendored RoPE-ViT functions.

**Spec:** `docs/superpowers/specs/2026-10-25-clevr-qonly-model-design.md`

## Global Constraints

- Follow PUB-R1.1 in `docs/rope_public_data_experiment_plan.md`; Q-only's only model difference from G is bypassing patch projection and injecting `[N,196,192]` exact zero tokens.
- Accept `image=None` only for `mode="Q-only"`; reject every non-`None` image in that mode and continue requiring RGB `[N,3,224,224]` for G, B, and M10.
- Q-only uses fixed global base-100 axial position, all 196 patch tokens, all six common blocks, and the common question encoder, CLS, norm, and answer head.
- Preserve every same-seed G state tensor and the trainable parameter count; add no parameter or buffer.
- Patch projection weight and bias stay in state but remain outside the forward graph, with gradient `None` after answer loss.
- Reject `lam` and `m10_seed` for Q-only; do not alter M10's required dedicated seed or existing G/B behavior.
- Do not import the runtime dataset branch, access data/scenes/images, download packages/weights, use GPU, train, evaluate, or add runner/optimizer/statistics/CLI behavior.
- Use only the locally installed PyTorch `Tensor.new_zeros`; add no dependency or custom zero initializer.

## Review Focus

- A caller passes a real tensor or arbitrary object to Q-only: reject it before patch projection rather than silently ignoring a loaded image; covered by Task 1 argument tests and a projection fail-sentinel.
- A caller passes `None` to G, B, or M10: preserve the RGB contract and reject it; covered by Task 1 mode-table tests.
- A refactor feeds black RGB through the biased convolution or drops patch tokens: require literal first-block zeros with sequence length 197 and nonzero patch states at block 1; covered by Task 1 token-flow test.
- Q-only accidentally selects B/M10 position or recreates phase tables per block: compare against the hand-frozen G builder and require one shared phase object across six blocks; covered by Task 1 phase-routing assertions.
- An optimizer helper later mistakes retained projection parameters for active ones: require both projection gradients to remain `None` while representative parameters in every nonvisual component receive finite nonzero gradients; covered by Task 1 gradient test.

---

### Task 1: Q-only model path and regression contract

**Files:**
- Modify: `tests/test_clevr_model.py`
- Modify: `src/clevr_model.py`

**Interfaces:**
- Consumes: `CLEVRQuestionEncoder.forward(question_tokens: Tensor, question_length: Tensor) -> Tensor`, the accepted `RoPEClassifier` backbone, and fixed G `position(14,14,device=...)`.
- Produces: `CLEVRQuestionAnswerer(vocab_size: int, *, mode: str = "G", m10_seed: int | None = None)` accepting `mode="Q-only"` with no extra state.
- Produces: `forward_tokens(image: Tensor | None, question_tokens: Tensor, question_length: Tensor, *, lam: Lambda | None = None) -> Tensor`, plus matching `forward_features` and `forward`; Q-only requires `image=None` and returns the existing token/features/logit shapes.

- [ ] **Step 1: Write the failing public-mode and state-pairing tests**

Add `CLEVRQuestionAnswererQuestionOnlyTests` with
`test_same_seed_state_matches_g_and_mode_specific_arguments_are_rejected`.
At one literal common seed, construct G and wished-for Q-only models and assert
identical ordered state keys, shapes, dtypes, values, named-parameter keys, and
trainable parameter counts. With valid literal question tensors, require
Q-only to reject a real image and `lam`, require Q-only construction to reject
`m10_seed` before consuming the common RNG, and require G/B/M10 to reject
`image=None`.

- [ ] **Step 2: Write the failing zero-token and G-phase test**

Add `test_zero_patch_tokens_use_g_phases_and_continue_through_all_blocks`.
Use real model hooks: a patch-projection pre-hook raises if called, and each
block pre-hook captures its real token input and `freqs_cis`. Call
`model(None, question_tokens, question_length)`, then assert two logits,
first-block shape `[N,197,192]`, exact hand-derived `CLS = cls_token +
question_encoder(...)`, exact zero patch tokens, block-1 patch tokens with a
nonzero absolute sum, six phase deliveries, object reuse across blocks, and
direct equality to fixed global `build_axial_cis(14,14,mode="G",theta=100.0)`.

- [ ] **Step 3: Write the failing gradient and empty-batch tests**

Add
`test_answer_loss_skips_projection_and_reaches_nonvisual_components` and
`test_rejects_empty_question_batch_without_an_image`. Backpropagate literal
two-class cross-entropy and require projection weight/bias gradients to be
`None`; require finite nonzero gradients for question embedding, both GRU
weight families, CLS, every block's QKV weight, final norm, and answer head.
For empty `[0,L]` int64 tokens and `[0]` lengths, require a direct `ValueError`
rather than a downstream attention failure.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest tests.test_clevr_model.CLEVRQuestionAnswererQuestionOnlyTests -v
```

Expected: all new tests fail because `CLEVRQuestionAnswerer` rejects
`mode="Q-only"`; no test should fail from malformed setup or an unrelated
exception.

- [ ] **Step 5: Implement the minimum Q-only path**

In `src/clevr_model.py`:

- accept `Q-only` in the constructor's mode set and construct its common
  classifier with G position, just as M10 uses G before replacing position;
- keep `m10_seed` exclusive to M10;
- widen the three public image annotations to `Tensor | None`;
- in `forward_tokens`, reject non-`None` Q-only images and Q-only `lam` before
  any projection or question computation;
- encode the question, reject `N == 0`, form the ordinary conditioned CLS, and
  call `cls.new_zeros((N, self.backbone.patch_embed.num_patches,
  self.backbone.embed_dim))` without invoking `patch_embed`;
- retain the existing RGB validation/projection path for G/B/M10;
- give Q-only the existing fixed-G phase branch and use the live token device
  instead of dereferencing `image.device`.

Do not introduce a helper, alternate backbone, parameter-free substitute
module, persistent zero buffer, or data-facing API unless a failing test makes
one necessary.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 4 command.

Expected: all four Q-only tests pass with no error. The existing installed
dependency warnings may appear only when their modules are imported; no new
warning is allowed.

- [ ] **Step 7: Run the complete CPU regression suite**

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest discover -s tests -v
```

Expected: 139 tests pass, the one existing CUDA-only test is skipped, and only
the two existing timm import warnings plus the vendored RoPE-ViT autocast
deprecation warning appear.

- [ ] **Step 8: Inspect scope and commit**

Run `git diff --check`, inspect `git diff`, and confirm only the two task files
changed since the documentation commits. Then commit:

```bash
git add src/clevr_model.py tests/test_clevr_model.py
git commit -m "feat: add CLEVR Q-only model path"
```

## Final Verification and Review

- [ ] Rerun the focused Q-only tests and the complete CPU suite on committed
  `HEAD`; record exact counts, skip, and warnings.
- [ ] Verify `git diff --check feat/clevr-m10..HEAD`, branch status, commit
  range, and the four allowed paths (design, plan, model, tests).
- [ ] Request one fresh read-only whole-branch review focused on image bypass,
  G-state/phase parity, first-block zeros, later token participation, argument
  boundaries, gradient exclusion, and accidental data/training scope.
- [ ] Fix only confirmed Critical/Important findings in one focused TDD pass,
  rerun verification, and create a real stacked PR with base
  `feat/clevr-m10`. Do not merge or tag it.
