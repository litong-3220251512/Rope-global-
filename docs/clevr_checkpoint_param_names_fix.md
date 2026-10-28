# Checkpoint optimizer-name contract repair

## Confirmed failure and invariant

The production optimizer emits tuple names while the checkpoint schema requires
literal lists. Saving update 0 therefore fails before training. For each group
g with n parameters, preserve the ordered bijection
`names[i] -> parameters[i]`, i in [0,n), and every tensor, hyperparameter and
group membership. Only the metadata container changes tuple -> list.
For example `(weight, bias)` becomes `[weight, bias]`; the order is unchanged.
There is no model, loss, data, G/B phase or optimization-math change.
The repair is falsified if the real factory/optimizer cannot save and restore
update 0 for G/B/M10/Q-only or tensor/state equality changes on restoration.
Keep the established checkpoint schema strict instead of broadening readers.

## Library review (2026-10-28)

Reuse installed PyTorch 2.6.0+cu124 (BSD-3-Clause), AdamW/state_dict and the
existing checkpoint writer/reader; reuse timm 1.0.24 (Apache-2.0),
`timm.optim.param_groups_weight_decay`. Both are maintained upstream libraries;
these installed versions are recorded, not claimed to be latest.
Read local `timm/optim/_param_groups.py` and the official
[PyTorch optimizer contract](https://docs.pytorch.org/docs/2.6/optim.html),
sections Constructing it and Per-parameter options. No new dependency or
optimizer implementation is needed. Python list preserves deterministic order.

## Verification plan and limits

First reproduce failure with the production optimizer and save API. Then test
save/load at update 0 in all four methods, with strict tensor equality after
deliberate model mutation, plus existing training/checkpoint CPU regressions.
Use synthetic CPU objects only; no dataset, downloaded weights, GPU or
experiment launch. This does not establish GPU or task-learning correctness.
