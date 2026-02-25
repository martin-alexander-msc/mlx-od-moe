# Implementation Plan

## Goal
Match Qwen3Next MoE routing behavior in OD runtime to improve generation quality.

## Files to Modify
- `mlx_od_moe/od_moe_layer.py`
- `mlx_od_moe/qwen3_next_od_model.py`
- `tests/test_od_moe_layer.py`

## Approach
1. Add a routing helper in `ODMoELayer` that:
   - validates finite router logits,
   - computes router probs with `mx.softmax(..., precise=True)`,
   - selects top-k experts from probabilities via `argpartition`,
   - conditionally normalizes top-k scores when `norm_topk_prob=True`.
2. Wire `Qwen3NextODConfig.norm_topk_prob` into `ODMoELayer`.
3. Add router-focused tests for:
   - normalization toggle behavior,
   - non-finite logits guard.
4. Run available validation commands and document test limitations if pytest is
   not present in the environment.
