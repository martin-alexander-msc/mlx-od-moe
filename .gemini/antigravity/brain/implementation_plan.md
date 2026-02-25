# Implementation Plan

## Goal
Apply the missing Qwen3Next norm sanitize step (`+1.0`) in base preprocessing.

## Files to Modify
- `mlx_od_moe/server.py`
- `tests/test_server_preprocess.py`

## Approach
1. Extend `_preprocess_qwen3next_source_tensors()` to add `+1.0` for 1D norm
   tensors:
   - `blk.*.attn_norm.weight`
   - `blk.*.post_attention_norm.weight`
   - `blk.*.attn_q_norm.weight`
   - `blk.*.attn_k_norm.weight`
   - `output_norm.weight`
2. Keep existing QKV+gate fusion behavior unchanged.
3. Add a regression test to verify:
   - qkvz fusion still happens,
   - target norms are shifted by +1,
   - non-target tensors remain unchanged.
4. Validate with compile checks and direct runtime sanity script.
