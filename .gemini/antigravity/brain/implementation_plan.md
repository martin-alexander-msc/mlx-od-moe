# Implementation Plan

## Goal
Apply correct Qwen3Next linear-attention qkvz fusion ordering.

## Files to Modify
- `mlx_od_moe/server.py`
- `tests/test_server_preprocess.py`

## Approach
1. Update `_preprocess_qwen3next_source_tensors()` to fuse linear-attention
   tensors per key head:
   - reshape `attn_qkv` and `attn_gate` by `linear_num_key_heads`,
   - concatenate as `[qkv_chunk, gate_chunk]` per head,
   - flatten back to `attn_qkvz`.
2. Keep norm sanitize shift behavior intact.
3. Add a regression test to verify:
   - qkvz fusion still happens,
   - qkvz ordering is interleaved (not append-only),
   - target norms are still shifted by +1,
   - non-target tensors remain unchanged.
4. Validate with compile checks and direct runtime sanity script.
