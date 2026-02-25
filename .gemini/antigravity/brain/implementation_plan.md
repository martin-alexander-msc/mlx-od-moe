# Implementation Plan

## Goal
Fix Qwen3Next preprocessing so norms are not double-shifted.

## Files to Modify
- `mlx_od_moe/server.py`
- `tests/test_server_preprocess.py`
- `README.md`

## Approach
1. In `server.py`, change Qwen3Next norm preprocessing from unconditional
   `+1.0` shift to conditional shift:
   - gather min/max/mean stats across target norm tensors,
   - apply shift only when tensors look zero-centered,
   - log detection stats and whether shift was applied.
2. Extend preprocessing tests:
   - case A: zero-centered norms -> shift applied,
   - case B: already-shifted norms -> no shift.
3. Update README wording to reflect auto-detected norm shifting.
4. Validate with py-compile and targeted preprocessing tests.
