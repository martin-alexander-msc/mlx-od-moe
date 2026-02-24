# Implementation Plan

## Goal
Fix safetensors converter failure on BF16 tensors in environments where numpy does not recognize `bfloat16`.

## Files to Modify
- `convert/safetensors_to_od_moe.py`
- `tests/test_safetensors_conversion.py`

## Approach
1. Keep fast-path `safe_open(..., framework='numpy')` loading.
2. Add fallback parser for BF16 TypeError:
   - parse safetensors header directly,
   - decode tensor payload bytes by dtype tag,
   - decode BF16 into float32.
3. Add unit test for BF16 byte decoding correctness.
4. Validate loader on real mlx-community snapshot shard.
