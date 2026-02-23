# Implementation Plan

## Goal
Add converter support for packed MoE expert tensor naming and prevent invalid outputs from quantized packed GGUF blobs.

## Files to Modify
- `convert/gguf_to_od_moe.py`
- `tests/test_expert_extraction.py`
- `README.md`

## Approach
1. Add packed expert extraction path for tensors:
   - `blk.{L}.ffn_gate_exps.weight`
   - `blk.{L}.ffn_down_exps.weight`
   - `blk.{L}.ffn_up_exps.weight`
2. Keep existing explicit-per-expert extraction path intact.
3. Auto-detect `num_layers` and `num_experts` from GGUF metadata by default.
4. Improve base extraction routing key inclusion for `ffn_gate_inp`.
5. Add fail-fast validation for quantized packed tensors so converter raises clear errors instead of writing invalid safetensors.
6. Add/extend tests for packed slicing and quantized guard behavior.
