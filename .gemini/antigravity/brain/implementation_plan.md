# Implementation Plan

## Goal
Add direct `.safetensors` input conversion path to OD-MoE layout, avoiding GGUF-specific dequantization complexity when MLX/HF safetensors are available.

## Files to Modify
- `convert/safetensors_to_od_moe.py` (new)
- `convert/__init__.py`
- `tests/test_safetensors_conversion.py` (new)
- `README.md`

## Approach
1. Implement safetensors loader for file or directory inputs.
2. Infer `num_layers`/`num_experts` from tensor keys when not provided.
3. Split base tensors into `base_model/*.safetensors`.
4. Extract per-expert tensors into `experts/layer_XX_expert_YYY.safetensors` and write `registry.json`.
5. Support both explicit expert keys and packed per-layer expert tensors.
6. Add focused tests for end-to-end safetensors conversion.
