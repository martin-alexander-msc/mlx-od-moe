# Implementation Plan

## Goal
Reduce conversion disk usage by supporting configurable output dtype and defaulting to float16 for dequantized tensors.

## Files to Modify
- `convert/gguf_to_od_moe.py`
- `tests/test_expert_extraction.py`
- `README.md`

## Approach
1. Add `--output-dtype {float16,float32}` (default `float16`) to converter CLI.
2. Thread output dtype through base and expert extraction paths.
3. Cast dequantized tensors to selected dtype before writing safetensors.
4. Add tests for `_read_tensor_data(..., output_dtype=...)` dtype behavior.
5. Document float16 recommendation for storage footprint.
