# Implementation Plan

## Goal
Enable conversion from Q4-quantized GGUF MoE models by dequantizing tensors during conversion and preserving correct tensor orientation.

## Files to Modify
- `convert/gguf_to_od_moe.py`
- `tests/test_expert_extraction.py`
- `README.md`

## Approach
1. Use `gguf.quants.dequantize` for all tensor reads in converter.
2. Align dequantized arrays to GGUF metadata shapes, including reversed-axis transpose handling.
3. Keep packed expert extraction (`ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`) and explicit expert extraction paths.
4. Keep metadata auto-detection for `num_layers` and `num_experts`.
5. Add tests for axis alignment and Q4 tensor dequantization path.
6. Update README to remove now-obsolete "quantized not supported" note.
