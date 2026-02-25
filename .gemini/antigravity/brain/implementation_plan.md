# Implementation Plan

## Goal
Make Qwen3 GGUF decoding coherent by matching tokenizer behavior and stop-token logic.

## Files to Modify
- `mlx_od_moe/server.py`
- `mlx_od_moe/gguf_tokenizer.py`
- `mlx_od_moe/model.py`
- `mlx_od_moe/qwen3_next_od_model.py`
- `README.md`

## Approach
1. Extend GGUF tokenizer loader with Qwen2-style pretokenization when
   `tokenizer.ggml.pre=qwen2`.
2. Register GGUF control tokens as tokenizer special tokens so chat markers are
   encoded as single IDs.
3. Parse GGUF special-token metadata (EOS/EOT/EOM/PAD) and derive stop IDs.
4. Propagate EOS override and stop IDs through server and generation loops.
5. Update both model `generate()` methods to support multiple stop IDs.
6. Document Qwen3-specific behavior in README.
