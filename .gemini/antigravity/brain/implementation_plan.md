# Implementation Plan

## Goal
Fix server startup so GGUF-converted `base_model/` directories load correctly, with key translation from GGUF tensor names to runtime model parameter names.

## Files to Modify
- `mlx_od_moe/server.py`
- `mlx_od_moe/weight_loader.py` (new)
- `tests/test_weight_loader.py` (new)
- `README.md`

## Approach
1. Add a dedicated weight loader utility that:
   - Accepts either a safetensors file path or a `base_model/` directory.
   - Loads known component files from directory output.
   - Maps GGUF tensor keys (e.g. `blk.0.attn_q.weight`) to model keys (e.g. `layers.0.attention.q_proj.weight`).
   - Performs safe transpose when source and destination shapes are reversed.
2. Update server initialization order to set up OD-MoE layers first, then load remapped base weights.
3. Add unit tests for mapping, directory loading, and key shape handling.
4. Correct README quick-start commands and output description.
