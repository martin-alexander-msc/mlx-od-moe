# Implementation Plan

## Goal
Reduce memory spikes and long-lived resident growth during GGUF OD-MoE inference.

## Files to Modify
- `mlx_od_moe/od_moe_layer.py`
- `mlx_od_moe/expert_store.py`
- `mlx_od_moe/gguf_expert_store.py`
- `mlx_od_moe/shadow_model.py`
- `mlx_od_moe/model.py`
- `mlx_od_moe/qwen3_next_od_model.py`
- `mlx_od_moe/server.py`
- `README.md`

## Approach
1. Remove stale per-layer expert references promptly and clear active layer working sets after each forward pass.
2. Make shadow prefetch opt-in rather than always enabled.
3. Wire shadow predictor dimensions to actual model config when enabled.
4. Enforce strict low-memory cache behavior:
   - support `--cache-size-gb 0` as no-retention mode
   - skip LRU insertion when a single expert exceeds cache budget.
5. Lower default cache size and document memory-oriented runtime flags.
