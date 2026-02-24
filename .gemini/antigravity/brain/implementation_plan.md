# Implementation Plan

## Goal
Support disk-safe Option 2 by using GGUF as live expert source at inference time, avoiding full expert conversion output.

## Files to Modify
- `mlx_od_moe/gguf_expert_store.py` (new)
- `mlx_od_moe/model.py`
- `mlx_od_moe/server.py`
- `mlx_od_moe/__init__.py`
- `convert/gguf_to_od_moe.py`
- `README.md`

## Approach
1. Add GGUF-backed expert store with LRU cache and prefetch API parity.
2. Add server/runtime mode `--gguf-experts` to use GGUF experts directly.
3. Infer model overrides from GGUF metadata in GGUF-expert mode.
4. Add converter mode flags `--base-only` / `--experts-only`.
5. Document low-disk workflow: base-only conversion + GGUF expert source.
