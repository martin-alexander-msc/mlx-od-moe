# Implementation Plan

## Goal
Improve startup robustness for non-Kimi conversions and surface actionable errors when conversion output is incompatible.

## Files to Modify
- `mlx_od_moe/model.py`
- `mlx_od_moe/__init__.py`
- `mlx_od_moe/server.py`
- `mlx_od_moe/weight_loader.py`
- `tests/test_weight_loader.py`

## Approach
1. Introduce a generic config class name (`ODMoEConfig`) and keep `KimiODMoEConfig` as compatibility alias.
2. Infer safe config overrides (`vocab_size`, `hidden_size`, `num_hidden_layers`) from base weight tensor metadata.
3. Validate expert conversion before model init and fail fast if all experts are empty.
4. Improve shape mismatch errors to include source tensor key and target model key.
5. Add unit tests for inferred config metadata and empty expert detection.
