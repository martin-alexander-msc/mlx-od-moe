# Walkthrough

1. Investigated suspected expert-axis mismatch by comparing runtime GGUF slice
   dequantization against full-tensor dequantization for Qwen3Next packed expert
   tensors.
   - Result: runtime slicing was correct for gate/up/down packed tensors.
2. Verified converted base tensor payloads matched source GGUF exactly for key
   attention/SSM/shared-expert tensors to rule out base/expert mismatch.
3. Implemented routing alignment in `ODMoELayer`:
   - introduced `_route_experts`,
   - switched to `softmax(..., precise=True)`,
   - selected top-k from probabilities via `argpartition`,
   - made top-k renormalization configurable with `norm_topk_prob`,
   - added non-finite router-logit validation.
4. Wired `Qwen3NextODConfig.norm_topk_prob` into `ODMoELayer` construction.
5. Added unit tests covering:
   - top-k normalization toggle behavior,
   - rejection of non-finite router logits.
6. Validation:
   - `py_compile` passed for modified files,
   - direct runtime sanity check confirmed normalized vs raw top-k score sums.
   - full pytest execution is currently blocked in this environment because
     `pytest` is not installed in `.venv`.
