# Walkthrough

1. Investigated remaining incoherent outputs after prior tokenizer/routing fixes.
2. Identified mismatch with upstream `mlx_lm` sanitize behavior for Qwen3Next:
   certain RMSNorm-style weights are expected as `(gguf_weight + 1.0)`.
3. Updated `_preprocess_qwen3next_source_tensors()` in `server.py` to apply
   `+1.0` to:
   - `blk.*.attn_norm.weight`
   - `blk.*.post_attention_norm.weight`
   - `blk.*.attn_q_norm.weight`
   - `blk.*.attn_k_norm.weight`
   - `output_norm.weight`
   while preserving existing `attn_qkv + attn_gate -> attn_qkvz` fusion.
4. Added regression test `tests/test_server_preprocess.py` to verify both fusion
   and norm-shift behavior.
5. Validation:
   - `uv run python3 -m py_compile mlx_od_moe/server.py tests/test_server_preprocess.py`
   - direct runtime sanity script confirmed norm means shift as expected
     (`0.0->1.0`, `0.5->1.5`, etc.).
