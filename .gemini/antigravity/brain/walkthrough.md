# Walkthrough

1. Investigated remaining incoherent outputs after prior tokenizer/routing/norm
   fixes.
2. Identified mismatch in linear-attention fusion ordering:
   `attn_qkv + attn_gate` was appended flat, while
   `Qwen3NextGatedDeltaNet.fix_query_key_value_ordering()` expects per-key-head
   `[q,k,v,z]` chunk layout.
3. Updated `_preprocess_qwen3next_source_tensors()` in `server.py` to:
   - interleave qkv/gate chunks per `linear_num_key_heads`,
   - pass `linear_num_key_heads` from resolved Qwen3Next config,
   - keep prior norm sanitize shift behavior.
4. Norm sanitize shift remains applied as `+1.0` for:
   - `blk.*.attn_norm.weight`
   - `blk.*.post_attention_norm.weight`
   - `blk.*.attn_q_norm.weight`
   - `blk.*.attn_k_norm.weight`
   - `output_norm.weight`
5. Extended regression test `tests/test_server_preprocess.py` to verify:
   - interleaved qkvz ordering on deterministic data,
   - norm-shift behavior,
   - unchanged non-target tensors.
6. Validation:
   - `uv run python3 -m py_compile mlx_od_moe/server.py tests/test_server_preprocess.py`
   - direct runtime sanity scripts confirmed:
     - norm means shift as expected (`0.0->1.0`, `0.5->1.5`, etc.),
     - real layer-0 fusion now interleaves gate chunk within each head block.
