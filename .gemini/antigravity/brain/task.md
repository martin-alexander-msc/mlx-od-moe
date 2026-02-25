# Task

Fix remaining Qwen3Next GGUF incoherent output after tokenizer, routing, and
norm-shift fixes.

Problem:
- output remained semantically broken.
- additional root cause identified: Qwen3Next linear-attention fusion
  (`attn_qkv + attn_gate`) was concatenated as a flat append instead of
  per-key-head `[q,k,v,z]` interleaving expected by
  `Qwen3NextGatedDeltaNet.fix_query_key_value_ordering`.

Requested outcome:
- apply head-wise interleaving in qkvz fusion and add regression coverage for
  ordering correctness.
