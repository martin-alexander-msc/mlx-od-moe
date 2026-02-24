# Walkthrough

1. Reproduced BF16 failure in `safe_open(..., framework='numpy')`.
2. Added fallback raw safetensors parser when BF16 TypeError occurs.
3. Implemented dtype-tag decoding for standard numeric types and BF16.
4. Added BF16 decoder unit test.
5. Validated fallback loader on real Qwen3-Next 4bit shard.
