# Walkthrough

1. Verified GGUF metadata for qwen3next (48 layers, 512 experts, packed MoE tensors).
2. Implemented `GGUFOnDemandExpertStore` to dequantize expert slices on fetch.
3. Added server CLI mode `--gguf-experts` and runtime wiring in model setup.
4. Added GGUF metadata inference path for config overrides in GGUF mode.
5. Added converter flags `--base-only` / `--experts-only`.
6. Updated README with low-disk workflow (base-only conversion + live GGUF experts).
