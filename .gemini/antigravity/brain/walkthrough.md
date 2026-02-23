# Walkthrough

1. Inspected user-provided GGUF blob and extracted metadata/tensor names.
2. Confirmed packed expert naming pattern (`ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`).
3. Added packed-layout extraction logic with expert-axis slicing.
4. Added metadata fallback for `llama.expert_count` and default auto-detection for layer/expert counts.
5. Included `ffn_gate_inp` in base model extraction for router weights.
6. Added a converter guard that rejects quantized packed payloads with a clear error.
7. Added tests for packed MoE extraction and quantized guard behavior.
