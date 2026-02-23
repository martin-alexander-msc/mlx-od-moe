# Task

User requested converter fixes for Codextral GGUF.

Investigation found:
- Source GGUF uses packed MoE tensors (`ffn_*_exps`) not explicit `ffn.experts.{i}.w*` keys.
- Prior conversion produced empty expert files.
- Source GGUF blob is quantized packed (uint8 blocks), which current converter cannot faithfully dequantize.

Requested outcome:
- Support packed tensor naming format.
- Surface explicit error for unsupported quantized packed GGUF payloads.
