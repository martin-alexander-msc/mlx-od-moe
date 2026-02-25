# Task

Fix remaining Qwen3Next GGUF incoherent output after routing changes.

Problem:
- output remained semantically broken after tokenizer and routing fixes.
- root cause identified: Qwen3Next norm tensors need a `+1.0` sanitize step
  during base-weight preprocessing, matching `mlx_lm` behavior.

Requested outcome:
- apply the missing norm shift in Qwen3Next preprocessing and add regression
  coverage.
