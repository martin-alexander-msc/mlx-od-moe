# Task

Fix missing tokenizer in GGUF-only runtime mode.

Problem:
- server generated token IDs but returned empty completion text when
  `--tokenizer` was omitted.
- `--gguf-experts` should be sufficient for inference setup.

Requested outcome:
- auto-load tokenizer from GGUF metadata so completions work without external
  tokenizer files.
