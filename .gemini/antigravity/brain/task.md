# Task

Fix Qwen3 GGUF output quality after tokenizer autoload.

Problem:
- tokenizer autoload made text non-empty, but responses were still garbled and
  included control-token artifacts.
- Qwen3 GGUF uses `tokenizer.ggml.pre=qwen2` and GGUF stop IDs that were not
  fully respected in runtime generation.

Requested outcome:
- apply correct Qwen2-style pretokenization and stop generation on GGUF EOS/EOT
  metadata so completions are coherent.
