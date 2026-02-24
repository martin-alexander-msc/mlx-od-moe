# Walkthrough

1. Verified target GGUF contains tokenizer metadata (`tokenizer.ggml.model=gpt2`,
   `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`).
2. Added `gguf_tokenizer.py` with GGUF tokenizer loader and runtime wrapper.
3. Implemented tokenizer build paths:
   - embedded `tokenizer.huggingface.json` when available
   - GPT-2 byte-level BPE from GGUF `tokens` and `merges` fields.
4. Wired server startup to auto-load tokenizer from GGUF in `--gguf-experts`
   mode when `--tokenizer` is omitted.
5. Updated README to document tokenizer autoload for GGUF-only operation.
