# Implementation Plan

## Goal
Make GGUF-only runtime self-contained by loading tokenizer directly from GGUF metadata.

## Files to Modify
- `mlx_od_moe/server.py`
- `mlx_od_moe/gguf_tokenizer.py` (new)
- `README.md`

## Approach
1. Add GGUF tokenizer loader that reads embedded tokenizer metadata.
2. Support tokenizer sources in priority order:
   - `tokenizer.huggingface.json` (when present)
   - `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` for GPT-2 byte-level BPE.
3. Wire server startup to auto-load tokenizer from `--gguf-experts` when `--tokenizer` is not provided.
4. Keep fallback byte encoding/decoding only as last-resort compatibility path.
5. Document GGUF tokenizer autoload behavior in README.
