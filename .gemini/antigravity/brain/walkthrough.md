# Walkthrough

1. Added completion diagnostics in `server.py`:
   - `debug_tokens` now returns prompt/generated token IDs and HEX values,
   - response includes `stop_reason` + `stop_token_id`,
   - `echo_prompt` support for non-stream responses.
2. Added tokenizer provenance visibility:
   - `tokenizer_source` tracked globally,
   - included in `/health` and debug completion payloads.
3. Hardened GGUF tokenizer loading in `gguf_tokenizer.py`:
   - robust extraction of embedded `tokenizer.huggingface.json` across bytes,
     uint8 arrays, and list forms,
   - strict mode when HF JSON exists but tokenizer build fails.
4. For Qwen3Next mode, server now requests strict GGUF tokenizer loading to
   avoid silent tokenizer fallback when embedded HF JSON is unusable.
5. Added tests:
   - `tests/test_server_debug.py` for debug payload, HEX IDs, and echo behavior,
   - `tests/test_gguf_tokenizer_utils.py` for tokenizer text coercion helper.
6. Updated README docs for `debug_tokens` and `echo_prompt`.
