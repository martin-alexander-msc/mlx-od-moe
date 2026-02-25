# Implementation Plan

## Goal
Add robust inference diagnostics and tokenizer-source reliability checks.

## Files to Modify
- `mlx_od_moe/server.py`
- `mlx_od_moe/gguf_tokenizer.py`
- `tests/test_server_debug.py`
- `tests/test_gguf_tokenizer_utils.py`
- `README.md`

## Approach
1. Add completion debug controls in `server.py`:
   - `debug_tokens` response fields (`prompt/generated IDs` + HEX),
   - `stop_reason` and `stop_token_id`,
   - `echo_prompt` control (default generated text only),
   - `tokenizer_source` surfaced in `/health`.
2. Harden GGUF tokenizer loading in `gguf_tokenizer.py`:
   - robust text extraction from `tokenizer.huggingface.json` field variants,
   - strict mode to prevent silent fallback when embedded HF JSON exists but
     tokenizer build fails.
3. Add tests for:
   - debug payload structure and echo behavior,
   - tokenizer text coercion helper behavior.
4. Update README docs for new debug/echo request options.
5. Validate with compile checks and direct execution of new test functions.
