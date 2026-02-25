# Walkthrough

1. Verified Qwen3 GGUF tokenizer metadata and IDs:
   - `tokenizer.ggml.pre=qwen2`
   - `tokenizer.ggml.eos_token_id=151645`
   - `tokenizer.ggml.eos_token_ids=[151645, 151643]` with pad filtering.
2. Extended `gguf_tokenizer.py`:
   - added Qwen2 pretokenization (`Split(regex) + ByteLevel(use_regex=False)`),
   - registered CONTROL tokens as special tokens,
   - added GGUF special-token/stop-id inference helper.
3. Updated server config flow:
   - apply GGUF EOS override into model config,
   - derive and log resolved stop IDs,
   - pass stop IDs into generation calls.
4. Updated both model generate loops to stop on any token in `stop_token_ids`
   (EOS/EOT set), not only a single default ID.
5. Updated README with Qwen3-specific tokenizer/stop-token behavior notes.
