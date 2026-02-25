# Task

Diagnose remaining Qwen3Next incoherent output with explicit token-level debug
visibility and tokenizer hardening.

Problem:
- output still appeared semantically broken after major tensor/mapping fixes.
- debugging lacked raw token IDs and tokenizer source visibility.
- potential tokenizer mismatch still possible due fallback handling.

Requested outcome:
- add token-level diagnostics (`debug_tokens` with HEX IDs), add `echo_prompt`
  control, expose tokenizer source, and harden GGUF tokenizer loading to prefer
  exact embedded HF JSON parsing.
