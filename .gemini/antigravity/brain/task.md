# Task

Align Qwen3Next OD-MoE routing behavior with Qwen3 reference semantics.

Problem:
- tokenizer and stop-token fixes improved outputs but generations remained
  incoherent.
- MoE routing in `ODMoELayer` used logits-based top-k with always-on
  renormalization, which diverged from Qwen3Next routing semantics.

Requested outcome:
- switch OD routing to precise probability-based top-k, make top-k
  renormalization configurable via `norm_topk_prob`, and add guardrails/tests.
