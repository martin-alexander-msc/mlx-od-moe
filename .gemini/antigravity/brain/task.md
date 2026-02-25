# Task

Fix the remaining Qwen3Next gibberish-output issue after tokenizer and mapping
improvements.

Problem:
- output still appears incoherent despite correct tokenizer loading and
  base/expert shape mapping.
- current preprocessing always applies `+1.0` norm shift for selected Qwen3Next
  norm tensors, which can over-shift already-runtime-scaled weights.

Requested outcome:
- make Qwen3Next norm shift conditional based on observed tensor ranges,
  add startup diagnostics to show whether shift was applied, and add tests for
  both zero-centered and already-shifted norm inputs.
