# Task

Stabilize runtime memory behavior for GGUF on-demand experts.

Problem:
- resident memory grew to ~14GiB for tiny requests and up to ~40GiB for longer runs.
- current defaults and retention behavior made the setup impractical on constrained machines.

Requested outcome:
- keep memory bounded and predictable so Qwen3-Next-Coder Q4_K_M is runnable locally.
