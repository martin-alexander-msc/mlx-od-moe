# Task

User needs to run `qwen3-coder-next:q4_K_M` GGUF with limited disk.

Problem:
- full expert conversion to fp16 produces ~150GiB outputs and exhausts disk.

Requested outcome:
- avoid full expert materialization while still running OD-MoE inference.
