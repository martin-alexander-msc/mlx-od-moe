# Walkthrough

1. Added converter output dtype option with default `float16`.
2. Updated tensor read/dequant path to cast to selected dtype.
3. Threaded dtype through base model and expert extraction.
4. Added unit tests to validate default fp16 and explicit fp32 behavior.
5. Updated README conversion example to include `--output-dtype float16`.
6. Fixed misplaced base-model save block so extraction writes files correctly.
