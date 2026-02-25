# Walkthrough

1. Updated `server.py` Qwen3Next preprocessing:
   - kept head-wise `attn_qkvz` fusion logic unchanged,
   - replaced unconditional norm `+1.0` shift with conditional detection using
     observed norm stats (min/max/mean),
   - added startup log output showing `shift_applied=true/false`.
2. Added/updated preprocessing tests in `tests/test_server_preprocess.py`:
   - verifies zero-centered norms are shifted,
   - verifies already-shifted norms are preserved.
3. Updated README to document automatic norm-shift detection for Qwen3Next
   base preprocessing.
