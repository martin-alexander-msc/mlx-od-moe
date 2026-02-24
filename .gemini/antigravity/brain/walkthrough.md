# Walkthrough

1. Added a new `safetensors_to_od_moe` converter module.
2. Implemented safetensors file/directory loading and metadata inference for layers/experts.
3. Added base/expert splitting and registry generation compatible with existing runtime expectations.
4. Supported explicit and packed expert tensor naming styles.
5. Added end-to-end test using synthetic safetensors input.
6. Updated README with safetensors conversion usage.
