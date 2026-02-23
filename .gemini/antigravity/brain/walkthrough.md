# Walkthrough

1. Inspected converted artifacts under `Volumes/Storage/experts`.
2. Confirmed base tensor shapes were non-Kimi and all expert files were empty (`tensors: []`).
3. Added generic config naming by introducing `ODMoEConfig` and preserving `KimiODMoEConfig` alias.
4. Added metadata-based config inference from base safetensors shapes.
5. Added explicit expert conversion validation to fail early when all experts are empty.
6. Improved shape mismatch messages to include source and destination tensor keys.
7. Added focused tests for config inference and empty expert detection.
