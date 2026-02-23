# Task

User approved additional changes after shape mismatch:
- Make config naming more generic (`KimiODMoEConfig` -> generic naming).
- Diagnose why converted artifacts still fail.

Observed runtime artifacts showed:
- Base embedding shape differs from hardcoded Kimi defaults.
- All expert files were empty (16B, no tensors), indicating conversion incompatibility for provided model.

Requested outcome:
- Improve runtime behavior and error clarity for incompatible conversions.
