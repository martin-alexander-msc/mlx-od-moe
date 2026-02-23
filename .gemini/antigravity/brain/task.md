# Task

User reported server startup failure after GGUF conversion:
- Converter output has `base_model/` directory.
- Server expected a single `base_model.safetensors` path.
- README command pathing was inconsistent with converter output.

Requested outcome:
- Make runtime support converted directory layout.
- Ensure compatible loading behavior for converted tensor names.
- Update documentation to match real usage.
