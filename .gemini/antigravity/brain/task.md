# Task

User requested support for converting from `.safetensors` input (e.g. MLX HF repos) because GGUF Q4/Q8 conversion dequantization inflates storage and adds complexity.

Requested outcome:
- provide native safetensors conversion path to OD-MoE outputs,
- preserve existing GGUF converter path.
