# Task

User requested support for Q4 models.

Required outcome:
- converter accepts Q4 quantized GGUF tensors,
- dequantizes to float tensors while extracting base and expert weights,
- keeps packed MoE layout extraction working for Codextral/Mixtral-style keys.
