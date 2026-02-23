# Walkthrough

1. Investigated mismatch between converter docs and server loader.
2. Confirmed converter writes `base_model/*.safetensors` + `experts/*.safetensors`.
3. Confirmed server used `mx.load()` on one path only.
4. Added `weight_loader.py` to:
   - load base weights from file or directory,
   - remap GGUF tensor names to model keys,
   - transpose 2D tensors when needed for shape compatibility.
5. Updated `server.py` to initialize OD-MoE first, then load remapped base weights.
6. Added focused unit tests for new loader behavior.
7. Updated README quick start to use converted folder layout paths.
