# GGUF → OD-MoE Converter

Converts monolithic GGUF models to per-expert safetensors files for memory-mapped on-demand loading.

## Purpose

Traditional MoE models store all experts in a single file, requiring the entire model to be loaded into memory. For massive models like Kimi-K2.5 (375GB), this requires 512GB+ RAM even with quantization.

This converter splits the model into:
- **Base model** (~35GB): Embeddings, attention layers, norms (always resident in memory)
- **Individual experts** (~30MB each): 10,752 separate files that can be memory-mapped and loaded on-demand

This enables running 375GB models on 192GB systems by keeping only a working set of experts in memory (~11GB) and streaming the rest from NVMe.

## Quick Start

### 1. Install Dependencies

**Note:** This guide uses `python` for brevity. On some systems (especially macOS), you may need to use `python3` instead.

**Prerequisites:** Python 3.8+

**Recommended:** Use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install gguf safetensors tqdm numpy
```

### 2. Create a Toy Model for Testing

```bash
python -m convert.create_toy_model \
  --output convert/toy_model.gguf \
  --layers 2 \
  --experts 4 \
  --dim 512
```

This creates a ~2MB test model with the same structure as Kimi-K2.5.

### 3. Convert GGUF to OD-MoE Format

```bash
python -m convert.gguf_to_od_moe \
  --input convert/toy_model.gguf \
  --output /path/to/output \
  --num-layers 2 \
  --num-experts 4
```

For the full Kimi-K2.5 model:

```bash
python -m convert.gguf_to_od_moe \
  --input /Volumes/MacStudio/Kimi-K2.5.gguf \
  --output /Volumes/MacStudio/kimi_od_moe \
  --num-layers 28 \
  --num-experts 384
```

**⚠️ Warning (Mac Studio w/ NVMe):** Full conversion takes ~30-45 minutes and writes ~375GB to disk. Time may vary based on disk speed.

## Output Structure

```
output_dir/
├── base_model/
│   ├── embeddings.safetensors          # Token embeddings (vocab_size × dim)
│   ├── attention_layers.safetensors    # All attention weights for 28 layers
│   ├── norms.safetensors               # Layer norms
│   ├── lm_head.safetensors             # Output projection
│   └── metadata.json                    # Component metadata
│
└── experts/
    ├── layer_00_expert_000.safetensors  # Layer 0, Expert 0 (w1, w2, w3)
    ├── layer_00_expert_001.safetensors
    ├── ...
    ├── layer_27_expert_383.safetensors  # Layer 27, Expert 383
    └── registry.json                     # Expert metadata (paths, sizes, indices)
```

### Base Model Components

**embeddings.safetensors** (~800MB for Kimi-K2.5)
- `token_embd.weight`: Token embeddings [vocab_size, dim]

**attention_layers.safetensors** (~30GB for Kimi-K2.5)
- `blk.{N}.attn_q.weight`: Query projection
- `blk.{N}.attn_k.weight`: Key projection
- `blk.{N}.attn_v.weight`: Value projection
- `blk.{N}.attn_output.weight`: Output projection
- `blk.{N}.ffn.gate.weight`: MoE router/gate

**norms.safetensors** (~4GB for Kimi-K2.5)
- `blk.{N}.attn_norm.weight`: Attention layer norm
- `blk.{N}.ffn_norm.weight`: FFN layer norm

**lm_head.safetensors** (~800MB for Kimi-K2.5)
- `output.weight`: Final language model head

### Expert Files

Each expert file contains the three weight matrices for a gated MLP:
- `w1.weight`: Gate projection [dim, ffn_dim]
- `w2.weight`: Down projection [ffn_dim, dim]
- `w3.weight`: Up projection [dim, ffn_dim]

For Kimi-K2.5:
- 28 layers × 384 experts = **10,752 expert files**
- Each file ~30MB
- Total: ~325GB

### Expert Registry

`experts/registry.json` contains metadata for all experts:

```json
{
  "layer_00_expert_000": {
    "path": "experts/layer_00_expert_000.safetensors",
    "size": 31457280,
    "layer": 0,
    "expert_id": 0,
    "tensors": ["w1.weight", "w2.weight", "w3.weight"]
  },
  ...
}
```

This registry enables fast expert lookup without scanning the filesystem.

## Memory-Mapped Loading

The converted files are designed for zero-copy loading using MLX:

```python
import mlx.core as mx
from safetensors import safe_open

# Load base model (always resident)
with safe_open("base_model/embeddings.safetensors", framework="mlx") as f:
    embeddings = f.get_tensor("token_embd.weight")  # Zero-copy

# Load expert on-demand
with safe_open("experts/layer_00_expert_042.safetensors", framework="mlx") as f:
    w1 = f.get_tensor("w1.weight")  # Zero-copy from mmap
```

## Performance

**Conversion (one-time cost):**
- Toy model (2 layers, 4 experts): ~1 second
- Kimi-K2.5 (28 layers, 384 experts): ~30-45 minutes

**Disk Space:**
- Input: 375GB GGUF file
- Output: 375GB (same size, different format)
- **Temporary space:** ~400GB peak (keep both during conversion)

**Expert Loading (runtime):**
- Cold expert (from NVMe): ~5ms
- Hot expert (LRU cache): ~0.1ms
- Working set: 8 experts × 28 layers × 30MB = ~6.7GB

## Validation

Run the test suite from the project root directory to verify conversion correctness:

```bash
# From project root: /Users/yuzucchi/Projects/mlx-od-moe/

# Unit tests
pytest tests/test_gguf_parser.py -v
pytest tests/test_base_extraction.py -v
pytest tests/test_expert_extraction.py -v

# Integration test
pytest tests/test_converter_integration.py -v
```

All tests should pass on the toy model.

## Troubleshooting

### "Not a GGUF file" Error

**Cause:** Input file is not a valid GGUF format.

**Solution:** Verify the file with Python:
```bash
python -c "import gguf; reader = gguf.GGUFReader('file.gguf'); print(reader.fields.keys())"
```

### Missing Expert Tensors

**Cause:** GGUF uses different tensor naming convention than expected.

**Solution:** Check actual tensor names:
```bash
python -c "import gguf; reader = gguf.GGUFReader('file.gguf'); print([t.name for t in reader.tensors if 'expert' in t.name][:10])"
```

Update `extract_experts()` to match your model's naming scheme.

### Out of Disk Space

**Cause:** Conversion requires ~2× model size temporarily.

**Solution:** Free up space or use `--output` on a larger drive:
```bash
python -m convert.gguf_to_od_moe \
  --input ~/models/kimi.gguf \
  --output /Volumes/ExternalSSD/kimi_od_moe
```

## Architecture Notes

This converter is part of the **mlx-od-moe** project, which implements on-demand MoE loading for Apple Silicon.

**Key insights:**
1. Apple Silicon has **unified memory** - no GPU/CPU split
2. The bottleneck is **working set size**, not data movement
3. NVMe on Mac Studio (~7GB/s) is fast enough to stream experts on-demand
4. Memory-mapped safetensors enable zero-copy loading from SSD

**Related components:**
- `mlx_od_moe/expert_store.py`: LRU cache + memory-mapped expert loading
- `mlx_od_moe/shadow_model.py`: Prefetch predictor (4 layers ahead)
- `mlx_od_moe/od_moe_layer.py`: On-demand expert loading during inference

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full system design.

## License

MIT
