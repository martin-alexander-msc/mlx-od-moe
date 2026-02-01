# mlx-od-moe

On-Demand Mixture of Experts for Apple Silicon — run 375GB models in 192GB RAM.

## Quick Start

```bash
# Install
pip install -e .

# Convert model
python -m mlx_od_moe.convert \
  --input models/Kimi-K2.5.gguf \
  --output /Volumes/Storage/experts

# Run server
python -m mlx_od_moe.server \
  --expert-dir /Volumes/Storage/experts \
  --base-weights /Volumes/Storage/base_model.safetensors \
  --port 8080
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

**Key insight:** Apple Silicon's unified memory + NVMe = memory-mapped experts work like L3 cache.

## Hardware Requirements

### Validation (M4 Max 36GB)
- Model: Qwen2-57B-A14B
- Context: 32K tokens
- Speed: ~45 tok/s

### Production (Mac Studio 192GB)
- Model: Kimi-K2.5
- Context: 256K tokens
- Speed: ~70 tok/s

## Development

```bash
# Run tests
pytest tests/

# Benchmark
python tests/benchmark_expert_fetch.py
python tests/benchmark_inference.py

# Monitor performance
curl http://localhost:8080/health
```

## Shadow Model Training

Train the expert predictor for prefetch optimization:

```bash
# Quick example with dummy data
python examples/train_shadow_example.py

# Production: Collect training data from pretrained model
python -m mlx_od_moe.training.collect_training_data \
  --model-path /path/to/model \
  --expert-dir /path/to/experts \
  --num-samples 50000 \
  --output training_data.npz

# Train shadow model
python -m mlx_od_moe.training.train_shadow \
  --data training_data.npz \
  --output shadow_model.safetensors \
  --epochs 20
```

**Performance Targets:**
- Top-8 accuracy: >90% (enables >90% cache hit rate)
- Latency: <1ms (4 layers × 1ms = 4ms prefetch window)
- Model size: <500MB

## Components

- `expert_store.py` - LRU cache + memory-mapped storage
- `shadow_model.py` - Expert predictor for prefetch
- `od_moe_layer.py` - On-demand expert loading
- `model.py` - Full Kimi-K2.5 integration
- `convert/` - GGUF → OD-MoE format converter
- `server.py` - Flask API server
- `training/` - Shadow model training pipeline

## License

MIT
