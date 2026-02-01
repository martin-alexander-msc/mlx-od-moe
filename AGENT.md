# AGENT.md - mlx-od-moe Development Agent

## Project Overview

**mlx-od-moe** - On-Demand Mixture of Experts implementation for Apple Silicon, optimized for massive context windows (256K+) without requiring 512GB RAM.

## Core Architecture Principles

### Unified Memory Insight
On Apple Silicon, "loading to GPU" is a myth. We have one unified memory pool. The challenge is **working set management**, not data movement.

**Key insight:** Keep only 8 experts per layer resident (~11GB total working set), memory-map the rest from NVMe. The 375GB Kimi-K2.5 becomes 80GB resident + 325GB on SSD.

### Three-Layer Memory Hierarchy
1. **Always resident** (~35GB): Base model (embeddings, attention, norms)
2. **Working set** (~11GB): 8 experts × 28 layers currently active in LRU cache
3. **Memory-mapped** (~325GB): 10,752 expert files on NVMe, accessed on-demand

### Shadow Model for Prefetch
Lightweight MLP predicts which experts will be needed 4 layers ahead. This overlaps SSD I/O with computation, hiding the ~5ms fetch latency.

## Implementation Components

### 1. Expert Store (`mlx_od_moe/expert_store.py`)
- **LRU cache** of mx.arrays (working set)
- **Memory-mapped** safetensors files (cold experts on SSD)
- **Async prefetch** via ThreadPoolExecutor
- **Telemetry** for cache hit rate tracking

**Key constraint:** Must stay thread-safe. File I/O in threads, mx.array creation on main thread only.

### 2. Shadow Model (`mlx_od_moe/shadow_model.py`)
- Predicts top-8 experts for next 4 layers
- Input: hidden states from layer N
- Output: List of expert indices to prefetch
- Target latency: <1ms per prediction

### 3. OD-MoE Layer (`mlx_od_moe/od_moe_layer.py`)
- Router network (always resident, ~1.5MB)
- Loads only selected 8 experts on-demand
- Triggers prefetch for next layer
- Evicts unused experts to stay under cache limit

### 4. Full Model (`mlx_od_moe/model.py`)
- Kimi-K2.5 architecture (28 layers, 384 experts/layer)
- Multi-head Latent Attention (MLA) for KV cache compression
- Integration point for shadow model + expert store

### 5. Converter (`convert/gguf_to_od_moe.py`)
- Splits GGUF into per-expert safetensors files
- Extracts base model (attention + norms) separately
- Enables memory-mapping without loading full 375GB

### 6. Server (`mlx_od_moe/server.py`)
- Flask API with `/v1/completions` endpoint
- Streaming generation
- Cache statistics reporting

## Hardware Targets

### M4 Max (36GB) - Validation Platform
- **Model:** Qwen2-57B-A14B (real MoE, fits in memory)
- **Working set limit:** 30GB (leaves 6GB safety margin)
- **Expected speed:** 45 tok/s at 32K context
- **Expert fetch:** ~5-8ms from NVMe
- **Role:** Validation + vision server (Qwen2-VL-7B)

### Mac Studio M3 Ultra (192GB) - Production Target
- **Model:** Kimi-K2.5 full (375GB total, 80GB resident)
- **Working set limit:** 50GB
- **Expected speed:** 70 tok/s at 256K context
- **Expert fetch:** ~3-5ms from NVMe
- **Role:** Primary inference engine

## Development Phases

### Phase 1: Core Components (Parallel)
- [x] Expert store with LRU cache
- [x] Shadow model predictor
- [x] OD-MoE layer implementation
- [x] GGUF converter

### Phase 2: Integration
- [ ] Wire shadow model → expert store prefetch
- [ ] Full model with 28 OD-MoE layers
- [ ] KV cache management

### Phase 3: Validation (M4 Max)
- [ ] Qwen2-57B conversion to OD-MoE format
- [ ] Memory bounds test (<30GB)
- [ ] Throughput benchmark (32K context)
- [ ] Cache hit rate measurement

### Phase 4: Production (Mac Studio)
- [ ] Kimi-K2.5 conversion
- [ ] 256K context testing
- [ ] Dual-device integration (vision on M4 Max)

## Validation Checklist

Before claiming "done" on any component:

1. **Memory proof**: `psutil` output showing working set <30GB
2. **Performance proof**: Timing logs for expert fetch (<10ms target)
3. **Correctness proof**: Unit test passing
4. **Integration proof**: Works with next component in chain

## Common Pitfalls

### ❌ MLX Threading Bugs
**Problem:** MLX isn't thread-safe for graph operations
**Solution:** Keep threading in numpy/file I/O land, only create mx.arrays on main thread

### ❌ Zero-Copy Assumptions
**Problem:** `mx.array(np.memmap(...))` actually copies data
**Solution:** Use `mx.load()` with safetensors for true zero-copy

### ❌ Cache Thrashing
**Problem:** LRU evicts expert immediately before reuse
**Solution:** Shadow model must predict with >90% accuracy; validate with telemetry

### ❌ Memory Bloat
**Problem:** KV cache grows unbounded on long contexts
**Solution:** MLA compression + periodic cache trimming

## File Naming Conventions

```
experts/
  layer_00_expert_000.safetensors
  layer_00_expert_001.safetensors
  ...
  layer_27_expert_383.safetensors

base_model/
  embeddings.safetensors
  attention_layers.safetensors
  norms.safetensors
```

## Testing Strategy

### Unit Tests
- Expert store: cache eviction, thread safety, memory bounds
- Shadow model: prediction accuracy, latency
- OD-MoE layer: correct expert selection, prefetch triggering

### Integration Tests
- Full forward pass with dummy data
- 32K context inference
- Multi-turn conversation (cache reuse)

### Performance Tests
- Expert fetch latency distribution
- Cache hit rate over 1000 tokens
- Memory high-water mark
- Tokens/sec on M4 Max vs Mac Studio

## Agent Collaboration

When spawning multiple agents, divide work by **component boundaries**:

- **Agent A:** Expert store + telemetry
- **Agent B:** Shadow model + training script
- **Agent C:** OD-MoE layer + router
- **Agent D:** Converter + data pipeline

Each agent gets a clean workspace with clear input/output contracts. No shared state beyond documented APIs.

## Questions to Ask Before Starting

1. **Memory target:** What's the hard limit? (30GB for M4 Max, 50GB for Studio)
2. **Model choice:** Qwen2-57B (validation) or Toy Kimi (unit tests)?
3. **Prefetch strategy:** How many layers ahead? (Default: 4)
4. **Cache size:** Fixed GB or adaptive based on available memory?

## Success Criteria

### M4 Max Validation
- ✅ Qwen2-57B runs with <30GB resident memory
- ✅ 45+ tok/s at 32K context
- ✅ Cache hit rate >85%
- ✅ No OOM errors over 1 hour inference

### Mac Studio Production
- ✅ Kimi-K2.5 runs with <80GB resident memory
- ✅ 70+ tok/s at 256K context
- ✅ Cache hit rate >90%
- ✅ Dual-device vision pipeline working

---

*This is a systems engineering project disguised as ML. Focus on memory management, not model architecture.*
