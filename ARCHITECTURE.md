# ARCHITECTURE.md - mlx-od-moe

## Executive Summary

**Problem:** Kimi-K2.5 is 375GB. Even with 4-bit quantization, traditional loading requires 512GB RAM.

**Solution:** On-Demand MoE loading using Apple Silicon's unified memory + NVMe as L3 cache.

**Result:** 80GB resident + 325GB memory-mapped on SSD = runs on 192GB Mac Studio.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
│                 (Clawdbot, API, etc.)                        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP /v1/completions
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Server                              │
│                   (server.py)                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Request Handler                                     │   │
│  │  - Tokenization                                      │   │
│  │  - Streaming response                                │   │
│  │  - Stats reporting                                   │   │
│  └────────────────┬─────────────────────────────────────┘   │
└───────────────────┼─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                 KimiODMoEModel                               │
│                   (model.py)                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Base Model (Always Resident ~35GB)                  │   │
│  │  - Embeddings (102K vocab × 4096 dim)                │   │
│  │  - 28 × Attention layers                             │   │
│  │  - Layer norms                                       │   │
│  │  - LM head                                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  28 × Transformer Layers                             │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Layer N                                        │  │   │
│  │  │  1. Multi-Head Attention (MLA)                  │  │   │
│  │  │  2. OD-MoE FFN ──────────────────┐              │  │   │
│  │  └──────────────────────────────────┼──────────────┘  │   │
│  └───────────────────────────────────────┼─────────────────┘
└──────────────────────────────────────────┼──────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        │                                   ▼                               │
        │                        ┌──────────────────────┐                   │
        │                        │  ODMoELayer          │                   │
        │                        │  (od_moe_layer.py)   │                   │
        │                        └──────────┬───────────┘                   │
        │                                   │                               │
        │         ┌─────────────────────────┼─────────────────────────┐     │
        │         │                         │                         │     │
        │         ▼                         ▼                         ▼     │
        │  ┌────────────┐          ┌─────────────────┐       ┌────────────┐│
        │  │   Router   │          │  Expert Store   │       │   Shadow   ││
        │  │   (gate)   │          │ (expert_store)  │       │   Model    ││
        │  │            │          │                 │       │  (shadow_  ││
        │  │ 4096→384   │          │  LRU Cache      │       │   model)   ││
        │  │  ~1.5MB    │          │  (~11GB)        │       │            ││
        │  └────────────┘          │                 │       │ Predictor  ││
        │                          │  ┌──────────┐   │       │  <1ms      ││
        │                          │  │ Expert 0 │   │       │            ││
        │                          │  │ Expert 5 │   │       │ Prefetch   ││
        │                          │  │ Expert 12│   │       │ Trigger    ││
        │                          │  │   ...    │   │       │            ││
        │                          │  │(8 active)│   │       └────────────┘│
        │                          │  └──────────┘   │                     │
        │                          │                 │                     │
        │                          │  Registry       │                     │
        │                          │  (10,752 files) │                     │
        │                          └─────────┬───────┘                     │
        │                                    │                             │
        └────────────────────────────────────┼─────────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │     NVMe Storage             │
                              │   (325GB memory-mapped)      │
                              │                              │
                              │  experts/                    │
                              │    layer_00_expert_000.st    │
                              │    layer_00_expert_001.st    │
                              │    ...                       │
                              │    layer_27_expert_383.st    │
                              │                              │
                              │  (10,752 files × 30MB each)  │
                              └──────────────────────────────┘
```

---

## Component Details

### 1. Expert Store

**File:** `mlx_od_moe/expert_store.py`

**Responsibilities:**
- Index all 10,752 expert files without loading them
- Maintain LRU cache of mx.arrays (working set)
- Async prefetch experts into cache
- Evict least-recently-used when cache full
- Thread-safe access (lock on cache operations)

**Data Flow:**
```
fetch(layer, expert)
  ↓
  In LRU cache? ──YES──> Return cached mx.array (fast path)
  ↓ NO
  Load from memory-mapped file
  ↓
  Convert np.memmap → mx.array
  ↓
  Add to LRU cache (evict if needed)
  ↓
  Return mx.array
```

**Key Methods:**
- `fetch(layer, expert) -> mx.array` - Get expert weights (cache or load)
- `prefetch(layer, experts)` - Async load into cache
- `get_stats() -> dict` - Cache hit rate, working set size, etc.

**Memory Layout:**
```
expert_registry: dict[str, dict]  ~10KB (metadata only)
lru_cache: OrderedDict[str, mx.array]  ~11GB (8 experts × 28 layers × 50MB)
prefetch_executor: ThreadPoolExecutor  (4 workers, minimal overhead)
```

---

### 2. Shadow Model

**File:** `mlx_od_moe/shadow_model.py`

**Purpose:** Predict which experts will be needed 4 layers ahead to trigger prefetch early enough to hide SSD latency.

**Architecture:**
```
Input: hidden_states (batch, seq, 4096)
  ↓
Mean pooling over sequence
  ↓
Encoder: Linear(4096 → 1024) + LayerNorm + SiLU
  ↓
4 prediction heads (one per lookahead step)
  ↓
Each head: Linear(1024 → 512) + SiLU + Linear(512 → 384)
  ↓
Top-k selection (k=8)
  ↓
Output: [top8_layer_t+1, top8_layer_t+2, top8_layer_t+3, top8_layer_t+4]
```

**Training:**
- Collect (hidden_state, expert_choice) pairs from pretrained Kimi-K2.5
- Minimize cross-entropy between predicted and actual expert usage
- Target: >90% top-8 accuracy

**Inference:**
- Runs async in parallel with main model
- <1ms latency target
- Predictions queued for OD-MoE layers to consume

---

### 3. OD-MoE Layer

**File:** `mlx_od_moe/od_moe_layer.py`

**Core Logic:**

```python
def __call__(self, x: mx.array) -> mx.array:
    # 1. Route
    router_logits = self.gate(x)  # (batch*seq, 384)
    weights, indices = mx.topk(router_logits, k=8)
    
    # 2. Determine unique experts needed
    unique_experts = mx.unique(indices.flatten()).tolist()
    
    # 3. Load on-demand (THE KEY STEP)
    self.load_experts(unique_experts)  # Fetch from store
    
    # 4. Compute weighted sum of expert outputs
    output = weighted_expert_compute(x, weights, indices)
    
    # 5. Prefetch for next layer
    next_experts = self.shadow_runner.get_predictions_for_layer(self.layer_idx + 1)
    self.expert_store.prefetch(self.layer_idx + 1, next_experts)
    
    return output
```

**Key Insight:** `load_experts()` is fast if already cached (prefetch worked), slow if not (~5ms SSD fetch). Shadow model's job is to make cache hits >90%.

---

### 4. Full Model

**File:** `mlx_od_moe/model.py`

**Two-Phase Initialization:**

```python
# Phase 1: Load base model (attention, norms, embeddings)
model = KimiODMoEModel(config)
model.load_weights("base_model.safetensors")  # ~35GB

# Phase 2: Setup OD-MoE (doesn't load experts yet, just indexes)
model.setup_od_moe(
    expert_dir="/Volumes/MacStudio/experts",
    predictor_path="shadow_model.safetensors"
)
```

**Memory Breakdown:**
- Embeddings: 102K vocab × 4096 × fp16 = 800MB
- Attention (28 layers): ~30GB
- Norms + misc: ~4GB
- **Total base:** ~35GB

- OD-MoE working set: 8 experts × 28 layers × 50MB = ~11GB
- **Total resident:** ~46GB (fits Mac Studio 192GB comfortably)

---

### 5. Converter

**File:** `convert/gguf_to_od_moe.py`

**Input:** `Kimi-K2.5.gguf` (375GB monolithic file)

**Output:**
```
base_model/
  embeddings.safetensors
  attention_layers.safetensors
  norms.safetensors

experts/
  layer_00_expert_000.safetensors  (~30MB)
  layer_00_expert_001.safetensors
  ...
  layer_27_expert_383.safetensors

  (10,752 files, 325GB total)
```

**Process:**
1. Parse GGUF metadata
2. Extract base model tensors → safetensors
3. For each layer's MoE block:
   - Extract 384 expert weights (w1, w2, w3)
   - Save each expert as separate file
4. Generate expert registry JSON (paths + sizes)

---

### 6. Server

**File:** `mlx_od_moe/server.py`

**Endpoints:**

#### `POST /v1/completions`
```json
{
  "prompt": "Explain quantum computing",
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Streaming Response:**
```
data: {"token": "Quantum"}
data: {"token": " computing"}
...
data: {"done": true, "stats": {...}, "time": 12.3}
```

#### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "expert_cache_stats": {
    "cache_hits": 1250,
    "cache_misses": 87,
    "hit_rate": 0.935,
    "working_set_experts": 224
  }
}
```

---

## Data Flow: Single Forward Pass

```
1. Input: "Explain quantum computing" (5 tokens)
   ↓
2. Embeddings: 5 → (1, 5, 4096)
   ↓
3. Layer 0:
   ├─ Attention: (1, 5, 4096) → (1, 5, 4096)
   ├─ Shadow Model: Predict experts for layers 1-4 → [expert_ids]
   ├─ Router: Select top-8 experts for current tokens
   ├─ Expert Store: Fetch experts (cache hit or SSD load)
   └─ OD-MoE: Apply experts → (1, 5, 4096)
   ↓
4. Layers 1-27: (repeat)
   ↓
5. Final norm + LM head: (1, 5, 4096) → (1, 5, 102400)
   ↓
6. Sample next token: "quantum"
   ↓
7. Repeat for next token...
```

**Timing (Mac Studio target):**
- Attention: ~10ms
- Shadow predict: <1ms
- Router: ~2ms
- Expert fetch (cache hit): ~0.1ms
- Expert fetch (cache miss): ~5ms
- Expert compute: ~15ms

**Total per token:** ~30ms = ~33 tok/s (without prefetch)
**With prefetch:** ~15ms = ~70 tok/s (SSD latency hidden)

---

## Memory Management Strategy

### Tier 1: Always Resident (~35GB)
- Base model components that every forward pass needs
- MLX allocates once, never freed

### Tier 2: Working Set (~11GB)
- LRU cache of expert mx.arrays
- Hot-swapped on every forward pass
- Cache size tunable (default: 48GB limit)

### Tier 3: Memory-Mapped (~325GB)
- Expert files on NVMe
- np.memmap for zero-copy read
- Converted to mx.array only when moving to Tier 2

### Tier 4: Cold Storage (not used at runtime)
- GGUF source files
- Conversion artifacts

---

## Performance Optimization Techniques

### 1. Prefetch Overlap
```
Layer N processing
  └─> Shadow predicts experts for N+1, N+2, N+3, N+4
       └─> Expert store starts async loading
            └─> By time layer N+1 starts, experts already cached
```

### 2. Batch Expert Loading
Instead of fetching 8 experts sequentially (8 × 5ms = 40ms), load in parallel:
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_expert, idx) for idx in expert_ids]
    experts = [f.result() for f in futures]
# Parallel load: ~10ms (limited by NVMe queue depth)
```

### 3. Router Caching
For repeated tokens (e.g., "the" appears 100 times), cache router decisions:
```python
router_cache[token_id] = top_k_experts  # Saves ~2ms per cache hit
```

### 4. KV Cache Compression (MLA)
Multi-head Latent Attention reduces KV cache from:
- Standard: 28 layers × 32 heads × 256K tokens × 128 dim × fp16 = **58GB**
- MLA: 28 layers × 8 compressed heads × 256K × 64 dim × fp16 = **7GB**

**8x reduction** enables 256K context in 192GB system.

---

## Failure Modes & Recovery

### OOM During Inference
**Cause:** KV cache grew beyond available memory

**Detection:** Monitor `psutil.virtual_memory().percent`

**Recovery:**
```python
if memory_usage > 85%:
    # Trim KV cache (drop oldest 25% of context)
    kv_cache = kv_cache[:, :, -int(0.75*seq_len):, :]
```

### Expert Fetch Timeout
**Cause:** SSD I/O stall (rare on NVMe, more common on SATA SSD)

**Detection:** fetch() takes >100ms

**Recovery:**
```python
try:
    expert = store.fetch(layer, idx, timeout=0.1)
except TimeoutError:
    # Use fallback: average of currently cached experts
    expert = mx.mean([e for e in active_experts.values()], axis=0)
```

### Cache Thrashing
**Cause:** Shadow model predictions inaccurate, frequent cache misses

**Detection:** `cache_hit_rate < 0.7`

**Recovery:**
```python
if cache_hit_rate < 0.7:
    # Increase cache size (if memory available)
    store.cache_size *= 1.5
    # Or: retrain shadow model
```

---

## Testing Strategy

### Unit Tests

#### Expert Store
```python
def test_lru_eviction():
    # Fill cache beyond limit
    # Verify oldest expert evicted
    
def test_thread_safety():
    # Concurrent fetches from multiple threads
    # Verify no race conditions
```

#### Shadow Model
```python
def test_prediction_accuracy():
    # Run on validation set
    # Verify top-8 accuracy >90%
    
def test_latency():
    # Measure prediction time
    # Verify <1ms on M4 Max
```

#### OD-MoE Layer
```python
def test_expert_selection():
    # Verify router selects correct top-k
    
def test_prefetch_trigger():
    # Verify shadow model called
    # Verify expert store prefetch queued
```

### Integration Tests

```python
def test_full_forward_pass():
    model = load_model()
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    logits = model(input_ids)
    assert logits.shape == (1, 5, 102400)
    
def test_32k_context():
    long_input = mx.random.randint(0, 102400, (1, 32768))
    logits = model(long_input)
    # Verify no OOM, reasonable speed
    
def test_generation_quality():
    prompt = "Explain quantum computing"
    output = model.generate(prompt, max_tokens=100)
    # Verify coherent output
```

### Performance Benchmarks

```python
def benchmark_expert_fetch():
    # Cold cache: measure SSD read speed
    # Hot cache: measure memory access speed
    
def benchmark_tokens_per_second():
    # 1K context
    # 32K context
    # 256K context (Mac Studio only)
    
def benchmark_cache_hit_rate():
    # Run on representative workload
    # Measure hit rate over 10K tokens
```

---

## Dual-Device Architecture (Future)

```
┌─────────────────────────────────────────┐
│  Mac Studio (192GB)                     │
│  ┌───────────────────────────────────┐  │
│  │  Kimi OD-MoE (Text)               │  │
│  │  - 256K context                   │  │
│  │  - 70 tok/s                       │  │
│  │  - Flask server :8080             │  │
│  └───────────────┬───────────────────┘  │
└──────────────────┼──────────────────────┘
                   │ Thunderbolt 4 (40Gb/s)
                   │ ~1ms latency
┌──────────────────┼──────────────────────┐
│  M4 Max MBP (36GB)                      │
│  ┌───────────────▼───────────────────┐  │
│  │  Qwen2-VL-7B (Vision)             │  │
│  │  - Image → text descriptions      │  │
│  │  - 25 tok/s                       │  │
│  │  - Flask server :5000             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Use case:** User sends image + prompt → Studio routes to M4 Max for vision → combines vision + text context → generates response.

---

## Success Metrics

### M4 Max Validation
- [x] Qwen2-57B converted to OD-MoE format
- [x] Memory usage <30GB during inference
- [x] 45+ tok/s at 32K context
- [x] Cache hit rate >85%
- [x] No OOM over 1 hour continuous inference

### Mac Studio Production
- [ ] Kimi-K2.5 converted to OD-MoE format
- [ ] Memory usage <80GB during inference
- [ ] 70+ tok/s at 256K context
- [ ] Cache hit rate >90%
- [ ] Vision pipeline integration working

---

*This architecture trades SSD bandwidth for RAM capacity. On Apple Silicon with NVMe, it's a profitable trade.*
