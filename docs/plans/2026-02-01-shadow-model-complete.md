# Shadow Model Complete Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build production-ready Shadow Model with <1ms inference, >90% top-8 accuracy, comprehensive training pipeline

**Architecture:** MLP predictor (4096→1024→384×4) trains on (hidden_state, expert_choice) pairs to predict expert usage 4 layers ahead

**Tech Stack:** MLX (lazy evaluation framework), safetensors, numpy, pytest

**IMPORTANT NOTE:** This plan uses `mx.eval()` which is MLX's function to force lazy computation - this is NOT Python's `eval()` for code execution. MLX builds computation graphs that only execute when explicitly evaluated.

See: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html

---

## Quick Start

```bash
# Run complete pipeline
python examples/train_shadow_example.py

# Expected output:
# - Top-8 accuracy: ~85-95% (dummy data)
# - Latency: <1ms
# - Model size: ~4MB
```

---

## Task 1: ExpertPredictor Optimization

**Create test file:** `tests/test_shadow_model.py`

Run test:
```bash
pytest tests/test_shadow_model.py::test_predictor_latency -v -s
```

Optimize `mlx_od_moe/shadow_model.py`:
- Remove Dropout from encoder
- Simplify prediction heads
- Add save/load methods

Commit:
```bash
git add tests/test_shadow_model.py mlx_od_moe/shadow_model.py
git commit -m "feat: optimize ExpertPredictor to <1ms"
```

---

## Task 2: Training Data Collection

**Create:** `mlx_od_moe/training/collect_training_data.py`

Implements `collect_expert_usage()` function that:
- Runs model forward passes (or generates dummy data)
- Records hidden states before MoE layers
- Records which experts router selects
- Saves to compressed .npz format

Commit:
```bash
git add mlx_od_moe/training/
git commit -m "feat: implement training data collection"
```

---

## Task 3: Training Script

**Create:** `mlx_od_moe/training/train_shadow.py`

Implements `train_shadow_model()` function:
- Load training data from .npz
- Train with binary cross-entropy loss
- Compute top-1, top-4, top-8 accuracy
- Save trained model to safetensors

Commit:
```bash
git add mlx_od_moe/training/train_shadow.py
git commit -m "feat: implement shadow model training with metrics"
```

---

## Task 4: Async ShadowRunner

**Modify:** `mlx_od_moe/shadow_model.py` (ShadowRunner class)

Optimize for async execution:
- Use mx.async_eval for non-blocking predictions
- Improve prediction queue lookup
- Add predictor weight loading

Commit:
```bash
git add mlx_od_moe/shadow_model.py
git commit -m "feat: optimize ShadowRunner with async execution"
```

---

## Task 5: Training Example

**Create:** `examples/train_shadow_example.py`

End-to-end pipeline demonstration:
1. Collect dummy data (5000 samples)
2. Train model (20 epochs)
3. Benchmark latency
4. Print metrics

Run:
```bash
python examples/train_shadow_example.py
```

Commit:
```bash
git add examples/train_shadow_example.py
git commit -m "feat: add shadow model training example"
```

---

## Task 6: Comprehensive Tests

Add to `tests/test_shadow_model.py`:
- Full pipeline test (collect → train → infer)
- Integration test with ExpertStore
- Accuracy computation tests
- Save/load tests

Run:
```bash
pytest tests/test_shadow_model.py -v
```

Commit:
```bash
git add tests/test_shadow_model.py
git commit -m "test: add comprehensive shadow model tests"
```

---

## Task 7: Documentation

Update:
- `README.md` - Add shadow model training section
- `ARCHITECTURE.md` - Update shadow model details
- Create `docs/shadow_model_training.md` - Full guide

Commit:
```bash
git add README.md ARCHITECTURE.md docs/shadow_model_training.md
git commit -m "docs: add shadow model training guide"
```

---

## Task 8: Final Validation

Run complete validation:

```bash
# All tests
pytest tests/test_shadow_model.py -v

# Example
python examples/train_shadow_example.py

# Check targets
# [x] Latency <1ms
# [x] Model size <500MB
# [x] Tests passing
# [x] Example runs successfully
```

Final commit:
```bash
git add -A
git commit -m "feat: complete shadow model implementation"
```

Wake clawdbot:
```bash
clawdbot gateway wake 'Shadow Model complete' --mode now
```

---

## Success Criteria

- [x] ExpertPredictor <1ms latency
- [x] Training data collection (dummy + real modes)
- [x] Training script with cross-entropy loss
- [x] Top-1/4/8 accuracy metrics
- [x] Async ShadowRunner
- [x] Save/load with safetensors
- [x] Model size <500MB
- [x] Integration tests
- [x] Training example
- [x] Documentation

**Production Next Steps:**
1. Collect 50K+ samples from real model
2. Train and validate >90% top-8 accuracy
3. Deploy in server
4. Monitor cache hit rates
