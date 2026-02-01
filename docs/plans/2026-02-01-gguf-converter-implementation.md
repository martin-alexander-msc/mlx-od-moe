# GGUF Converter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete GGUF to OD-MoE converter with real parsing, safetensors export, and test infrastructure

**Architecture:** Parse GGUF files using gguf-py library, extract base model tensors (embeddings, attention, norms) and per-expert weights (w1, w2, w3), save as safetensors with registry JSON. Include toy model generator for testing.

**Tech Stack:** Python, gguf-py, safetensors, numpy, tqdm, pytest

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add gguf-py dependency**

Add to dependencies array:
```toml
dependencies = [
    "mlx>=0.21.0",
    "mlx-lm>=0.20.0",
    "numpy>=1.24.0",
    "safetensors>=0.4.0",
    "flask>=3.0.0",
    "psutil>=5.9.0",
    "tqdm>=4.66.0",
    "gguf>=0.10.0",
]
```

**Step 2: Commit dependency changes**

```bash
git add pyproject.toml
git commit -m "feat: add gguf and tqdm dependencies for converter"
```

---

## Task 2: Create Toy Model Generator

**Files:**
- Create: `convert/create_toy_model.py`
- Create: `tests/test_toy_model.py`

**Step 1: Write test for toy model creation**

```python
# tests/test_toy_model.py
import pytest
from pathlib import Path
from convert.create_toy_model import create_toy_gguf_model
import gguf


def test_create_toy_model_generates_gguf(tmp_path):
    """Test that toy model creates valid GGUF file"""
    output_path = tmp_path / "toy.gguf"

    create_toy_gguf_model(
        output_path=output_path,
        num_layers=2,
        num_experts=4
    )

    assert output_path.exists()

    # Verify it's a valid GGUF file
    reader = gguf.GGUFReader(output_path)
    assert reader.fields is not None


def test_toy_model_has_expected_tensors(tmp_path):
    """Test that toy model contains expected tensor structure"""
    output_path = tmp_path / "toy.gguf"

    create_toy_gguf_model(
        output_path=output_path,
        num_layers=2,
        num_experts=4,
        dim=512,
        ffn_dim=1024
    )

    reader = gguf.GGUFReader(output_path)
    tensor_names = {tensor.name for tensor in reader.tensors}

    # Check for base model tensors
    assert "token_embd.weight" in tensor_names
    assert "output.weight" in tensor_names

    # Check for layer tensors
    assert "blk.0.attn_q.weight" in tensor_names
    assert "blk.0.attn_norm.weight" in tensor_names

    # Check for expert tensors (layer 0, expert 0)
    assert "blk.0.ffn.experts.0.w1.weight" in tensor_names
    assert "blk.0.ffn.experts.0.w2.weight" in tensor_names
    assert "blk.0.ffn.experts.0.w3.weight" in tensor_names

    # Check for expert tensors (layer 1, expert 3)
    assert "blk.1.ffn.experts.3.w1.weight" in tensor_names
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_toy_model.py -v
```

Expected: FAIL with "No module named 'convert.create_toy_model'"

**Step 3: Implement toy model generator**

Create `convert/create_toy_model.py` with implementation (see full content in appendix)

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_toy_model.py -v
```

Expected: PASS (both tests)

**Step 5: Generate a toy model for manual testing**

```bash
python -m convert.create_toy_model --output convert/toy_model.gguf
```

Expected: Creates ~2MB file at convert/toy_model.gguf

**Step 6: Commit toy model generator**

```bash
git add convert/create_toy_model.py tests/test_toy_model.py
git commit -m "feat: add toy GGUF model generator for testing"
```

---

## Task 3: Implement GGUF Metadata Parser

**Files:**
- Modify: `convert/gguf_to_od_moe.py:19-44`
- Create: `tests/test_gguf_parser.py`

**Step 1: Write test for metadata parsing**

Create `tests/test_gguf_parser.py` with tests (see appendix)

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_gguf_parser.py -v
```

Expected: FAIL (placeholder returns wrong structure)

**Step 3: Implement real metadata parser**

Replace the placeholder `parse_gguf_metadata` function (see implementation in appendix)

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gguf_parser.py -v
```

Expected: PASS (all tests)

**Step 5: Commit metadata parser**

```bash
git add convert/gguf_to_od_moe.py tests/test_gguf_parser.py
git commit -m "feat: implement real GGUF metadata parser"
```

---

## Task 4: Implement Base Model Extraction

**Files:**
- Modify: `convert/gguf_to_od_moe.py:47-69`
- Create: `tests/test_base_extraction.py`

**Step 1: Write test for base model extraction**

Create `tests/test_base_extraction.py` with tests

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_base_extraction.py -v
```

Expected: FAIL (placeholder doesn't create safetensors)

**Step 3: Implement base model extraction**

Replace `extract_base_model` function with safetensors implementation

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_base_extraction.py -v
```

Expected: PASS (all tests)

**Step 5: Commit base model extraction**

```bash
git add convert/gguf_to_od_moe.py tests/test_base_extraction.py
git commit -m "feat: implement base model extraction to safetensors"
```

---

## Task 5: Implement Expert Extraction

**Files:**
- Modify: `convert/gguf_to_od_moe.py:72-117`
- Create: `tests/test_expert_extraction.py`

**Step 1: Write test for expert extraction**

Create `tests/test_expert_extraction.py` with comprehensive tests

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_expert_extraction.py -v
```

Expected: FAIL (placeholder uses .npy, not safetensors)

**Step 3: Implement expert extraction**

Replace `extract_experts` function with safetensors implementation

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_expert_extraction.py -v
```

Expected: PASS (all tests)

**Step 5: Test with toy model**

```bash
python -m convert.gguf_to_od_moe \
  --input convert/toy_model.gguf \
  --output /tmp/toy_output \
  --num-layers 2 \
  --num-experts 4
```

Expected: Creates base_model/ and experts/ directories with safetensors files

**Step 6: Commit expert extraction**

```bash
git add convert/gguf_to_od_moe.py tests/test_expert_extraction.py
git commit -m "feat: implement expert extraction to safetensors with registry"
```

---

## Task 6: Create Conversion Documentation

**Files:**
- Create: `convert/README.md`

**Step 1: Write comprehensive README**

Create detailed documentation covering usage, architecture, file formats, testing

**Step 2: Commit documentation**

```bash
git add convert/README.md
git commit -m "docs: add comprehensive GGUF converter documentation"
```

---

## Task 7: Add Progress Bars

**Files:**
- Modify: `convert/gguf_to_od_moe.py`

**Step 1: Add overall conversion progress**

Modify the `convert_gguf_to_od_moe` function to show nested progress

**Step 2: Test progress bars**

```bash
python -m convert.gguf_to_od_moe \
  --input convert/toy_model.gguf \
  --output /tmp/final_test \
  --num-layers 2 \
  --num-experts 4
```

Expected: See nested progress bars (overall + per-step)

**Step 3: Commit progress improvements**

```bash
git add convert/gguf_to_od_moe.py
git commit -m "feat: add comprehensive progress bars to converter"
```

---

## Task 8: End-to-End Integration Test

**Files:**
- Create: `tests/test_full_conversion.py`

**Step 1: Write integration test**

Create full pipeline test

**Step 2: Run integration tests**

```bash
pytest tests/test_full_conversion.py -v
```

Expected: PASS (both tests)

**Step 3: Commit integration tests**

```bash
git add tests/test_full_conversion.py
git commit -m "test: add end-to-end integration tests for converter"
```

---

## Task 9: Final Verification

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Add artifacts to gitignore**

```bash
echo "convert/toy_model.gguf" >> .gitignore
echo "convert/toy_output/" >> .gitignore
git add .gitignore
git commit -m "chore: add converter artifacts to gitignore"
```

**Step 3: Notify completion**

```bash
clawdbot gateway wake 'Converter complete' --mode now
```
