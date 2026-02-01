"""
Tests for base model extraction from GGUF.

Validates that the extractor can:
1. Create safetensors files for base model components
2. Extract embeddings with correct shape
3. Extract attention layers correctly
4. Exclude expert weights from base model
"""

import pytest
from pathlib import Path
from convert.create_toy_model import create_toy_gguf_model
from convert.gguf_to_od_moe import extract_base_model
from safetensors import safe_open


@pytest.fixture
def toy_gguf(tmp_path):
    """Create a toy GGUF file for testing"""
    gguf_path = tmp_path / "test.gguf"
    create_toy_gguf_model(
        output_path=gguf_path,
        num_layers=2,
        num_experts=4,
        dim=512
    )
    return gguf_path


def test_extract_base_model_creates_safetensors(toy_gguf, tmp_path):
    """Verify extract_base_model creates all required safetensors files."""
    output_dir = tmp_path / "output"
    extract_base_model(toy_gguf, output_dir)

    # Check base_dir exists
    base_dir = output_dir / "base_model"
    assert base_dir.exists()
    assert base_dir.is_dir()

    # Check all required files exist
    assert (base_dir / "embeddings.safetensors").exists()
    assert (base_dir / "attention_layers.safetensors").exists()
    assert (base_dir / "norms.safetensors").exists()
    assert (base_dir / "lm_head.safetensors").exists()


def test_extract_base_model_embeddings_shape(toy_gguf, tmp_path):
    """Verify embeddings tensor has correct shape."""
    output_dir = tmp_path / "output"
    extract_base_model(toy_gguf, output_dir)

    # Open embeddings.safetensors
    embeddings_path = output_dir / "base_model" / "embeddings.safetensors"
    with safe_open(embeddings_path, framework="numpy") as f:
        # Get token_embd.weight tensor
        token_embd = f.get_tensor("token_embd.weight")

        # GGUF format: token embeddings are [vocab_size, dim]
        # The toy model has vocab_size=1000, dim=512
        assert token_embd.shape == (1000, 512)


def test_extract_base_model_attention_layers(toy_gguf, tmp_path):
    """Verify attention layers are extracted correctly."""
    output_dir = tmp_path / "output"
    extract_base_model(toy_gguf, output_dir)

    # Open attention_layers.safetensors
    attention_path = output_dir / "base_model" / "attention_layers.safetensors"
    with safe_open(attention_path, framework="numpy") as f:
        # Check that attention tensors exist
        assert "blk.0.attn_q.weight" in f.keys()
        assert "blk.0.attn_k.weight" in f.keys()
        assert "blk.0.attn_v.weight" in f.keys()
        assert "blk.0.attn_output.weight" in f.keys()

        # Check shape: q should be (dim, dim) = (512, 512)
        attn_q = f.get_tensor("blk.0.attn_q.weight")
        assert attn_q.shape == (512, 512)


def test_extract_base_model_excludes_experts(toy_gguf, tmp_path):
    """Verify that expert weights are excluded from base model."""
    output_dir = tmp_path / "output"
    extract_base_model(toy_gguf, output_dir)

    # Open attention_layers.safetensors
    attention_path = output_dir / "base_model" / "attention_layers.safetensors"
    with safe_open(attention_path, framework="numpy") as f:
        # Get all keys
        all_keys = list(f.keys())

        # Filter for keys containing "experts"
        expert_keys = [k for k in all_keys if "experts" in k]

        # Assert no expert weights in base model
        assert len(expert_keys) == 0
