"""
Tests for toy GGUF model generator.

Verifies that the toy model:
1. Creates a valid GGUF file
2. Contains expected tensor structure (28 layers × 384 experts in full model)
"""

import pytest
from pathlib import Path
import gguf
from convert.create_toy_model import create_toy_gguf_model


def test_create_toy_model_generates_gguf(tmp_path):
    """Test that create_toy_gguf_model generates a valid GGUF file."""
    output_file = tmp_path / "test_toy.gguf"

    # Create toy model with minimal params
    create_toy_gguf_model(
        output_path=output_file,
        num_layers=2,
        num_experts=4
    )

    # Assert file exists
    assert output_file.exists(), "GGUF file should be created"

    # Verify it's valid GGUF using GGUFReader
    reader = gguf.GGUFReader(output_file)
    assert reader.fields is not None, "GGUF reader should parse the file successfully"


def test_toy_model_has_expected_tensors(tmp_path):
    """Test that toy model contains all expected tensors with correct naming."""
    output_file = tmp_path / "test_toy.gguf"

    # Create toy model with specific dimensions
    create_toy_gguf_model(
        output_path=output_file,
        num_layers=2,
        num_experts=4,
        dim=512,
        ffn_dim=1024
    )

    # Read with GGUFReader and get tensor names
    reader = gguf.GGUFReader(output_file)
    tensor_names = {tensor.name for tensor in reader.tensors}

    # Assert base model tensors exist
    assert "token_embd.weight" in tensor_names, "Should have token embeddings"
    assert "output.weight" in tensor_names, "Should have output layer"

    # Assert layer tensors exist (layer 0)
    assert "blk.0.attn_q.weight" in tensor_names, "Should have attention query weights"
    assert "blk.0.attn_norm.weight" in tensor_names, "Should have attention norm"

    # Assert expert tensors exist (layer 0, expert 0)
    assert "blk.0.ffn.experts.0.w1.weight" in tensor_names, "Should have expert w1"
    assert "blk.0.ffn.experts.0.w2.weight" in tensor_names, "Should have expert w2"
    assert "blk.0.ffn.experts.0.w3.weight" in tensor_names, "Should have expert w3"

    # Assert layer 1 expert 3 exists
    assert "blk.1.ffn.experts.3.w1.weight" in tensor_names, "Should have layer 1 expert 3 w1"
