"""
Tests for GGUF metadata parser.

Validates that the parser can:
1. Extract model architecture metadata
2. List all tensors with their properties
3. Reject invalid GGUF files
"""

import pytest
from pathlib import Path
from convert.create_toy_model import create_toy_gguf_model
from convert.gguf_to_od_moe import parse_gguf_metadata


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


def test_parse_gguf_metadata_reads_architecture(toy_gguf):
    """Verify parser extracts architecture metadata correctly."""
    metadata = parse_gguf_metadata(toy_gguf)

    # Check basic architecture info
    assert metadata["architecture"] == "kimi-moe"
    assert metadata["num_layers"] == 2
    assert metadata["num_experts"] == 4
    assert metadata["dim"] == 512
    assert metadata["vocab_size"] == 1000


def test_parse_gguf_metadata_lists_tensors(toy_gguf):
    """Verify parser lists all tensors with their properties."""
    metadata = parse_gguf_metadata(toy_gguf)

    # Check that tensors dict exists
    assert "tensors" in metadata
    tensors = metadata["tensors"]

    # Check specific tensors exist
    assert "token_embd.weight" in tensors
    assert "blk.0.ffn.experts.0.w1.weight" in tensors

    # Check tensor properties
    for tensor_name, tensor_info in tensors.items():
        assert "shape" in tensor_info
        assert "dtype" in tensor_info
        assert "offset" in tensor_info
        assert isinstance(tensor_info["shape"], list)
        assert isinstance(tensor_info["dtype"], str)


def test_parse_gguf_metadata_rejects_non_gguf(tmp_path):
    """Verify parser rejects invalid GGUF files."""
    fake_file = tmp_path / "fake.gguf"
    fake_file.write_bytes(b"NOT A GGUF FILE")

    with pytest.raises(ValueError, match="Not a GGUF file"):
        parse_gguf_metadata(fake_file)
