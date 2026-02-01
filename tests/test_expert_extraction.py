"""
Tests for expert extraction from GGUF.

Validates that the extractor can:
1. Create safetensors files for each expert
2. Create registry.json with all metadata
3. Extract correct expert weights (w1, w2, w3)
4. Save proper metadata for each expert
"""

import pytest
import json
from pathlib import Path
from convert.create_toy_model import create_toy_gguf_model
from convert.gguf_to_od_moe import extract_experts
from safetensors import safe_open


@pytest.fixture
def toy_gguf(tmp_path):
    """Create a toy GGUF file for testing"""
    gguf_path = tmp_path / "test.gguf"
    create_toy_gguf_model(
        output_path=gguf_path,
        num_layers=2,
        num_experts=4,
        dim=512,
        ffn_dim=1024
    )
    return gguf_path


def test_extract_experts_creates_safetensors(toy_gguf, tmp_path):
    """Verify extract_experts creates safetensors files for all experts."""
    output_dir = tmp_path / "output"
    extract_experts(toy_gguf, output_dir, num_layers=2, num_experts=4)

    # Check expert_dir exists
    expert_dir = output_dir / "experts"
    assert expert_dir.exists()
    assert expert_dir.is_dir()

    # Check specific files exist (2 layers × 4 experts = 8 files)
    assert (expert_dir / "layer_00_expert_000.safetensors").exists()
    assert (expert_dir / "layer_00_expert_003.safetensors").exists()
    assert (expert_dir / "layer_01_expert_000.safetensors").exists()
    assert (expert_dir / "layer_01_expert_003.safetensors").exists()

    # Count total safetensors files
    safetensors_files = list(expert_dir.glob("*.safetensors"))
    assert len(safetensors_files) == 8


def test_extract_experts_creates_registry(toy_gguf, tmp_path):
    """Verify registry.json is created with all expert metadata."""
    output_dir = tmp_path / "output"
    extract_experts(toy_gguf, output_dir, num_layers=2, num_experts=4)

    # Check registry.json exists
    registry_path = output_dir / "experts" / "registry.json"
    assert registry_path.exists()

    # Load and validate registry
    with open(registry_path) as f:
        registry = json.load(f)

    # Should have 8 entries (2 layers × 4 experts)
    assert len(registry) == 8

    # Check specific keys exist
    assert "layer_00_expert_000" in registry
    assert "layer_01_expert_003" in registry


def test_extract_experts_correct_weights(toy_gguf, tmp_path):
    """Verify expert safetensors contain correct weight tensors."""
    output_dir = tmp_path / "output"
    extract_experts(toy_gguf, output_dir, num_layers=2, num_experts=4)

    # Open a specific expert file
    expert_path = output_dir / "experts" / "layer_00_expert_000.safetensors"
    with safe_open(expert_path, framework="numpy") as f:
        # Check that all required keys exist
        assert "w1.weight" in f.keys()
        assert "w2.weight" in f.keys()
        assert "w3.weight" in f.keys()

        # Check shapes (dim=512, ffn_dim=1024)
        w1 = f.get_tensor("w1.weight")
        w2 = f.get_tensor("w2.weight")
        w3 = f.get_tensor("w3.weight")

        # w1: (dim, ffn_dim) = (512, 1024)
        assert w1.shape == (512, 1024)

        # w2: (ffn_dim, dim) = (1024, 512)
        assert w2.shape == (1024, 512)

        # w3: (dim, ffn_dim) = (512, 1024)
        assert w3.shape == (512, 1024)


def test_extract_experts_registry_metadata(toy_gguf, tmp_path):
    """Verify registry entries contain all required metadata fields."""
    output_dir = tmp_path / "output"
    extract_experts(toy_gguf, output_dir, num_layers=2, num_experts=4)

    # Load registry
    registry_path = output_dir / "experts" / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    # Get a specific expert's metadata
    expert_info = registry["layer_00_expert_000"]

    # Check all required keys exist
    assert "path" in expert_info
    assert "size" in expert_info
    assert "layer" in expert_info
    assert "expert_id" in expert_info

    # Verify values
    assert expert_info["layer"] == 0
    assert expert_info["expert_id"] == 0
    assert expert_info["size"] > 0  # File should have some size
