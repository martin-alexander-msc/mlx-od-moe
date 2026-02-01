"""
Integration test for GGUF → OD-MoE conversion.

Tests the complete pipeline:
1. Create toy GGUF model
2. Run full conversion
3. Verify all outputs exist and are valid
4. Verify expert registry is correct
5. Verify base model can be loaded
"""

import pytest
from pathlib import Path
import json
from convert.create_toy_model import create_toy_gguf_model
from convert.gguf_to_od_moe import convert_gguf_to_od_moe
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


def test_full_conversion_pipeline(toy_gguf, tmp_path):
    """Test complete GGUF → OD-MoE conversion end-to-end."""
    output_dir = tmp_path / "od_moe_output"

    # Run full conversion
    convert_gguf_to_od_moe(
        input_path=str(toy_gguf),
        output_dir=str(output_dir),
        num_layers=2,
        num_experts=4
    )

    # Verify directory structure
    assert output_dir.exists()
    assert (output_dir / "base_model").exists()
    assert (output_dir / "experts").exists()

    # Verify base model files
    base_dir = output_dir / "base_model"
    assert (base_dir / "embeddings.safetensors").exists()
    assert (base_dir / "attention_layers.safetensors").exists()
    assert (base_dir / "norms.safetensors").exists()
    assert (base_dir / "lm_head.safetensors").exists()
    assert (base_dir / "metadata.json").exists()

    # Verify expert files (2 layers × 4 experts = 8 files)
    expert_dir = output_dir / "experts"
    safetensors_files = list(expert_dir.glob("*.safetensors"))
    assert len(safetensors_files) == 8

    # Verify registry
    registry_path = expert_dir / "registry.json"
    assert registry_path.exists()

    with open(registry_path) as f:
        registry = json.load(f)

    assert len(registry) == 8
    assert "layer_00_expert_000" in registry
    assert "layer_01_expert_003" in registry


def test_converted_base_model_loadable(toy_gguf, tmp_path):
    """Verify that converted base model safetensors files can be loaded."""
    output_dir = tmp_path / "od_moe_output"

    convert_gguf_to_od_moe(
        input_path=str(toy_gguf),
        output_dir=str(output_dir),
        num_layers=2,
        num_experts=4
    )

    # Try loading each base model file
    base_dir = output_dir / "base_model"

    with safe_open(base_dir / "embeddings.safetensors", framework="numpy") as f:
        assert "token_embd.weight" in f.keys()
        token_embd = f.get_tensor("token_embd.weight")
        assert token_embd.shape == (1000, 512)  # vocab_size=1000, dim=512

    with safe_open(base_dir / "attention_layers.safetensors", framework="numpy") as f:
        assert "blk.0.attn_q.weight" in f.keys()
        assert "blk.1.attn_q.weight" in f.keys()

    with safe_open(base_dir / "norms.safetensors", framework="numpy") as f:
        assert "blk.0.attn_norm.weight" in f.keys()
        assert "blk.0.ffn_norm.weight" in f.keys()


def test_converted_experts_loadable(toy_gguf, tmp_path):
    """Verify that all converted expert files can be loaded."""
    output_dir = tmp_path / "od_moe_output"

    convert_gguf_to_od_moe(
        input_path=str(toy_gguf),
        output_dir=str(output_dir),
        num_layers=2,
        num_experts=4
    )

    expert_dir = output_dir / "experts"

    # Try loading all expert files
    for layer in range(2):
        for expert in range(4):
            expert_file = expert_dir / f"layer_{layer:02d}_expert_{expert:03d}.safetensors"

            with safe_open(expert_file, framework="numpy") as f:
                # Verify all expert weights exist
                assert "w1.weight" in f.keys()
                assert "w2.weight" in f.keys()
                assert "w3.weight" in f.keys()

                # Verify shapes (dim=512, ffn_dim=1024)
                w1 = f.get_tensor("w1.weight")
                w2 = f.get_tensor("w2.weight")
                w3 = f.get_tensor("w3.weight")

                assert w1.shape == (512, 1024)
                assert w2.shape == (1024, 512)
                assert w3.shape == (512, 1024)


def test_expert_registry_completeness(toy_gguf, tmp_path):
    """Verify expert registry contains complete metadata for all experts."""
    output_dir = tmp_path / "od_moe_output"

    convert_gguf_to_od_moe(
        input_path=str(toy_gguf),
        output_dir=str(output_dir),
        num_layers=2,
        num_experts=4
    )

    registry_path = output_dir / "experts" / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    # Check each expert's metadata
    for layer in range(2):
        for expert in range(4):
            key = f"layer_{layer:02d}_expert_{expert:03d}"

            assert key in registry
            expert_info = registry[key]

            # Verify all required fields
            assert "path" in expert_info
            assert "size" in expert_info
            assert "layer" in expert_info
            assert "expert_id" in expert_info
            assert "tensors" in expert_info

            # Verify values
            assert expert_info["layer"] == layer
            assert expert_info["expert_id"] == expert
            assert expert_info["size"] > 0
            assert set(expert_info["tensors"]) == {"w1.weight", "w2.weight", "w3.weight"}
